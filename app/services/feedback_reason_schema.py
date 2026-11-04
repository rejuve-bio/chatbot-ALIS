"""Strict, non-numeric contract for Omega's proposed review interpretation."""
from __future__ import annotations

import hashlib
import json
import copy
import re
try:
    from app.services.feedback_source_schema import SOURCE_SCHEMA, validate_source_assessments
except ModuleNotFoundError:
    from source_schema import SOURCE_SCHEMA, validate_source_assessments

POLICY = "source-grounded-review-v4"
DECISIONS = ("support", "oppose", "uncertain")
ASPECTS = ("evidence_sufficiency", "human_applicability", "benefit_claim", "safety", "patient_suitability")
BASES = ("reported_effect", "evidence_gap", "applicability_mismatch", "reviewer_judgment", "safety_concern")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def text(maximum):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["summary", "proposed_decision", "reviewer_excerpt", "assessments", "limitations", "source_assessments"],
    "properties": {
        "summary": text(700), "proposed_decision": {"type": "string", "enum": list(DECISIONS)},
        "reviewer_excerpt": text(1000),
        "source_assessments": {"type":"array", "maxItems":8, "items":SOURCE_SCHEMA},
        "assessments": {"type": "array", "minItems": 0, "maxItems": 5, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["aspect", "decision", "basis", "explanation", "reviewer_excerpt", "source_checks"],
            "properties": {"aspect": {"type": "string", "enum": list(ASPECTS)},
                "decision": {"type": "string", "enum": list(DECISIONS)},
                "basis": {"type": "string", "enum": list(BASES)},
                "explanation": text(500), "reviewer_excerpt": text(1000),
                "source_checks": {"type": "array", "maxItems": 3, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["source_id", "quote"], "properties": {"source_id": text(70), "quote": text(1000)}}}}}},
        "limitations": {"type": "array", "maxItems": 6, "items": text(350)}}}


def validate_review_reason(reason):
    """Catch judgment/citation input at the reason prompt before a paid call."""
    if (not isinstance(reason,str) or not any(c.isalpha() for c in reason)
            or reason.strip().casefold() in DECISIONS
            or re.fullmatch(r'(?:PMID\s*:?\s*\d+|NCT\d{8}|https?://\S+)',reason.strip(),re.I)):
        raise ValueError('Please write why you chose that judgment. A judgment number or reference alone is not a review reason.')
    return reason


def schema_for(context):
    """Bind echoed reason fields to trusted input, without relaxing validation."""
    reason=validate_review_reason(context['review']['reason'])[:1000]
    schema=copy.deepcopy(SCHEMA)
    schema['properties']['reviewer_excerpt']['enum']=[reason]
    schema['properties']['assessments']['items']['properties']['reviewer_excerpt']['enum']=[reason]
    return schema


def validate(result, context):
    def object_keys(value, expected):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError("Omega interpretation has unexpected or missing fields.")

    def string(value, maximum):
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError("Omega interpretation contains invalid text.")

    object_keys(result, SCHEMA["required"])
    string(result["summary"], 700)
    if result["proposed_decision"] not in DECISIONS:
        raise ValueError("Omega returned an invalid proposed decision.")
    rationale = context["review"]["reason"]
    string(result["reviewer_excerpt"], 1000)
    if result["reviewer_excerpt"] not in rationale:
        raise ValueError("Omega's interpretation does not quote the reviewer's actual reason.")
    assessments = result["assessments"]
    if not isinstance(assessments, list) or not 0 <= len(assessments) <= 5:
        raise ValueError("Omega must identify the specific assessment affected by the reason.")
    sources = {s["id"]: s["text"] for s in context["sources"]}
    seen = set()
    for row in assessments:
        object_keys(row, ("aspect", "decision", "basis", "explanation", "reviewer_excerpt", "source_checks"))
        if row["aspect"] not in ASPECTS or row["aspect"] in seen or row["decision"] not in DECISIONS:
            raise ValueError("Omega returned an invalid or repeated claim assessment.")
        seen.add(row["aspect"])
        if row["basis"] not in BASES:
            raise ValueError("Unknown basis for the reason interpretation.")
        if row["aspect"] in ("benefit_claim", "safety") and row["basis"] == "evidence_gap" and row["decision"] != "uncertain":
            raise ValueError("Missing evidence cannot become a positive or negative biological effect/safety premise.")
        string(row["explanation"], 500)
        string(row["reviewer_excerpt"], 1000)
        if row["reviewer_excerpt"] not in rationale:
            raise ValueError("An assessment is not anchored to the supplied reviewer reason.")
        if not isinstance(row["source_checks"], list) or len(row["source_checks"]) > 3:
            raise ValueError("Invalid source checks.")
        for check in row["source_checks"]:
            object_keys(check, ("source_id", "quote"))
            string(check["source_id"], 70)
            string(check["quote"], 1000)
            if check["source_id"] not in sources or check["quote"] not in sources[check["source_id"]]:
                raise ValueError("Omega cited a quote that is not in the inspected source packet.")
    if not isinstance(result["limitations"], list) or len(result["limitations"]) > 6:
        raise ValueError("Invalid interpretation limitations.")
    for limitation in result["limitations"]:
        string(limitation, 350)
    validate_source_assessments(result['source_assessments'],context)
    return result


def apply_reason_policy(proposal):
    """A typed evidence-gap rule; no numerical confidence is assigned here."""
    result=copy.deepcopy(proposal)
    changes=[]
    kept=[]
    for row in result.get('source_assessments',[]):
        if str(row.get('source_id','')).startswith('handoff:') and row.get('effect') is None:
            changes.append({'field':'source_assessments','rule':'upstream-summary-is-not-study-data','source_id':row['source_id']})
        else:kept.append(row)
    result['source_assessments']=kept
    reason=str(result.get('reviewer_excerpt','')).casefold()
    if not any(term in reason for term in ('this patient','my patient','our patient','this case','this person')):
        if any(r.get('aspect')=='patient_suitability' for r in result.get('assessments',[])):
            result['assessments']=[r for r in result['assessments'] if r.get('aspect')!='patient_suitability']
            changes.append({'field':'assessments','rule':'study-population-is-not-patient-eligibility'})
    for row in result.get('source_assessments',[]):
        effect=row.get('effect')
        if not isinstance(effect,dict):continue
        for key in ('estimate','ci_lower','ci_upper','ci_level'):
            raw=effect.get(key)
            if isinstance(raw,str):
                normalized=raw.strip().removesuffix('%').strip()
                if raw!=normalized:
                    effect[key]=normalized
                    changes.append({'field':key,'rule':'reported-literal-formatting','from':raw,'to':normalized})
    for row in result.get('assessments',[]):
        if (row.get('aspect') in ('benefit_claim','safety') and row.get('basis')=='evidence_gap'
                and row.get('decision')!='uncertain'):
            changes.append({'aspect':row['aspect'],'rule':'missing-evidence-is-not-effect-evidence',
                            'from':row['decision'],'to':'uncertain'})
            row['decision']='uncertain'
            row['explanation']='An evidence gap leaves this effect/safety claim unresolved; it does not establish a positive or negative effect. Original interpretation: '+str(row.get('explanation',''))[:320]
    return result,changes
