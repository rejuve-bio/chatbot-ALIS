"""Current clinician-appraisal counts, revised by native Omega PLN.

Each distinct clinician contributes one current categorical appraisal. The
confidence quantity measures decisive-review volume, not study confidence.
Reference provenance remains visible separately. Reasons are shared notes.
The previous reference-group adapter is retained for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from app.services.feedback_pln import execute_native, NativePLNError
from app.services.feedback_reason_schema import ASPECTS

REFERENCE_ADAPTER = "reason-grounded-appraisal-v2"
REFERENCE_SEMANTICS = (
    "Strength is support for the scoped research-retention appraisal. Reviews sharing a public "
    "reference form one contribution: its strength is the fraction of decisive judgments supporting "
    "the assertion. Each reference component has one appraisal unit. Confidence is G/(G+1), "
    "where G is the number of decisive reference components; it measures recorded appraisal "
    "coverage, not study quality, independent participants, reviewer reliability or clinical certainty. "
    "Unreferenced reviews remain visible without numerical weight. Different references can still be dependent. These quantities are not clinically calibrated."
)
ADAPTER = "clinician-appraisal-count-v3"
SEMANTICS = (
    "Strength is the proportion of current decisive clinician judgments supporting the scoped "
    "research-retention claim. Each authenticated clinician contributes at most one current "
    "judgment; replacements and withdrawals rebuild the appraisal. Review-count confidence is "
    "N/(N+1), where N counts current Support/Oppose judgments. It measures review volume under "
    "the declared unit-count convention, not scientific certainty or reviewer independence. "
    "Uncertain judgments remain visible without a binary premise. Citations and shared notes "
    "add no numerical weight. Accounts may be correlated; independence is not estimated."
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS belief_assessments (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 candidate_id TEXT NOT NULL REFERENCES candidates(id),
 version_id TEXT NOT NULL REFERENCES versions(id),
 input_digest TEXT NOT NULL, result_json TEXT NOT NULL,
 trigger TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS belief_version ON belief_assessments(candidate_id,version_id,id);
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def reference_ids(text):
    value = str(text or "")
    refs = {"PMID:" + x for x in re.findall(r"\bPMID\s*:?\s*(\d+)\b", value, re.I)}
    refs.update("PMID:" + x for x in re.findall(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", value, re.I))
    refs.update(x.upper() for x in re.findall(r"\bNCT\d{8}\b", value, re.I))
    refs.update("DOI:" + x.casefold().rstrip(".,;)") for x in re.findall(r"\b10\.\d{4,9}/[^\s\"<>]+", value, re.I))
    if not refs:
        refs.update("URL:" + x.rstrip("/.,;)").casefold() for x in re.findall(r"https?://[^\s\"<>]+", value, re.I))
    return sorted(refs)


def observations(reviews, aspect=None, source_families=None):
    items = []
    for review in reviews:
        analysis = (review.get("reason_analysis") or {}).get("analysis", {})
        notes_only = review.get("reason_mode") == "note"
        confirmed = bool(review.get("analysis_confirmed")) and not notes_only
        rows = analysis.get("assessments", []) if confirmed else []
        if aspect:
            row = next((r for r in rows if r["aspect"] == aspect), None)
            if row is None:
                continue
            decision = row["decision"]
            # Enforce the boundary even for previously stored interpretations.
            if row["basis"] == "evidence_gap" and aspect in ("benefit_claim", "safety"):
                decision = "uncertain"
            checks = row["source_checks"]
        else:
            decision = review["decision"]
            checks = [check for row in rows for check in row["source_checks"]]
        # Preserve the reference handling of older interpreted reviews. New
        # shared notes are display-only, including any identifiers in the note.
        roots = set(reference_ids(review.get("citation")))
        if not notes_only:
            roots.update(reference_ids(review.get("rationale")))
        for check in checks:
            if not check["source_id"].startswith("handoff:"):
                roots.update(reference_ids(check["source_id"]) or [check["source_id"].upper()])
        for root in list(roots):
            roots.update((source_families or {}).get(root, []))
        # Uncited reviews are visible, but cannot pretend to be independently
        # grounded units merely because names or wording differ.
        if not roots:
            roots.add("UNREFERENCED-APPRAISAL")
        items.append(dict(review_id=review["id"], reviewer_key=review["reviewer_key"], decision=decision,
                          roots=sorted(roots), interpreted=confirmed,
                          reason=review["rationale"], citation=review.get("citation", "")))
    return items


def group_observations(items):
    groups = []
    for item in items:
        matches = [g for g in groups if set(g["roots"]) & set(item["roots"])]
        group = dict(roots=set(item["roots"]), items=[item])
        for previous in matches:
            group["roots"].update(previous["roots"])
            group["items"].extend(previous["items"])
            groups.remove(previous)
        groups.append(group)
    result = []
    for group in groups:
        decisive = [x for x in group["items"] if x["decision"] in ("support", "oppose")]
        support = sum(x["decision"] == "support" for x in decisive)
        result.append(dict(roots=sorted(group["roots"]), review_ids=sorted(x["review_id"] for x in group["items"]),
                           support=support, oppose=len(decisive)-support,
                           uncertain=len(group["items"])-len(decisive),
                           strength=None, confidence=None,
                           conflict=bool(support and support < len(decisive))))
    return sorted(result, key=lambda g: (g["roots"], g["review_ids"]))


def assess_reference_groups(reviews, source_families=None):
    """Archived v2 calculation; current appraisals use assess() below."""
    identities = [r["reviewer_key"] for r in reviews]
    if len(set(identities)) != len(identities):
        raise ValueError("Only one current judgment per authenticated clinician is allowed.")
    scopes = {"retention": observations(reviews, source_families=source_families)}
    for aspect in ASPECTS:
        items = observations(reviews, aspect, source_families)
        if items:
            scopes[aspect] = items
    assessments = {}
    lines = ["!(import! &self lib_pln)"]
    expected = {}
    for scope, items in scopes.items():
        groups = group_observations(items)
        decisive = [g for g in groups if g["support"] + g["oppose"] and "UNREFERENCED-APPRAISAL" not in g["roots"]]
        assessment = dict(strength=None, confidence=None, support=sum(g["support"] for g in groups),
                          oppose=sum(g["oppose"] for g in groups), uncertain=sum(g["uncertain"] for g in groups),
                          ground_count=len(decisive), groups=groups,
                          conflict=any(g["conflict"] for g in groups) or
                          bool(any(g["support"] for g in groups) and any(g["oppose"] for g in groups)))
        assessments[scope] = assessment
        for i, group in enumerate(decisive):
            count = group["support"] + group["oppose"]
            # 0/1 are observed categorical judgments. Unit weight is the
            # explicitly declared provenance-group counting convention.
            lines.append(f'(= ({scope}_group_{i}) (stv (/ {float(group["support"])} {float(count)}) (Truth__w2c 1.0)))')
            lines.append(f'!(let $t ({scope}_group_{i}) (GROUP {scope} {i} $t))')
            term = f"({scope}_group_{i})" if i == 0 else f"(Truth__Revision ({scope}_combined_{i-1}) ({scope}_group_{i}))"
            lines.append(f"(= ({scope}_combined_{i}) {term})")
        if decisive:
            lines.append(f'!(let $t ({scope}_combined_{len(decisive)-1}) (BELIEF {scope} $t))')
            expected[scope] = (sum(g["support"]/(g["support"]+g["oppose"]) for g in decisive)/len(decisive),
                               len(decisive)/(len(decisive)+1))
    result = dict(adapter=REFERENCE_ADAPTER, semantics=REFERENCE_SEMANTICS, calibrated=False, scientific_confidence=None,
                  input_review_ids=[r["id"] for r in reviews], input_digest=digest([REFERENCE_ADAPTER, reviews, source_families]),
                  confirmed_reason_reviews=sum(bool(r.get("analysis_confirmed")) for r in reviews),
                  native=None, pln_used=False, aspect_assessments={k:v for k,v in assessments.items() if k != "retention"})
    result.update(assessments["retention"])
    if expected:
        execution = execute_native("\n".join(lines) + "\n")
        pairs = re.findall(r"\(BELIEF ([a-z_]+) \(stv ([-+\deE.]+) ([-+\deE.]+)\)\)", execution["stdout"])
        if len(pairs) != len(expected) or {p[0] for p in pairs} != set(expected):
            raise NativePLNError("No unique native appraisal result; no replacement belief saved.")
        for scope, s, c in pairs:
            s, c = float(s), float(c)
            if not math.isclose(s, expected[scope][0], abs_tol=1e-10) or not math.isclose(c, expected[scope][1], abs_tol=1e-10):
                raise NativePLNError("Native appraisal result violated the declared counting model.")
            assessments[scope].update(strength=s, confidence=c)
        for scope, i, s, c in re.findall(r"\(GROUP ([a-z_]+) (\d+) \(stv ([-+\deE.]+) ([-+\deE.]+)\)\)", execution["stdout"]):
            group = [g for g in assessments[scope]["groups"] if g["support"]+g["oppose"] and "UNREFERENCED-APPRAISAL" not in g["roots"]][int(i)]
            group.update(strength=float(s), confidence=float(c))
        result.update(assessments["retention"], native=execution, pln_used=True)
    result["status"] = "unreviewed" if not reviews else "needs_reference" if result["strength"] is None and any(r["decision"] in ("support","oppose") for r in reviews) else "uncertain" if result["strength"] is None else "disagreement" if result["conflict"] else "reviewed"
    return result


def assess(reviews, source_families=None):
    """Count current clinician appraisals; keep paper provenance separate.

    Binary values encode the selected Support/Oppose judgment. One unit per
    decisive reviewer is an explicit count convention, not an estimate of
    reviewer reliability or independent biological evidence.
    """
    identities = [r["reviewer_key"] for r in reviews]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("Only one current judgment per authenticated clinician is allowed.")
    if any(r["decision"] not in ("support", "oppose", "uncertain") for r in reviews):
        raise ValueError("Only current Support/Oppose/Uncertain judgments can enter the appraisal.")
    decisive = [r for r in reviews if r["decision"] in ("support", "oppose")]
    support = sum(r["decision"] == "support" for r in decisive)
    n = len(decisive)
    # Provenance is exported for inspection, with no reference-group weighting.
    # Neither old Omega reason interpretations nor note identifiers add premises.
    items = observations([dict(r, reason_mode="note") for r in reviews], source_families=source_families)
    groups = group_observations(items)
    reference_count = sum("UNREFERENCED-APPRAISAL" not in g["roots"] and
                          bool(g["support"] + g["oppose"]) for g in groups)
    result = dict(
        adapter=ADAPTER, semantics=SEMANTICS, calibrated=False,
        scientific_confidence=None, independence_assessed=False,
        confidence_scope="decisive-review-count",
        count_profile={"unit_weight": 1.0, "native_k": 1.0},
        strength=None, confidence=None, support=support, oppose=n-support,
        uncertain=len(reviews)-n, reviewer_count=len(reviews), decisive_count=n,
        reference_group_count=reference_count, ground_count=reference_count, groups=groups,
        conflict=bool(support and support<n),
        status="unreviewed" if not reviews else "uncertain" if not n else
               "disagreement" if support and support<n else "reviewed",
        input_review_ids=[r["id"] for r in reviews],
        input_digest=digest([ADAPTER, reviews, source_families]),
        confirmed_reason_reviews=0, aspect_assessments={}, native=None, pln_used=False,
    )
    if not decisive:
        return result
    lines = ["!(import! &self lib_pln)"]
    for i, review in enumerate(decisive):
        value = "1.0" if review["decision"] == "support" else "0.0"
        lines.append(f"(= (appraisal_review_{i}) (stv {value} (Truth__w2c 1.0)))")
        term = f"(appraisal_review_{i})" if i==0 else f"(Truth__Revision (appraisal_combined_{i-1}) (appraisal_review_{i}))"
        lines.append(f"(= (appraisal_combined_{i}) {term})")
    lines.append(f"!(let $t (appraisal_combined_{n-1}) (BELIEF retention $t))")
    execution = execute_native("\n".join(lines) + "\n")
    pairs = re.findall(r"\(BELIEF retention \(stv ([-+\deE.]+) ([-+\deE.]+)\)\)", execution["stdout"])
    if len(pairs) != 1:
        raise NativePLNError("No unique native review-count result; no replacement appraisal saved.")
    strength, confidence = map(float, pairs[0])
    if not math.isclose(strength, support/n, abs_tol=1e-10) or not math.isclose(confidence, n/(n+1), abs_tol=1e-10):
        raise NativePLNError("Native appraisal result violated the declared review-count model.")
    result.update(strength=strength, confidence=confidence, native=execution, pln_used=True)
    return result


class AppraisalBeliefs:
    def belief_assessment(self, candidate):
        row = self.db.execute('SELECT * FROM belief_assessments WHERE candidate_id=? AND version_id=? ORDER BY id DESC LIMIT 1',
                              (candidate['id'], candidate['current_version'])).fetchone()
        if row is None:
            return None
        return dict(json.loads(row['result_json']), assessment_id=row['id'], created_at=row['created_at'], trigger=row['trigger'])

    def _rebuild_belief(self, candidate, trigger):
        from app.services.feedback_store import now
        reviews = self.current_reviews(candidate)
        families = {}
        for record in self.source_records(candidate):
            study = record['facts'].get('study_id')
            if study:
                for key in reference_ids(record['source']['id']) or [record['source']['id'].upper()]:
                    families[key] = ['STUDY:' + study.upper()]
        key = digest([ADAPTER, reviews, families])
        previous = self.belief_assessment(candidate)
        if previous and previous['input_digest'] == key:
            return previous
        result = assess(reviews, families)
        cursor = self.db.execute('INSERT INTO belief_assessments(candidate_id,version_id,input_digest,result_json,trigger,created_at) VALUES (?,?,?,?,?,?)',
                                 (candidate['id'], candidate['current_version'], key, json.dumps(result,sort_keys=True,ensure_ascii=False), trigger, now()))
        return dict(result, assessment_id=cursor.lastrowid, trigger=trigger)

    def belief_history(self, candidate):
        return [dict(json.loads(r['result_json']), assessment_id=r['id'], version_id=r['version_id'],
                     created_at=r['created_at'], trigger=r['trigger']) for r in self.db.execute(
            'SELECT * FROM belief_assessments WHERE candidate_id=? ORDER BY id', (candidate['id'],))]
