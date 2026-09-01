import logging
from datetime import datetime

from app.services.codebook import (
    get_label,
    decode_questionnaire_value,
    is_questionnaire,
    get_force_included_variables,
)
from app.services.backend_api import fetch_longitudinal
from app.services.llm_service import call_llm

logger = logging.getLogger(__name__)


def _format_date(date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return date_str


LONGITUDINAL_SYSTEM_PROMPT = """
You are an AI clinical assistant. The data provided has already been fetched and verified — always answer from it. Never use your own training knowledge to add values, invent trends, or fill gaps.

RULES:
- Every value, date, and trend you state must come directly from the context data.
- Do not mention any platform, product, or system name.
- Do not explain what measurements are. Do not give generic medical information.

WHEN TO USE A TABLE:
- Use a markdown table ONLY when a biomarker has 2 or more time points to display.
- Use a markdown table for life events ONLY if the question explicitly asks to list events.
- Do NOT use tables for single values, trend sentences, before/after comparisons, or narrative observations.

FORMAT for a biomarker with multiple time points:

## Biomarker Name (CODE)
| Date | Value | Change |
|------|-------|--------|
| DATE | VALUE | — or +X.X / -X.X |
Trend: one sentence on direction and magnitude.

FORMAT for a biomarker with only one data point:
State it as plain text: "Biomarker Name (CODE): VALUE as of DATE. Insufficient history to determine a trend."

FORMAT for before/after an event:
Use plain bullet points:
- Before [event] (DATE): VALUE
- After [event] (DATE): VALUE

FORMAT for biological age with multiple points:

## Biological Age Over Time
| Date | Biological Age | Chronological Age | Delta |
|------|---------------|-------------------|-------|
| DATE | X | Y | Z |
Trend: one sentence on overall direction.

- Refer to the patient by name if available in the profile; otherwise use their ID.
"""


def _extract_variables_with_llm(
    question: str,
    available_variables: dict[str, str],
    llm_generate,
    prior_question: str = None,
) -> tuple[list[str], list[str], bool, bool]:
    """Returns (biomarker_codes, pc_names, needs_full_biomarkers, needs_disease_risks)."""
    var_list = "\n".join([
        f"{code}: {label}"
        for code, label in available_variables.items()
    ])

    context_line = ""
    if prior_question:
        context_line = f"\nPrevious question (for context): {prior_question}\n"

    prompt = f"""You are deciding whether this clinical question needs its own
extra data fetched from the patient's longitudinal record, on top of what's
already available.

Available biomarker variables:
{var_list}

Available PC values: PC1 through PC59
{context_line}
Clinical question: {question}

The patient's current biomarker values, PC contributions, and disease risks
are ALREADY included elsewhere in this conversation's context. You are not
answering the question — only deciding whether fetching a SPECIFIC named
variable's own record from the longitudinal API is genuinely needed beyond
that.

The principle: only select a variable if the question calls out that exact
variable by name — a named biomarker (blood pressure, glucose, pulse...) or
a specific PC (PC1, PC32...) — whether it's asking for that variable's
current value or how it changed over time. If the question is broad or
general instead — "what are the biomarkers," "show all PCs," an overview,
a diagnosis/history question ("does this patient have asthma"), a ranking
("top 3 PCs") — return EMPTY lists for both. None of those name a specific
variable, and the broader information they need is already in the main
patient context; this step exists only for named, specific lookups.
Aging/delta/biological-age questions also return empty — those come from
clock_results automatically, not from this lookup.

When genuinely unsure, return empty lists — the main context already
covers general and single-value lookups; this is only for the narrower
case of a clearly named variable.

Separately: the patient's context ALSO has a full "Biomarkers" section
(every lab/vital/questionnaire value, ~70 lines) and a full "Disease
Risks" section (every disease risk with evidence score and mechanisms).
Both get included by default, but reading through them costs real time
for questions that don't need them. Decide honestly whether THIS question
needs each one:
- needs_full_biomarkers: true if the question is actually about the
  patient's lab values, vitals, medical history/questionnaire answers,
  or asks to see biomarkers broadly (even without naming one specifically,
  e.g. "what are this patient's biomarkers"). false for questions purely
  about PCs, disease risk rankings, or biological age/delta.
- needs_disease_risks: true if the question is about the patient's
  disease risks, what conditions they're at risk for, or asks for a
  diagnosis/history check not answered by the questionnaire alone. false
  for pure biomarker or PC questions with no disease-risk angle.
When unsure, default both to true — the safe side is including
information, not silently omitting something the answer needed.

Reply with ONLY a JSON object in this exact format, nothing else:
{{
  "biomarkers": ["CODE1", "CODE2"],
  "pcs": ["PC1", "PC5"],
  "terms_identified": ["pulse rate", "blood pressure"],
  "needs_full_biomarkers": true,
  "needs_disease_risks": true
}}

Maximum 5 biomarker codes, maximum 5 PC codes — never list all PCs.
"""

    try:
        import json
        import re
        response = llm_generate(prompt)
        # extract the first JSON object from the response
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")
        cleaned = match.group(0)
        parsed = json.loads(cleaned)
        biomarkers = parsed.get("biomarkers", [])
        pcs = parsed.get("pcs", [])
        terms = parsed.get("terms_identified", [])
        needs_full_biomarkers = parsed.get("needs_full_biomarkers", True)
        needs_disease_risks = parsed.get("needs_disease_risks", True)
        logger.info(
            f"LLM variable extraction | terms={terms} | "
            f"biomarkers={biomarkers} | pcs={pcs} | "
            f"needs_full_biomarkers={needs_full_biomarkers} | needs_disease_risks={needs_disease_risks}"
        )
        return biomarkers, pcs, needs_full_biomarkers, needs_disease_risks
    except Exception as e:
        logger.warning(f"LLM variable extraction failed: {e} — falling back to codebook")
        return [], [], True, True



def _format_longitudinal_context(
    question: str,
    data: dict,
    biomarkers_requested: list[str],
    pcs_requested: list[str],
    patient_payload: dict | None,
) -> str:
    parts = []

    if patient_payload:
        chron = patient_payload.get("latest_chron_age", "N/A")
        bio = patient_payload.get("latest_bio_age", "N/A")
        delta = patient_payload.get("latest_delta", "N/A")
        gender = patient_payload.get("gender", "N/A")
        seqn = patient_payload.get("seqn", "N/A")

        try:
            delta_float = float(delta)
            aging_status = (
                "aging faster than normal" if delta_float > 0
                else "aging slower than normal"
            )
        except Exception:
            aging_status = "unknown aging status"

        risks = patient_payload.get("risks", [])
        top_risks = " | ".join([
            f"{r['disease_name']} (score: {r['evidence_score']})"
            for r in risks[:5]
        ])

        sig_pcs = sorted(
            [
                (k, v)
                for k, v in patient_payload.get(
                    "total_pc_contributions", {}
                ).items()
                if v != 0
            ],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        pc_str = " | ".join([f"{pc}: {val:+.3f}" for pc, val in sig_pcs])

        first = (patient_payload.get("first_name") or "").strip()
        last = (patient_payload.get("last_name") or "").strip()
        display_name = f"{first} {last}".strip() if (first or last) else f"SEQN {seqn}"

        clinician = patient_payload.get("clinician_name") or "N/A"
        parts.append(
            f"=== Patient Profile ===\n"
            f"Name: {display_name} | SEQN: {seqn} | Gender: {gender} | "
            f"Chronological Age: {chron} | Biological Age: {bio} | "
            f"Delta: {delta} | Status: {aging_status} | Clinician: {clinician}\n"
            f"Top PC Contributions: {pc_str}\n"
            f"Top Disease Risks: {top_risks}"
        )

        events = patient_payload.get("events", [])
        if events:
            parts.append("\n=== Patient Life Events ===")
            for e in sorted(events, key=lambda x: x.get("date", "")):
                parts.append(f"- {_format_date(e.get('date', 'unknown'))}: {e.get('label', '')}")

    if patient_payload:
        biomarkers_snap = patient_payload.get("biomarkers", {})
        q_lines = []
        for k, v in biomarkers_snap.items():
            if not is_questionnaire(k):
                continue
            raw = v.get("value") if isinstance(v, dict) else v
            if raw is None:
                q_lines.append(f"{get_label(k)}: Not answered")
                continue
            q_lines.append(f"{get_label(k)}: {decode_questionnaire_value(k, raw)}")
        if q_lines:
            parts.append("\n=== Patient Questionnaire (Medical History) ===\n" + "\n".join(q_lines))

    biomarker_data = data.get("biomarkers", {})
    if biomarker_data:
        parts.append("=== Biomarker Time Series ===")
        for code, readings in biomarker_data.items():
            label = get_label(code)
            parts.append(f"\n{label} ({code}):")
            if not readings:
                parts.append("  No readings available")
            else:
                rows = ["| Date | Value | Change |", "|---|---|---|"]
                prev_val = None
                for r in readings:
                    date = _format_date(r.get("date", "unknown date"))
                    value = r.get("value", "N/A")
                    if prev_val is not None and isinstance(value, (int, float)) and isinstance(prev_val, (int, float)):
                        change = f"{value - prev_val:+.3f}"
                    else:
                        change = "—"
                    rows.append(f"| {date} | {value} | {change} |")
                    if isinstance(value, (int, float)):
                        prev_val = value
                parts.append("\n".join(rows))

    age_keywords = ["bio age", "biological age", "delta", "aging", "clock", "older", "younger"]
    include_clock = any(kw in question.lower() for kw in age_keywords)
    clock_results = data.get("clock_results", []) if include_clock else []
    if clock_results:
        parts.append("\n=== Biological Age Over Time ===")
        rows = ["| Date | Bio Age | Chron Age | Delta |", "|---|---|---|---|"]
        for cr in clock_results:
            date = _format_date(cr.get("date", "unknown"))
            bio_age = cr.get("bio_age", "N/A")
            chron_age = cr.get("chron_age", "N/A")
            delta = cr.get("delta", "N/A")
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)
            rows.append(f"| {date} | {bio_age} | {chron_age} | {delta_str} |")
        parts.append("\n".join(rows))

    pc_data = data.get("pcs", {})
    if pc_data:
        parts.append("\n=== PC Values Over Time ===")
        for pc_name, readings in pc_data.items():
            parts.append(f"\n{pc_name}:")
            for r in readings:
                date = _format_date(r.get("date", "unknown"))
                value = r.get("value", "N/A")
                parts.append(f"  {date}: {value:+.4f}" if isinstance(value, float) else f"  {date}: {value}")

    parts.append("\n=== Variables Requested ===")
    for code in biomarkers_requested:
        parts.append(f"  {code}: {get_label(code)}")
    for pc in pcs_requested:
        parts.append(f"  {pc}")

    parts.append(f"\n=== Clinician Question ===\n{question}")

    return "\n".join(parts)


def answer_longitudinal_question(
    question: str,
    patient_id: str,
    token: str,
    patient_payload: dict | None = None,
    llm_generate=None,
) -> tuple[str, list[str]]:
    """Returns (answer_text, sources)."""
    logger.info(
        f"Longitudinal query | patient={patient_id} | question={question[:60]}"
    )

    biomarkers = []
    pcs = []

    if llm_generate:
        available = get_force_included_variables()
        biomarkers, pcs, _, _ = _extract_variables_with_llm(
            question, available, llm_generate
        )

    if not biomarkers and not pcs:
        logger.warning(
            "No variables detected — fetching default vital signs"
        )
        biomarkers = ["BPXPLS", "BPXSAR", "BPXDAR", "BMXBMI"]

    logger.info(
        f"Final variable selection | biomarkers={biomarkers} | pcs={pcs}"
    )

    data = fetch_longitudinal(
        patient_id=patient_id,
        token=token,
        biomarkers=biomarkers,
        pcs=pcs,
    )

    if not data:
        return (
            "I was unable to retrieve longitudinal data for this patient. "
            "The ALIS API did not return any time series data.",
            []
        )

    context = _format_longitudinal_context(
        question=question,
        data=data,
        biomarkers_requested=biomarkers,
        pcs_requested=pcs,
        patient_payload=patient_payload,
    )

    prompt = f"""
Clinical longitudinal data for patient analysis:

{context}
"""

    answer = call_llm(prompt, system_prompt=LONGITUDINAL_SYSTEM_PROMPT, raw_markdown=True)

    sources = [f"longitudinal:{patient_id}"]
    for code in biomarkers:
        sources.append(f"biomarker:{code}")
    for pc in pcs:
        sources.append(f"pc_longitudinal:{pc}")

    return answer, sources