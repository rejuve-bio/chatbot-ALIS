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

    prompt = f"""Two different data sources could answer parts of this clinical
question. Your only job is to pick correctly between them — you are not
answering the question yourself.

SOURCE 1 — the main patient context (already loaded, already in front of
you elsewhere in this conversation, costs nothing extra to use):
- Patient Profile: chronological age, biological age, delta.
- PC Contributions Table: EVERY one of this patient's PCs that has a
  nonzero contribution, each already listed with its exact value AND
  direction (positive = aging faster, negative = protective), already
  sorted by size. Any question that ranks, filters, compares, or asks
  "which/how many/top/most/highest/lowest" PCs by direction or magnitude
  is fully answered from this table alone — nothing needs to be fetched
  for it, no matter what numbers or counts appear in the question's
  wording (a "3" in "the 3 pcs raising aging" is not a PC number; it's
  asking for 3 rows out of this table, filtered by direction).
- Biomarkers section: every lab/vital/questionnaire value this patient
  has, current snapshot only (~70 lines).
- Disease Risks section: every disease risk with its evidence score.
- Life Events.

Available biomarker variables (their codes, for reference only —
these are already covered by SOURCE 1 above for their current value):
{var_list}

SOURCE 2 — the longitudinal fetch (what you are deciding whether to
trigger): adds exactly one thing SOURCE 1 doesn't have — a per-visit
HISTORY table, across past dates, for ONE variable that the question
already names explicitly by its own code or label (e.g. "PC32", "blood
pressure", "glucose"). It adds no new PCs, no new biomarkers, no
rankings, and no direction/comparison info beyond what SOURCE 1 already
has — it only shows how one already-named variable's value moved across
visits.
{context_line}
Clinical question: {question}

DECISION: does answering this question require SOURCE 2 — i.e. does it
name one specific variable (a biomarker or a PC, by its own code/label)
and ask how that one variable's value changed across dates/visits ("over
time", "trend", "history", "since", "at each visit", "from X to Y")? If
so, list only that variable's code.

If instead the question is broad, general, a ranking/filter across many
PCs or biomarkers, a diagnosis/history check, or an overview — return
EMPTY lists for both, even if it mentions specific numbers or counts.
SOURCE 1 already answers all of those. Aging/delta/biological-age
questions also return empty — those come from clock_results
automatically, not from either source above.

When genuinely unsure, return empty lists — SOURCE 1 already covers
general and single-value lookups; SOURCE 2 is only for the narrower case
of a clearly named variable's own history across visits.

Separately: SOURCE 1's "Biomarkers" section and "Disease Risks" section
get included by default, but reading through them costs real time for
questions that don't need them. Decide honestly whether THIS question
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

        mentioned = set(re.findall(r"pc\s*0*(\d{1,2})\b", question, re.IGNORECASE))
        pcs = [pc for pc in pcs if re.sub(r"\D", "", pc).lstrip("0") in mentioned]

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

    parts.extend(_time_series_parts(question, data, biomarkers_requested, pcs_requested))
    parts.append(f"\n=== Clinician Question ===\n{question}")

    return "\n".join(parts)


def _time_series_parts(
    question: str,
    data: dict,
    biomarkers_requested: list[str],
    pcs_requested: list[str],
) -> list[str]:
    """The per-visit history tables a longitudinal fetch adds — everything
    _format_longitudinal_context has beyond the patient profile/questionnaire,
    which a caller that already has those from the main patient context
    (see _format_longitudinal_addendum) doesn't need duplicated."""
    parts = []

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

    return parts


def _format_longitudinal_addendum(
    question: str,
    data: dict,
    biomarkers_requested: list[str],
    pcs_requested: list[str],
) -> str:
    """History tables to append to the main patient context, when a question
    needs both — e.g. a PC ranking that also asks how one of those PCs
    trended over time. The main context already has the patient profile,
    full PC Contributions Table, biomarkers, and disease risks; this adds
    only what a named-variable longitudinal fetch has beyond that."""
    return "\n".join(_time_series_parts(question, data, biomarkers_requested, pcs_requested))


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