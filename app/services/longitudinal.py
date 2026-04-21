import logging
from datetime import datetime
from typing import Optional

from app.services.codebook import (
    get_label,
    get_force_included_variables,
)
from app.services.alis_api import fetch_longitudinal
from app.services.llm_service import call_llm

logger = logging.getLogger(__name__)


def _format_date(date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return date_str


LONGITUDINAL_SYSTEM_PROMPT = """
You are a clinical assistant for the LinAge2 biological aging platform.

Rules:
- The data provided has already been fetched and verified. Always answer using it — never say data is unavailable.
- When the data contains multiple time points, present each biomarker as a section using this format:

## Biomarker Name (CODE)
- DATE: VALUE (CHANGE)
- DATE: VALUE (CHANGE)
Trend: one sentence on direction and magnitude.

- For biological age questions use:

## Biological Age Over Time
- DATE: Bio Age = X | Chron Age = Y | Delta = Z
Trend: one sentence on overall direction.

- If only one data point exists, state the single value and note there is insufficient history to determine a trend.
- Do not explain what measurements are. Do not give generic information.
- Do not use pipe tables (|). Use ## headers and bullet points only.
"""


def _extract_variables_with_llm(
    question: str,
    available_variables: dict[str, str],
    llm_generate,
) -> tuple[list[str], list[str]]:
    """
    Ask the LLM to identify which biomarker codes and PC names
    are needed to answer the question.
    Returns (biomarker_codes, pc_names)
    """
    var_list = "\n".join([
        f"{code}: {label}"
        for code, label in available_variables.items()
    ])

    prompt = f"""You are helping select variables to answer a clinical question about a patient's health over time.

Available biomarker variables:
{var_list}

Available PC values: PC1 through PC59

Clinical question: {question}

Which variables are needed to answer this question?
Reply with ONLY a JSON object in this exact format, nothing else:
{{
  "biomarkers": ["CODE1", "CODE2"],
  "pcs": ["PC1", "PC5"],
  "terms_identified": ["pulse rate", "blood pressure"]
}}

Rules:
- Only include variables directly relevant to the question
- For questions about blood work, biomarkers, or lab results: include only biomarker codes, leave pcs as empty list
- For questions about aging, delta, biological age, or clock results: leave both biomarkers and pcs as empty lists — those come from clock_results automatically
- For questions explicitly about a specific PC (e.g. "how has PC1 changed"): include it in pcs, leave biomarkers empty
- If unsure, include the most likely biomarker variable rather than leaving it empty
- Maximum 5 biomarker codes
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
        logger.info(
            f"LLM variable extraction | terms={terms} | "
            f"biomarkers={biomarkers} | pcs={pcs}"
        )
        return biomarkers, pcs
    except Exception as e:
        logger.warning(f"LLM variable extraction failed: {e} — falling back to codebook")
        return [], []



def _format_longitudinal_context(
    question: str,
    data: dict,
    biomarkers_requested: list[str],
    pcs_requested: list[str],
    patient_payload: dict | None,
) -> str:
    """
    Build the full context string for the LLM to reason about.
    """
    parts = []

    # patient profile from Qdrant
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

        parts.append(
            f"=== Patient Profile ===\n"
            f"SEQN: {seqn} | Gender: {gender} | "
            f"Chronological Age: {chron} | Biological Age: {bio} | "
            f"Delta: {delta} | Status: {aging_status}\n"
            f"Top PC Contributions: {pc_str}\n"
            f"Top Disease Risks: {top_risks}"
        )

    # biomarker time series — pre-built as markdown tables
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

    # clock results — only include if the question is about biological age or aging
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

    # PC values over time
    pc_data = data.get("pcs", {})
    if pc_data:
        parts.append("\n=== PC Values Over Time ===")
        for pc_name, readings in pc_data.items():
            parts.append(f"\n{pc_name}:")
            for r in readings:
                date = _format_date(r.get("date", "unknown"))
                value = r.get("value", "N/A")
                parts.append(f"  {date}: {value:+.4f}" if isinstance(value, float) else f"  {date}: {value}")

    # what was requested — for the LLM to explain its variable choices
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
    """
    Main entry point for longitudinal questions.
    Returns (answer_text, sources)
    """
    logger.info(
        f"Longitudinal query | patient={patient_id} | question={question[:60]}"
    )

    # step 1: detect which variables are needed
    biomarkers = []
    pcs = []

    if llm_generate:
        available = get_force_included_variables()
        biomarkers, pcs = _extract_variables_with_llm(
            question, available, llm_generate
        )

    # if nothing detected, fetch a minimal default set
    if not biomarkers and not pcs:
        logger.warning(
            "No variables detected — fetching default vital signs"
        )
        biomarkers = ["BPXPLS", "BPXSAR", "BPXDAR", "BMXBMI"]

    logger.info(
        f"Final variable selection | biomarkers={biomarkers} | pcs={pcs}"
    )

    # step 2: fetch longitudinal data from ALIS API
    data = fetch_longitudinal(
        patient_id=patient_id,
        token=token,
        biomarkers=biomarkers,
        pcs=pcs,
    )

    if not data:
        return (
            f"I was unable to retrieve longitudinal data for this patient. "
            f"The ALIS API did not return any time series data.",
            []
        )

    # step 3: build context
    context = _format_longitudinal_context(
        question=question,
        data=data,
        biomarkers_requested=biomarkers,
        pcs_requested=pcs,
        patient_payload=patient_payload,
    )

    # step 4: call LLM with structured context
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