
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from dotenv import load_dotenv

from app.services.llm_service import embed_text, embed_batch, call_llm
from app.services.qdrant_service import (
    search_pc_knowledge,
    list_patients,
    upsert_pc_chunks, search_chunks,
)
from app.services.backend_api import fetch_patient, fetch_longitudinal, fetch_all_patients, fetch_biomarker_data_latest
from app.services.ingestion.excel_parser import parse_excel
from app.services.ingestion.pdf_parser import parse_pdf
from app.services.longitudinal import (
    _extract_variables_with_llm, _format_longitudinal_context,
    _format_date, LONGITUDINAL_SYSTEM_PROMPT,
)
from app.services.codebook import (
    get_force_included_variables, get_label, decode_questionnaire_value,
    is_questionnaire, DEMOGRAPHIC_CODES,
)
from app.services.memory import get_history, save_turn
from app.services.glossary import match_glossary_question, needs_population_data, needs_biology_evidence

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an AI clinical assistant. Answer clinician questions using only the patient data provided in the context.

GROUNDING — CRITICAL:
- Answer only from the context. Never use your own training knowledge to fill gaps.
- Never invent values, trends, diagnoses, or interpretations not in the context.
- Do not mention any platform, product, or system name.

PATIENT SELECTION RULES — read this before answering anything:
- The context will either contain a single patient's data OR an "All Patients Summary".
- If the context is "All Patients Summary": answer population questions (who is aging fastest, compare patients, list all patients, most common/significant PCs across all subjects, which PC appears most often, rank PCs by contribution across patients). For anything else that requires a single patient's full record, respond: "Please select a specific patient or provide a patient name, ID, or SEQN to view their [biomarkers / heart rate / PC scores / etc.]."
- If a patient IS selected but their data does not contain what was asked: respond "I wasn't able to find that information in this patient's data."
- Never say "I wasn't able to find that information" when no patient is selected — that response is only for when a patient IS selected.
- If NO patient is selected and the context is not an "All Patients Summary" either — e.g. someone just asked what a specific PC group means — this is a plain informational lookup, not a patient consultation. Answer naturally in your own words, at least 3-4 sentences, explaining what it is and why it matters. Do not use a table, do not use bullet points, and do not add a "most actionable implication" sentence — there is no patient here for anything to be actionable for.
- USING CONVERSATION HISTORY TO RESOLVE FOLLOW-UPS: if an earlier message in this conversation asked a specific question (e.g. "what are the biomarkers") but got the "please select a specific patient" response because no patient was identified yet, and a patient IS identified now (by name, ID, or SEQN) with the current message otherwise adding no new question of its own — treat the current message as answering that pending question for this now-identified patient. Answer the earlier question directly; do not ask the clinician to repeat or clarify what they already asked.

SCOPE:
- Greetings (hi, hello, good morning): respond briefly and warmly, e.g. "Hello! How can I help with your patient today?"
- Questions outside clinical data (general medical knowledge, coding, news, personal): respond "I can only assist with patient data."

FORMATTING — use the simplest format that fits:
- Single value or direct fact: one or two plain sentences. No table.
- Multiple values side by side (PC rankings, biomarker comparison, patient list): markdown table.
- PC ranking (single patient): if the question asks for a specific number of PCs (e.g. "top 3", "highest 3", "three"), show only that many rows from the PC Contributions Table sorted by absolute contribution. Otherwise output the full table. End with one sentence on the most urgent PC.
- PC ranking (population / All Patients Summary): aggregate PCs across all patients by summing absolute contributions or counting frequency. If the question specifies a number (e.g. "3 most significant"), return ONLY that many rows — no more. Always end with one sentence on the top-ranked PC.
- PC comparison (e.g. PC1M vs PC1F): markdown table (| Dimension | PC_A | PC_B |).
- Life events: only use a table if the clinician explicitly asks to list events; otherwise mention them inline.
- Separate sections: use ## headers. Lists of observations: use bullet points.
- Broad biomarker requests with nothing specific named (e.g. "what are this patient's biomarkers", "show me their labs"): do NOT retype every line from the context verbatim — that's a wall of text, not an answer. Lead with anything abnormal or flagged ("Yes" questionnaire answers, out-of-range values), then a handful of key vitals (blood pressure, BMI, glucose). Never omit an abnormal or clinically notable value from this summary. End by noting more measured values exist and offering to list a specific one or the full set if the clinician wants it.

CLINICAL RULES:
- PC contribution values must always include direction: positive = aging faster, negative = protective.
- If a PC does not match the patient gender, redirect to the correct gender PC in one sentence then answer using it.
- Use biomarker names and labels from the context. Do not guess codes.
- Answer only what was asked. Do not volunteer unrelated data.
- Check "Patient Life Events" first for any question about interventions, medications, or lifestyle changes.
- Questionnaire answers appear in the Biomarkers section with Yes/No values. Use these to answer questions about medical history (hypertension, diabetes, smoking, fractures, etc.).
- When asked about a condition (e.g. "does this patient have hypertension"), look for the relevant questionnaire variable in the Biomarkers section and answer from it directly.
- If the Biomarkers section has no questionnaire answer for the condition asked, check the Disease Risks section — if the condition appears there with an evidence score, report it (e.g. "The questionnaire data does not confirm this, but Hypertension is listed as a top disease risk with an evidence score of 9.9").

INVESTIGATIONAL EVIDENCE RULES:
- The "Investigational, Preclinical Evidence" section (if present) is preclinical/experimental data from sources like DrugAge, GenAge, CellAge, and ClinPGx — it is NOT clinical guidance and never overrides it.
- Never present this evidence as a treatment recommendation, and never imply it is validated in humans unless its evidence_tier is explicitly "human".
- If evidence_tier is "animal_model", always name the species tested and state plainly that it has not been validated in humans.
- Keep this evidence clearly separate from the Clinical guidance / Disease Risks sections — never blend the two into one claim.
- Only bring this section into the answer if the clinician's question is actually about interventions, compounds, genes, or mechanisms — not for routine biomarker/PC questions.

SYNTHESIS:
- End every clinical answer with one sentence on the single most actionable implication, citing an actual number from the context.
- No filler phrases. Stop after the synthesis sentence.
"""


def _patient_display_name(payload: dict) -> str:
    first = (payload.get("first_name") or "").strip()
    last = (payload.get("last_name") or "").strip()
    if first or last:
        return f"{first} {last}".strip()
    return f"SEQN {payload.get('seqn', 'unknown')}"


def build_patient_text_summary(data: dict) -> str:
    seqn = data.get("seqn", "unknown")
    gender = data.get("gender", "unknown")
    chron_age = data.get("latest_chron_age", "N/A")
    bio_age = data.get("latest_bio_age", "N/A")
    delta = data.get("latest_delta", "N/A")
    try:
        aging_status = "aging faster than normal" if delta not in (None, "N/A") and float(delta) > 0 else "aging slower than normal"
    except (ValueError, TypeError):
        aging_status = "unknown"

    heatmap_data = data.get("latest_heatmap", {})
    heatmap_rows = heatmap_data.get("rows", [])
    label_to_human = {row["label"]: row["human"] for row in heatmap_rows}

    biomarkers = data.get("biomarkers", {})
    biomarker_str = " | ".join([
        f"{label_to_human.get(k, k)} ({k}): {v}"
        for k, v in biomarkers.items()
        if v is not None and k not in ["id", "patient_id", "source", "created_at", "updated_at"]
    ])

    risks = data.get("risks", [])
    risk_str = " | ".join([
        f"{r['disease_name']} (score: {r['evidence_score']})"
        for r in risks[:10]
    ])

    top_pcs = sorted(
        [(k, v) for k, v in heatmap_data.get("total_pc_contributions", {}).items() if v != 0],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    pc_str = " | ".join([f"{pc}: {val:+.2f}" for pc, val in top_pcs])

    return (
        f"Patient UUID: {data.get('id')} | SEQN: {seqn} | Gender: {gender} | "
        f"Chronological Age: {chron_age} | Biological Age: {bio_age} | "
        f"Delta: {delta} | Aging Status: {aging_status} | "
        f"Top PC Contributions: {pc_str} | "
        f"Biomarkers: {biomarker_str} | "
        f"Top Disease Risks: {risk_str}"
    )


_PATIENT_FETCH_CACHE: dict[str, tuple[float, dict]] = {}
_PATIENT_FETCH_TTL_SECONDS = 180  # 3 minutes

def fetch_and_store_patient(patient_uuid: str, token: str) -> dict | None:
    """Fetches only — no embed/upsert. 3-minute in-memory TTL cache."""
    now = time.time()
    cached = _PATIENT_FETCH_CACHE.get(patient_uuid)
    if cached and (now - cached[0]) < _PATIENT_FETCH_TTL_SECONDS:
        logger.info(f"Patient {patient_uuid} served from short-TTL cache "
                    f"({now - cached[0]:.0f}s old) — skipping ALIS re-fetch")
        return cached[1]

    logger.info(f"Fetching patient {patient_uuid} from ALIS API")
    data = fetch_patient(patient_uuid, token=token)
    if not data:
        logger.warning(f"No data returned from ALIS API for patient {patient_uuid}")
        return None

    heatmap = data.get("latest_heatmap", {})
    label_to_human = {row["label"]: row["human"] for row in heatmap.get("rows", [])}
    total_pc_contributions = heatmap.get("total_pc_contributions", {})

    biomarker_latest = fetch_biomarker_data_latest(patient_uuid, token=token)
    if biomarker_latest:
        _skip = ("id", "patient_id", "source", "created_at", "updated_at", "measurement_date")
        merged = {k: v for k, v in biomarker_latest.items()
                  if k not in _skip and v is not None}
        existing = data.get("biomarkers", {}) or {}
        data["biomarkers"] = {**merged, **existing}  # existing values win on conflict
        logger.info(f"Merged {len(merged)} biomarker-data/latest values for {patient_uuid}")

    significant_pcs = {k: v for k, v in total_pc_contributions.items() if v != 0}
    sorted_pcs = sorted(significant_pcs.items(), key=lambda x: abs(x[1]), reverse=True)

    payload = {
        "seqn": data.get("seqn"),
        "first_name": data.get("first_name"),
        "last_name": data.get("last_name"),
        "gender": data.get("gender"),
        "latest_chron_age": data.get("latest_chron_age"),
        "latest_bio_age": data.get("latest_bio_age"),
        "latest_delta": data.get("latest_delta"),
        "label_to_human": label_to_human,
        "biomarkers": data.get("biomarkers", {}),
        "risks": data.get("risks", []),
        "total_pc_contributions": total_pc_contributions,
        "significant_pcs_ranked": sorted_pcs,
        "events": data.get("events", []),
        "clinician_name": (data.get("clinician") or {}).get("name"),
    }

    _PATIENT_FETCH_CACHE[patient_uuid] = (time.time(), payload)
    return payload


def populate_all_patients(token: str) -> dict[str, dict]:
    """Unused — kept for a future scheduler that syncs Qdrant on its own cadence."""
    logger.info("Bulk-fetching all patients from /patients")
    all_items = fetch_all_patients(token=token)
    results: dict[str, dict] = {}
    for item in all_items:
        pid = item.get("id")
        if not pid:
            continue
        try:
            payload = fetch_and_store_patient(pid, token=token)
            if payload:
                results[pid] = payload
        except Exception as e:
            logger.warning(f"populate_all_patients: failed for {pid}: {e}")
    logger.info(f"Bulk fetch complete — {len(results)} patients fetched")
    return results


def build_context(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None,
    token: Optional[str] = None,
    query_vector: Optional[list] = None,
    check_biology_evidence: bool = True,
    include_biomarkers: bool = True,
    include_disease_risks: bool = True,
) -> tuple[str, list[str], Optional[dict], Optional[list[str]]]:
    logger.info(f"Building context | patient_id: {patient_id} | pc_group: {pc_group}")
    if query_vector is None:
        query_vector = embed_text(question)
    context_parts = []
    sources = []
    patient_payload = None

    if patient_id:
        if token:
            try:
                patient_payload = fetch_and_store_patient(patient_id, token=token)
                logger.info(f"Patient {patient_id} fetched fresh from ALIS API")
            except Exception as e:
                logger.warning(f"ALIS API unavailable for {patient_id}: {e} — no patient data for this message")

        if patient_payload:
            biomarkers = patient_payload.get("biomarkers", {})
            risks = patient_payload.get("risks", [])

            logger.info(f"Patient {patient_id} biomarkers: {biomarkers}")
            logger.info(f"Patient {patient_id} risks: {risks}")
            biomarker_lines = []
            for k, v in biomarkers.items():
                if k in ("id", "patient_id", "source", "created_at", "updated_at"):
                    continue
                if k in DEMOGRAPHIC_CODES:
                    continue
                raw = v.get("value") if isinstance(v, dict) else v
                label = get_label(k)
                if is_questionnaire(k):
                    if raw is None:
                        biomarker_lines.append(f"{label}: Not answered")
                    else:
                        biomarker_lines.append(f"{label}: {decode_questionnaire_value(k, raw)}")
                else:
                    if raw is None:
                        continue
                    biomarker_lines.append(f"{label} ({k}): {raw}")
            biomarker_str = "\n".join(biomarker_lines)

            risk_str = "\n".join([
                f"{r['disease_name']} — evidence score: {r['evidence_score']} — contributing PCs: {', '.join(r['contributing_pcs'])}"
                for r in risks
            ])

            total_pc_contributions = patient_payload.get("total_pc_contributions", {})
            significant_pcs = sorted(
                [(k, v) for k, v in total_pc_contributions.items() if v != 0],
                key=lambda x: abs(x[1]),
                reverse=True
            )

            pc_table_rows = ["| PC | Contribution | Direction |", "|---|---|---|"]
            for pc, val in significant_pcs[:8]:
                direction = "Aging faster" if val > 0 else "Protective"
                pc_table_rows.append(f"| {pc} | {val:+.3f} yrs | {direction} |")
            pc_table = "\n".join(pc_table_rows)

            context_parts.append(f"\n=== PC Contributions Table ===\n{pc_table}")

            context_parts.append("=== Patient Profile ===")
            context_parts.append(
                f"Name: {_patient_display_name(patient_payload)} | "
                f"Patient ID: {patient_id} | "
                f"SEQN: {patient_payload.get('seqn')} | "
                f"Gender: {patient_payload.get('gender')} | "
                f"Chronological Age: {patient_payload.get('latest_chron_age')} | "
                f"Biological Age: {patient_payload.get('latest_bio_age')} | "
                f"Delta: {patient_payload.get('latest_delta')} | "
                f"Clinician: {patient_payload.get('clinician_name') or 'N/A'}"
            )

            events = patient_payload.get("events", [])
            if events:
                event_lines = [
                    f"- {_format_date(e.get('date', 'unknown'))}: {e.get('label', '')}"
                    for e in sorted(events, key=lambda x: x.get("date", ""))
                ]
                context_parts.append("\n=== Patient Life Events ===\n" + "\n".join(event_lines))

            if include_biomarkers:
                context_parts.append(f"\n=== Biomarkers ===\n{biomarker_str}")
            else:
                logger.info(f"Skipping full Biomarkers section for {patient_id} — not relevant to this question")
            if include_disease_risks:
                context_parts.append(f"\n=== Disease Risks ===\n{risk_str}")
            else:
                logger.info(f"Skipping full Disease Risks section for {patient_id} — not relevant to this question")
            logger.info(f"Patient {patient_id} context parts: {context_parts}")
            sources.append(f"patient:{patient_id}")
            logger.info(f"Patient context built successfully for {patient_id}")
        else:
            logger.warning(f"No data found for patient {patient_id}")
            context_parts.append(f"No data found for patient {patient_id}.")

    elif needs_population_data(question):
        api_available = False
        clinic_ids: set[str] = set()
        qdrant_map: dict[str, dict] = {}

        if token:
            try:
                clinic_items = fetch_all_patients(token=token)
                clinic_ids = {p.get("id") for p in clinic_items if p.get("id")}
                api_available = True
                logger.info(f"Fetched {len(clinic_ids)} clinic patients from API")
            except Exception as e:
                logger.warning(f"ALIS API unavailable for population query: {e} — falling back to Qdrant")

        qdrant_patients = list_patients()
        qdrant_map = {p["id"]: p.get("payload", {}) for p in qdrant_patients}

        if api_available and clinic_ids:
            for pid in clinic_ids:
                try:
                    payload = fetch_and_store_patient(pid, token=token)
                    if payload:
                        qdrant_map[pid] = payload
                except Exception as e:
                    logger.warning(f"Population: could not refresh patient {pid}: {e}")
            display_ids = clinic_ids
        else:
            display_ids = qdrant_map.keys()

        logger.info(f"No patient_id — building population context from {len(display_ids)} patients")

        if display_ids:
            context_parts.append("=== All Patients Summary ===")
            for pid in display_ids:
                payload = qdrant_map.get(pid, {})
                delta = payload.get("latest_delta")
                if delta is None:
                    continue
                try:
                    aging_status = "aging faster" if float(delta) > 0 else "aging slower"
                except Exception:
                    aging_status = "unknown"
                top_pcs = sorted(
                    [(k, v) for k, v in payload.get("total_pc_contributions", {}).items() if v != 0],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:3]
                pc_str = ", ".join(f"{pc}:{val:+.2f}" for pc, val in top_pcs) if top_pcs else "N/A"
                context_parts.append(
                    f"Name: {_patient_display_name(payload)} | "
                    f"Patient ID: {pid} | "
                    f"SEQN: {payload.get('seqn')} | "
                    f"Gender: {payload.get('gender')} | "
                    f"Chron Age: {payload.get('latest_chron_age')} | "
                    f"Bio Age: {payload.get('latest_bio_age')} | "
                    f"Delta: {delta} | "
                    f"Status: {aging_status} | "
                    f"Top PCs: {pc_str}"
                )
            sources.append("all_patients")
        else:
            context_parts.append("No patients found in the database.")

    pc_hits = search_pc_knowledge(query_vector, pc_group=pc_group, limit=3)
    logger.info(f"PC knowledge hits: {len(pc_hits)}")
    if pc_hits:
        patient_gender = (patient_payload or {}).get("gender", "").lower()
        total_pc_contributions = (patient_payload or {}).get("total_pc_contributions", {})
        context_parts.append("\n=== PC Clinical Interpretation ===")
        for hit in pc_hits:
            hit_pc_group = hit.get("pc_group", "")
            hit_suffix = hit_pc_group[-1].lower() if hit_pc_group and hit_pc_group[-1].lower() in ("m", "f") else None
            gender_mismatch = (
                hit_suffix == "m" and patient_gender == "female"
                or hit_suffix == "f" and patient_gender == "male"
            )
            if gender_mismatch:
                correct_suffix = "F" if hit_suffix == "m" else "M"
                correct_pc_group = hit_pc_group[:-1] + correct_suffix
                pc_number = hit_pc_group[:-1]
                patient_pc_value = total_pc_contributions.get(pc_number)

                context_parts.append(
                    f"NOTE: The clinician asked about {hit_pc_group} but this patient is {patient_gender}. "
                    f"The gender-appropriate knowledge is {correct_pc_group}."
                )
                if patient_pc_value is not None:
                    context_parts.append(
                        f"This patient's {pc_number} contribution is {patient_pc_value:+.3f}."
                    )

                correct_hits = search_pc_knowledge(query_vector, pc_group=correct_pc_group, limit=2)
                if correct_hits:
                    for correct_hit in correct_hits:
                        context_parts.append(
                            f"PC Group: {correct_hit.get('pc_group')} | "
                            f"Risk Window: {correct_hit.get('risk_window')} | "
                            f"Causes of Death: {', '.join(correct_hit.get('causes_of_death', []))} | "
                            f"Associated Diseases: {', '.join(correct_hit.get('diseases', []))} | "
                            f"Mechanisms: {', '.join(correct_hit.get('mechanisms', []))} | "
                            f"Interventions: {', '.join(correct_hit.get('interventions', []))}"
                        )
                    sources.append(f"pc_knowledge:{correct_pc_group}")
                else:
                    context_parts.append(f"No knowledge found for {correct_pc_group} in the database.")
                sources.append(f"pc_knowledge:{hit_pc_group}:redirected_to:{correct_pc_group}")
            else:
                context_parts.append(
                    f"PC Group: {hit_pc_group} | "
                    f"Risk Window: {hit.get('risk_window')} | "
                    f"Causes of Death: {', '.join(hit.get('causes_of_death', []))} | "
                    f"Associated Diseases: {', '.join(hit.get('diseases', []))} | "
                    f"Mechanisms: {', '.join(hit.get('mechanisms', []))} | "
                    f"Interventions: {', '.join(hit.get('interventions', []))}"
                )
                sources.append(f"pc_knowledge:{hit_pc_group}")

    resource = None
    biology_hits = []
    if check_biology_evidence:
        from app.services.biology_service import ATLAS_RESOURCES_FOR_CHAT

        with ThreadPoolExecutor(max_workers=len(ATLAS_RESOURCES_FOR_CHAT)) as executor:
            futures = [
                executor.submit(search_chunks, collection, query_vector, limit=2, min_score=0.6)
                for collection in ATLAS_RESOURCES_FOR_CHAT
            ]
            biology_hits = [hit for f in futures for hit in f.result()]
        logger.info(f"Investigational biology search | collections={ATLAS_RESOURCES_FOR_CHAT} | "
                    f"total_hits={len(biology_hits)}")
    else:
        logger.info("Skipping investigational biology search — question not relevant to compounds/genes/evidence")

    if biology_hits:
        context_parts.append("\n=== Investigational, Preclinical Evidence (not validated in humans unless noted) ===")
        for hit in biology_hits:
            name = hit.get("name") or hit.get("compound_name") or "unknown"
            context_parts.append(
                f"{name} | source: {hit.get('source')} | evidence_tier: {hit.get('evidence_tier')} | "
                f"{hit.get('raw_text', '')}"
            )
        sources.append("biology_evidence")
        resource = sorted({hit.get("source") for hit in biology_hits if hit.get("source")})
        logger.info(f"Resource key set from biology collections: {resource}")

    context_str = "\n".join(context_parts) if context_parts else "No relevant context found."
    logger.info(f"Context built after searching pc knowledge {context_str}")
    return context_str, sources, patient_payload, resource


def build_prompt(question: str, context: str) -> str:
    return f"""
Clinical context from LinAge2 database:

{context}

---

Clinician question: {question}
"""


def ingest_excel(file_bytes: bytes, patient_id: str, pc_group: Optional[str] = None):
    logger.info(f"Ingesting Excel for patient: {patient_id}")
    chunks = parse_excel(file_bytes, patient_id, pc_group)
    if not chunks:
        logger.warning("No chunks extracted from Excel")
        return 0
    texts = [chunk["text_summary"] for chunk in chunks]
    vectors = embed_batch(texts)
    from app.services.qdrant_service import upsert_patient_chunks
    upsert_patient_chunks(chunks, vectors)
    logger.info(f"Ingested {len(chunks)} chunks for patient {patient_id}")
    return len(chunks)


def ingest_pdf(file_bytes: bytes):
    logger.info("Ingesting PDF for PC knowledge")
    chunks = parse_pdf(file_bytes)
    if not chunks:
        logger.warning("No chunks extracted from PDF")
        return 0
    texts = [chunk["raw_text"] for chunk in chunks]
    vectors = embed_batch(texts)
    upsert_pc_chunks(chunks, vectors)
    logger.info(f"Ingested {len(chunks)} PC knowledge chunks")
    return len(chunks)


def _resolve_seqn_from_message(question: str, token: Optional[str]) -> Optional[str]:
    """Matches a 5-digit SEQN-like number (e.g. 90002) and resolves it to a patient UUID."""
    match = re.search(r'\b(9\d{4})\b', question)
    if not match:
        return None
    seqn_str = match.group(1)
    try:
        patients = fetch_all_patients(token)
        for p in patients:
            if str(p.get("seqn", "")) == seqn_str:
                logger.info(f"Resolved SEQN {seqn_str} → patient_id {p.get('id')}")
                return p.get("id")
    except Exception as e:
        logger.warning(f"SEQN resolution failed: {e}")
    logger.warning(f"SEQN {seqn_str} mentioned in message but no matching patient found")
    return None


def rag_query(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[str, list[str], Optional[list[str]]]:
    history = get_history(token) if token else []
    prior_question = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)

    # glossary classify (LLM_HOST) and variable-extraction (LLM_HOST_2) run
    # concurrently — separate GPUs, safe; two concurrent calls to the SAME
    # host is not (commit 1a3157c).
    extraction_result = None
    with ThreadPoolExecutor(max_workers=2) as speculative_executor:
        glossary_future = speculative_executor.submit(
            match_glossary_question, question, has_patient=patient_id is not None
        )
        extraction_future = None
        if patient_id:
            available = get_force_included_variables()
            extraction_future = speculative_executor.submit(
                _extract_variables_with_llm,
                question, available, lambda p: call_llm(p, use_secondary=True), prior_question,
            )

        general_answer = glossary_future.result()
        if general_answer is not None:
            logger.info(f"rag_query | answered generally, skipping full pipeline: {question!r}")
            if token:
                save_turn(token, question, general_answer)
            return general_answer, ["glossary"], None

        if extraction_future:
            extraction_result = extraction_future.result()

    if not patient_id:
        resolved = _resolve_seqn_from_message(question, token)
        if resolved:
            patient_id = resolved
            if re.match(
                r'^(please\s+)?(look at|show|pull up|open|view|display|check|get)\s+\d+',
                question.strip(), re.IGNORECASE
            ):
                question = (
                    "Give me a brief clinical summary of this patient: their age, gender, "
                    "biological age, delta, top PC contributions, and the top 3 disease risks."
                )

    query_vector = embed_text(question)

    biomarkers, pcs = [], []
    needs_full_biomarkers, needs_disease_risks = True, True
    if extraction_result is not None:
        biomarkers, pcs, needs_full_biomarkers, needs_disease_risks = extraction_result
    elif patient_id:
        available = get_force_included_variables()
        biomarkers, pcs, needs_full_biomarkers, needs_disease_risks = _extract_variables_with_llm(
            question,
            available,
            lambda p: call_llm(p),
            prior_question=prior_question,
        )

    if patient_id:
        biomarkers = biomarkers[:5]
        pcs = pcs[:5]

    needs_longitudinal = bool(biomarkers or pcs)
    logger.info(f"Longitudinal decision | needs={needs_longitudinal} | biomarkers={biomarkers} | pcs={pcs}")

    # build_context() must stay free of its own call_llm() calls — it runs
    # concurrently with fetch_longitudinal() below (commit 1a3157c).
    check_biology = needs_biology_evidence(question)

    with ThreadPoolExecutor(max_workers=2) as executor:
        context_future = executor.submit(
            build_context, question, patient_id, pc_group, token, query_vector,
            check_biology, needs_full_biomarkers, needs_disease_risks
        )
        longitudinal_future = executor.submit(
            fetch_longitudinal,
            patient_id=patient_id,
            token=token,
            biomarkers=biomarkers,
            pcs=pcs,
        ) if (patient_id and needs_longitudinal) else None

    context, sources, patient_payload, resource = context_future.result()

    if longitudinal_future:
        longitudinal_data = longitudinal_future.result()
        if longitudinal_data:
            long_context = _format_longitudinal_context(
                question=question,
                data=longitudinal_data,
                biomarkers_requested=biomarkers,
                pcs_requested=pcs,
                patient_payload=patient_payload,
            )
            sources.append(f"longitudinal:{patient_id}")
            prompt = build_prompt(question, long_context)
            answer = call_llm(prompt, system_prompt=LONGITUDINAL_SYSTEM_PROMPT, raw_markdown=True, history=history)
            if token:
                save_turn(token, question, answer)
            return answer, sources, None

    prompt = build_prompt(question, context)
    answer = call_llm(prompt, system_prompt=SYSTEM_PROMPT, history=history)
    if token:
        save_turn(token, question, answer)
    return answer, sources, resource
