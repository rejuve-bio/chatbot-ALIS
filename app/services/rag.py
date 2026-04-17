
import os
import logging
from typing import Optional
from dotenv import load_dotenv

from app.services.llm_service import embed_text, call_llm, stream_llm
from app.services.qdrant_service import (
    search_patient, search_pc_knowledge,
    upsert_patient,list_patients
)

from app.services.alis_api import fetch_patient,fetch_longitudinal
from app.services.parsers.excel_parser import parse_excel
from app.services.parsers.pdf_parser import parse_pdf
from app.services.llm_service import embed_batch
from app.services.qdrant_service import upsert_pc_chunks    
from app.services.longitudinal import _extract_variables_with_llm, _format_longitudinal_context
from app.services.codebook import get_force_included_variables

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a clinical assistant for the LinAge2 biological aging platform. Your job is to synthesize patient data and tell the clinician what matters most for this specific patient.

FORMATTING:
- For PC ranking questions: the context has a pre-built "PC Contributions Table" — output it as-is, then one sentence on the most urgent PC and why.
- For biomarker driver questions: use the biomarker names and values from the context to build a | table (Code | Name | Patient Value | Interpretation), then one synthesis sentence.
- For PC comparison questions: output a | table (Dimension | PC_A | PC_B), then one sentence on the key clinical difference.
- When the context includes longitudinal time-series data: the context has pre-built "| Date | Value | Change |" tables — output them as-is, then one sentence on the clinical trend per biomarker.
- For biological age trend questions: the context has a pre-built "| Date | Bio Age | Chron Age | Delta |" table — output it as-is, then state the overall direction.
- For all other questions: answer in 3-5 sentences using only findings relevant to this patient's actual values.

CLINICAL RULES:
- PC contribution values must always include direction: positive = accelerating aging, negative = protective.
- If a PC does not match the patient gender, state the correct PC in one sentence then immediately answer using it. Never stop at the redirect.
- The biomarker names and what they measure are in the context — use those labels to reason about which biomarkers are relevant to the question. Do not guess codes from memory.
- If the question is about data not available in LinAge2, say: "This information is not available in LinAge2. Please refer to the patient's EHR."
- Only answer what was asked. If the question is about blood work, do not mention PC contributions, disease risks, or biological age. If the question is about aging, do not list raw biomarker values.

SYNTHESIS:
- End every answer with one sentence stating the single most actionable clinical implication, grounded in this patient's specific values.
- Never give generic advice. Every synthesis sentence must cite an actual number from the context.
- Never add filler like "if you need more details", "consult clinical guidelines", "feel free to ask", or "for further interpretation". Stop after the synthesis sentence.
"""


def build_patient_text_summary(data: dict) -> str:
    seqn = data.get("seqn", "unknown")
    gender = data.get("gender", "unknown")
    chron_age = data.get("latest_chron_age", "N/A")
    bio_age = data.get("latest_bio_age", "N/A")
    delta = data.get("latest_delta", "N/A")
    aging_status = "aging faster than normal" if delta and float(delta) > 0 else "aging slower than normal"

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


def fetch_and_store_patient(patient_uuid: str, token: str) -> dict | None:
    logger.info(f"Fetching patient {patient_uuid} from ALIS API")
    data = fetch_patient(patient_uuid, token=token)
    if not data:
        logger.warning(f"No data returned from ALIS API for patient {patient_uuid}")
        return None

    heatmap = data.get("latest_heatmap", {})
    label_to_human = {row["label"]: row["human"] for row in heatmap.get("rows", [])}
    total_pc_contributions = heatmap.get("total_pc_contributions", {})

    significant_pcs = {k: v for k, v in total_pc_contributions.items() if v != 0}
    sorted_pcs = sorted(significant_pcs.items(), key=lambda x: abs(x[1]), reverse=True)

    text_summary = build_patient_text_summary(data)  # ← was missing
    vector = embed_text(text_summary)                 # ← was missing

    payload = {
        "seqn": data.get("seqn"),
        "gender": data.get("gender"),
        "latest_chron_age": data.get("latest_chron_age"),
        "latest_bio_age": data.get("latest_bio_age"),
        "latest_delta": data.get("latest_delta"),
        "label_to_human": label_to_human,
        "biomarkers": data.get("biomarkers", {}),
        "risks": data.get("risks", []),
        "total_pc_contributions": total_pc_contributions,
        "significant_pcs_ranked": sorted_pcs,
        "text_summary": text_summary
    }

    upsert_patient(patient_uuid, text_summary, vector, payload)
    logger.info(f"Patient {patient_uuid} stored in Qdrant successfully")
    return payload


def build_context(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None,
    token: Optional[str] = None,
    query_vector: Optional[list] = None,
) -> tuple[str, list[str], Optional[dict]]:
    logger.info(f"Building context | patient_id: {patient_id} | pc_group: {pc_group}")
    if query_vector is None:
        query_vector = embed_text(question)
    context_parts = []
    sources = []
    patient_payload = None

    if patient_id:
        patient_payload = search_patient(patient_id, query_vector)
        logger.info(f"Qdrant search result for {patient_id}: {'found' if patient_payload else 'not found'}")

        if not patient_payload:
            existing = list_patients()
            if not existing:
                logger.info("Collection empty — bulk ingesting all patients...")
                from app.services.alis_api import fetch_all_patients
                all_patients = fetch_all_patients(token=token)
                for p in all_patients:
                    pid = p.get("id")
                    if pid:
                        fetch_and_store_patient(pid, token=token)
                # now try search again
                patient_payload = search_patient(patient_id, query_vector)

            # if still not found fetch just this one
            if not patient_payload:
                logger.info(f"Patient {patient_id} not in Qdrant — fetching from ALIS API")
                patient_payload = fetch_and_store_patient(patient_id, token=token)
                
        if patient_payload:
            from app.services.codebook import get_label
            
            biomarkers = patient_payload.get("biomarkers", {})
            risks = patient_payload.get("risks", [])

            

            logger.info(f"Patient {patient_id} biomarkers: {biomarkers}")
            logger.info(f"Patient {patient_id} risks: {risks}")
            biomarker_str = "\n".join([
                f"{get_label(k)} ({k}): {v}"
                for k, v in biomarkers.items()
                if v is not None and k not in ["id", "patient_id", "source", "created_at", "updated_at"]
            ])

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

            # pre-built markdown table for "most significant PCs" questions
            pc_table_rows = ["| PC | Contribution | Direction |", "|---|---|---|"]
            for pc, val in significant_pcs[:8]:
                direction = "Aging faster" if val > 0 else "Protective"
                pc_table_rows.append(f"| {pc} | {val:+.3f} yrs | {direction} |")
            pc_table = "\n".join(pc_table_rows)

            context_parts.append(f"\n=== PC Contributions Table ===\n{pc_table}")

            context_parts.append("=== Patient Profile ===")
            context_parts.append(
                f"Patient ID: {patient_id} | "
                f"SEQN: {patient_payload.get('seqn')} | "
                f"Gender: {patient_payload.get('gender')} | "
                f"Chronological Age: {patient_payload.get('latest_chron_age')} | "
                f"Biological Age: {patient_payload.get('latest_bio_age')} | "
                f"Delta: {patient_payload.get('latest_delta')}"
            )
            context_parts.append(f"\n=== Biomarkers ===\n{biomarker_str}")
            context_parts.append(f"\n=== Disease Risks ===\n{risk_str}")
            logger.info(f"Patient {patient_id} context parts: {context_parts}")
            sources.append(f"patient:{patient_id}")
            logger.info(f"Patient context built successfully for {patient_id}")
        else:
            logger.warning(f"No data found for patient {patient_id}")
            context_parts.append(f"No data found for patient {patient_id}.")

    pc_hits = search_pc_knowledge(query_vector, pc_group=pc_group, limit=3)
    logger.info(f"PC knowledge hits: {len(pc_hits)}")
    if pc_hits:
        patient_gender = (patient_payload or {}).get("gender", "").lower()  # "female" or "male"
        total_pc_contributions = (patient_payload or {}).get("total_pc_contributions", {})
        context_parts.append("\n=== PC Clinical Interpretation ===")
        for hit in pc_hits:
            hit_pc_group = hit.get("pc_group", "")
            # detect gender suffix on the PC group name (e.g. PC1M, PC2F)
            hit_suffix = hit_pc_group[-1].lower() if hit_pc_group and hit_pc_group[-1].lower() in ("m", "f") else None
            gender_mismatch = (
                hit_suffix == "m" and patient_gender == "female"
                or hit_suffix == "f" and patient_gender == "male"
            )
            if gender_mismatch:
                correct_suffix = "F" if hit_suffix == "m" else "M"
                correct_pc_group = hit_pc_group[:-1] + correct_suffix
                # the patient's contribution key is the number part only e.g. "PC1"
                pc_number = hit_pc_group[:-1]  # e.g. "PC1" from "PC1M"
                patient_pc_value = total_pc_contributions.get(pc_number)

                context_parts.append(
                    f"NOTE: The clinician asked about {hit_pc_group} but this patient is {patient_gender}. "
                    f"The gender-appropriate knowledge is {correct_pc_group}."
                )
                if patient_pc_value is not None:
                    context_parts.append(
                        f"This patient's {pc_number} contribution is {patient_pc_value:+.3f}."
                    )

                # fetch the correct gender's PC knowledge and include it
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

    context_str = "\n".join(context_parts) if context_parts else "No relevant context found."
    logger.info("Context built after searching pc knowledge {context_str}")
    return context_str, sources, patient_payload


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


def rag_query_stream(question: str, patient_id: Optional[str] = None, pc_group: Optional[str] = None, token: Optional[str] = None) -> tuple:
    logger.info(f"RAG stream query | question: {question[:50]}...")
    context, sources, _ = build_context(question, patient_id, pc_group, token=token)
    prompt = build_prompt(question, context)
    stream = stream_llm(prompt, system_prompt=SYSTEM_PROMPT)
    return stream, sources

def rag_query(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[str, list[str]]:

    # STEP 1: embed first (blocking)
    query_vector = embed_text(question)

    # STEP 2: then ask LLM for variable extraction (after embed is done)
    biomarkers, pcs = [], []
    if patient_id:
        available = get_force_included_variables()
        biomarkers, pcs = _extract_variables_with_llm(
            question,
            available,
            lambda p: call_llm(p),
        )

    needs_longitudinal = bool(biomarkers or pcs)
    logger.info(f"Longitudinal decision | needs={needs_longitudinal} | biomarkers={biomarkers} | pcs={pcs}")

    # STEP 3: NOW parallel is safe — both are API calls, not Ollama
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        context_future = executor.submit(
            build_context, question, patient_id, pc_group, token, query_vector
        )
        longitudinal_future = executor.submit(
            fetch_longitudinal,
            patient_id=patient_id,
            token=token,
            biomarkers=biomarkers,
            pcs=pcs,
        ) if (patient_id and needs_longitudinal) else None

    context, sources, patient_payload = context_future.result()

    if longitudinal_future:
        longitudinal_data = longitudinal_future.result()
        if longitudinal_data:
            from app.services.longitudinal import LONGITUDINAL_SYSTEM_PROMPT
            long_context = _format_longitudinal_context(
                question=question,
                data=longitudinal_data,
                biomarkers_requested=biomarkers,
                pcs_requested=pcs,
                patient_payload=patient_payload,
            )
            sources.append(f"longitudinal:{patient_id}")
            prompt = build_prompt(question, long_context)
            answer = call_llm(prompt, system_prompt=LONGITUDINAL_SYSTEM_PROMPT)
            return answer, sources

    prompt = build_prompt(question, context)
    answer = call_llm(prompt, system_prompt=SYSTEM_PROMPT)
    return answer, sources