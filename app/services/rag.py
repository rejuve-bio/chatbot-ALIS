
import os
import logging
from typing import Optional
from dotenv import load_dotenv

from app.services.llm_service import embed_text, call_llm, stream_llm
from app.services.qdrant_service import (
    search_patient, search_pc_knowledge,
    upsert_patient
)
from app.services.alis_api import fetch_patient
from app.services.parsers.excel_parser import parse_excel
from app.services.parsers.pdf_parser import parse_pdf
from app.services.llm_service import embed_batch
from app.services.qdrant_service import upsert_pc_chunks

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a clinical assistant for the LinAge2 longevity research platform.
Answer clinician questions clearly and concisely.

Rules:
- Keep answers short and to the point
- Use plain text, no markdown, no headers, no bullet points
- Only include what is directly relevant to the question
- Analyze biomarker values yourself and identify patterns
- If data is missing say so in one sentence
- Never repeat the same information twice
- Never give medical advice, only clinical observations based on data
"""


def build_patient_text_summary(data: dict) -> str:
    seqn = data.get("seqn", "unknown")
    gender = data.get("gender", "unknown")
    chron_age = data.get("latest_chron_age", "N/A")
    bio_age = data.get("latest_bio_age", "N/A")
    delta = data.get("latest_delta", "N/A")
    aging_status = "aging faster than normal" if delta and float(delta) > 0 else "aging slower than normal"

    heatmap = data.get("latest_heatmap", {}).get("rows", [])
    label_to_human = {row["label"]: row["human"] for row in heatmap}

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

    return (
        f"Patient UUID: {data.get('id')} | SEQN: {seqn} | Gender: {gender} | "
        f"Chronological Age: {chron_age} | Biological Age: {bio_age} | "
        f"Delta: {delta} | Aging Status: {aging_status} | "
        f"Biomarkers: {biomarker_str} | "
        f"Top Disease Risks: {risk_str}"
    )


def fetch_and_store_patient(patient_uuid: str) -> dict | None:
    logger.info(f"Fetching patient {patient_uuid} from ALIS API")
    data = fetch_patient(patient_uuid)
    if not data:
        logger.warning(f"No data returned from ALIS API for patient {patient_uuid}")
        return None

    heatmap = data.get("latest_heatmap", {}).get("rows", [])
    label_to_human = {row["label"]: row["human"] for row in heatmap}

    text_summary = build_patient_text_summary(data)
    vector = embed_text(text_summary)

    payload = {
        "seqn": data.get("seqn"),
        "gender": data.get("gender"),
        "latest_chron_age": data.get("latest_chron_age"),
        "latest_bio_age": data.get("latest_bio_age"),
        "latest_delta": data.get("latest_delta"),
        "label_to_human": label_to_human,
        "biomarkers": data.get("biomarkers", {}),
        "risks": data.get("risks", []),
        "text_summary": text_summary
    }

    upsert_patient(patient_uuid, text_summary, vector, payload)
    logger.info(f"Patient {patient_uuid} stored in Qdrant successfully")
    return payload


def build_context(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None
) -> tuple[str, list[str]]:
    logger.info(f"Building context | patient_id: {patient_id} | pc_group: {pc_group}")
    query_vector = embed_text(question)
    context_parts = []
    sources = []

    if patient_id:
        patient_payload = search_patient(patient_id, query_vector)
        logger.info(f"Qdrant search result for {patient_id}: {'found' if patient_payload else 'not found'}")

        if not patient_payload:
            logger.info(f"Patient {patient_id} not in Qdrant — fetching from ALIS API")
            patient_payload = fetch_and_store_patient(patient_id)
            logger.info(f"ALIS API fetch result: {'success' if patient_payload else 'failed'}")

        if patient_payload:
            biomarkers = patient_payload.get("biomarkers", {})
            risks = patient_payload.get("risks", [])
            label_to_human = patient_payload.get("label_to_human", {})

            biomarker_str = "\n".join([
                f"{label_to_human.get(k, k)} ({k}): {v}"
                for k, v in biomarkers.items()
                if v is not None and k not in ["id", "patient_id", "source", "created_at", "updated_at"]
            ])

            risk_str = "\n".join([
                f"{r['disease_name']} — evidence score: {r['evidence_score']} — contributing PCs: {', '.join(r['contributing_pcs'])}"
                for r in risks
            ])

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
            sources.append(f"patient:{patient_id}")
            logger.info(f"Patient context built successfully for {patient_id}")
        else:
            logger.warning(f"No data found for patient {patient_id}")
            context_parts.append(f"No data found for patient {patient_id}.")

    pc_hits = search_pc_knowledge(query_vector, pc_group=pc_group, limit=3)
    logger.info(f"PC knowledge hits: {len(pc_hits)}")
    if pc_hits:
        context_parts.append("\n=== PC Clinical Interpretation ===")
        for hit in pc_hits:
            context_parts.append(
                f"PC Group: {hit.get('pc_group')} | "
                f"Risk Window: {hit.get('risk_window')} | "
                f"Causes of Death: {', '.join(hit.get('causes_of_death', []))} | "
                f"Associated Diseases: {', '.join(hit.get('diseases', []))} | "
                f"Mechanisms: {', '.join(hit.get('mechanisms', []))} | "
                f"Interventions: {', '.join(hit.get('interventions', []))}"
            )
            sources.append(f"pc_knowledge:{hit.get('pc_group')}")

    context_str = "\n".join(context_parts) if context_parts else "No relevant context found."
    return context_str, sources


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


def rag_query(question: str, patient_id: Optional[str] = None, pc_group: Optional[str] = None) -> tuple[str, list[str]]:
    logger.info(f"RAG query | question: {question[:50]}...")
    context, sources = build_context(question, patient_id, pc_group)
    prompt = build_prompt(question, context)
    answer = call_llm(prompt, system_prompt=SYSTEM_PROMPT)
    logger.info(f"RAG answer generated | sources: {sources}")
    return answer, sources


def rag_query_stream(question: str, patient_id: Optional[str] = None, pc_group: Optional[str] = None) -> tuple:
    logger.info(f"RAG stream query | question: {question[:50]}...")
    context, sources = build_context(question, patient_id, pc_group)
    prompt = build_prompt(question, context)
    stream = stream_llm(prompt, system_prompt=SYSTEM_PROMPT)
    return stream, sources