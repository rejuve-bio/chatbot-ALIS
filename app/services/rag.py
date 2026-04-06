
import os
from dotenv import load_dotenv
from typing import Optional, AsyncGenerator

from app.services.llm_service import embed_text, call_llm, stream_llm
from app.services.qdrant_service import (
    search_patient_biomarkers,
    search_pc_knowledge,
    upsert_patient_chunks,
    upsert_pc_chunks
)
from app.services.parsers.excel_parser import parse_excel
from app.services.parsers.pdf_parser import parse_pdf
from app.services.llm_service import embed_batch

load_dotenv()

SYSTEM_PROMPT = """
You are a clinical assistant for the LinAge2 longevity research platform.
You help clinicians understand patient biomarker data, biological aging patterns,
and clinical interpretations based on PC (principal component) groupings.

Guidelines:
- Always be precise and clinical in your responses
- You will be given raw biomarker readings across multiple timepoints — analyze the numbers yourself and identify trends, improvements, or concerning patterns
- Reference specific values and timepoints when making observations
- If a patient ID is mentioned, focus your answer on that patient's data
- When PC groups are relevant, explain what they mean clinically
- If the context does not contain enough information to answer, say so clearly
- Never make up values that are not in the provided context
- Format your responses in a clean readable way for clinicians
"""



# ── File Ingestion ─────────────────────────────────────────────────────────────

def ingest_excel(file_bytes: bytes, patient_id: str, pc_group: Optional[str] = None):
    """
    Parse excel file → embed each chunk → upsert to Qdrant.
    Returns number of chunks ingested.
    """
    print(f"Parsing Excel for patient: {patient_id}")
    chunks = parse_excel(file_bytes, patient_id, pc_group)

    if not chunks:
        print("No chunks extracted from Excel")
        return 0

    texts = [chunk["text_summary"] for chunk in chunks]
    vectors = embed_batch(texts)

    upsert_patient_chunks(chunks, vectors)
    print(f"Ingested {len(chunks)} biomarker chunks for patient {patient_id}")
    return len(chunks)


def ingest_pdf(file_bytes: bytes):
    """
    Parse PDF → embed each PC chunk → upsert to Qdrant.
    Returns number of chunks ingested.
    """
    print("Parsing PDF for PC knowledge...")
    chunks = parse_pdf(file_bytes)

    if not chunks:
        print("No chunks extracted from PDF")
        return 0

    texts = [chunk["raw_text"] for chunk in chunks]
    vectors = embed_batch(texts)

    upsert_pc_chunks(chunks, vectors)
    print(f"Ingested {len(chunks)} PC knowledge chunks")
    return len(chunks)


# ── Context Builder ────────────────────────────────────────────────────────────
def build_context(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None
) -> tuple[str, list[str]]:
    query_vector = embed_text(question)

    patient_hits = search_patient_biomarkers(
        query_vector=query_vector,
        patient_id=patient_id,
        limit=5
    )

    pc_hits = search_pc_knowledge(
        query_vector=query_vector,
        pc_group=pc_group,
        limit=3
    )

    context_parts = []
    sources = []

    if patient_hits:
        context_parts.append("=== Patient Biomarker Data ===")
        for hit in patient_hits:
            readings_str = ", ".join([f"{k}: {v}" for k, v in hit.get("readings", {}).items()])
            context_parts.append(
                f"Patient: {hit.get('patient_id')} | "
                f"Parameter: {hit.get('param_name')} ({hit.get('param_code')}) | "
                f"Reference Range: {hit.get('reference_range')} | "
                f"Readings: {readings_str}"
            )
            sources.append(f"biomarker:{hit.get('param_code')}:{hit.get('patient_id')}")

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
            sources.append(f"pc_knowledge:{hit.get('pc_group')}:{hit.get('risk_window')}")

    context_str = "\n".join(context_parts) if context_parts else "No relevant context found."
    return context_str, sources
    

# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str) -> str:
    return f"""
You have been provided with the following clinical context retrieved from the LinAge2 database:

{context}

---

Based on the above context, answer the following clinician question:

{question}
"""


# ── Main RAG Functions ─────────────────────────────────────────────────────────

def rag_query(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None
) -> tuple[str, list[str]]:
    """
    Full RAG pipeline — non-streaming.
    Returns (answer, sources).
    """
    context, sources = build_context(question, patient_id, pc_group)
    prompt = build_prompt(question, context)
    answer = call_llm(prompt, system_prompt=SYSTEM_PROMPT)
    return answer, sources


def rag_query_stream(
    question: str,
    patient_id: Optional[str] = None,
    pc_group: Optional[str] = None
) -> tuple[AsyncGenerator, list[str]]:
    """
    Full RAG pipeline — streaming version.
    Returns (generator, sources).
    """
    context, sources = build_context(question, patient_id, pc_group)
    prompt = build_prompt(question, context)
    stream = stream_llm(prompt, system_prompt=SYSTEM_PROMPT)
    return stream, sources