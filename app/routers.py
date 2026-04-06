
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional

from app.schema import ChatResponse, HealthCheckResponse
from app.services.rag import ingest_excel, ingest_pdf, rag_query, rag_query_stream
from app.services.qdrant_service import list_patients, check_qdrant_health
from app.services.llm_service import embed_text

router = APIRouter()


# ── Health Check ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    qdrant_status = check_qdrant_health()

    # check ollama by doing a tiny embed
    try:
        embed_text("ping")
        ollama_status = "ok"
    except Exception as e:
        ollama_status = f"unreachable: {str(e)}"

    overall = "ok" if qdrant_status == "ok" and ollama_status == "ok" else "degraded"

    return HealthCheckResponse(
        status=overall,
        qdrant=qdrant_status,
        ollama=ollama_status
    )


# ── List Patients ──────────────────────────────────────────────────────────────

@router.get("/patients")
def get_patients():
    try:
        patients = list_patients()
        return {"patients": patients, "total": len(patients)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat ───────────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    message: str = Form(...),
    patient_id: Optional[str] = Form(None),
    pc_group: Optional[str] = Form(None),
    stream: Optional[bool] = Form(False),
    file: Optional[UploadFile] = File(None)
):
    """
    Main chat endpoint.
    - Accepts a message from the clinician
    - Optionally accepts a file (Excel or PDF) to ingest before answering
    - Optionally filter by patient_id or pc_group
    - Returns streamed or full response
    """

    # if a file was uploaded, ingest it first before answering
    if file:
        file_bytes = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            if not patient_id:
                raise HTTPException(
                    status_code=400,
                    detail="patient_id is required when uploading an Excel file"
                )
            count = ingest_excel(file_bytes, patient_id, pc_group)
            if count == 0:
                raise HTTPException(
                    status_code=422,
                    detail="Excel file was uploaded but no data could be extracted"
                )

        elif filename.endswith(".pdf"):
            count = ingest_pdf(file_bytes)
            if count == 0:
                raise HTTPException(
                    status_code=422,
                    detail="PDF was uploaded but no PC knowledge could be extracted"
                )

        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Only .xlsx and .pdf are accepted"
            )

    # streaming response
    if stream:
        generator, sources = rag_query_stream(
            question=message,
            patient_id=patient_id,
            pc_group=pc_group
        )

        def event_stream():
            for chunk in generator:
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/plain")

    # non-streaming response
    answer, sources = rag_query(
        question=message,
        patient_id=patient_id,
        pc_group=pc_group
    )

    return ChatResponse(answer=answer, sources=sources)