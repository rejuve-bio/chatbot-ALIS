
import logging
from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Optional

from app.schema import ChatResponse, HealthCheckResponse
from app.services.rag import rag_query, rag_query_stream, ingest_excel, ingest_pdf, fetch_and_store_patient
from app.services.qdrant_service import list_patients, check_qdrant_health
from app.services.llm_service import embed_text
from fastapi import APIRouter, Form, HTTPException, UploadFile, File, Header

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    logger.info("Health check requested")
    qdrant_status = check_qdrant_health()
    try:
        embed_text("ping")
        ollama_status = "ok"
    except Exception as e:
        ollama_status = f"unreachable: {str(e)}"
        logger.error(f"Ollama unreachable: {e}")
    overall = "ok" if qdrant_status == "ok" and ollama_status == "ok" else "degraded"
    logger.info(f"Health status: {overall} | qdrant: {qdrant_status} | ollama: {ollama_status}")
    return HealthCheckResponse(status=overall, qdrant=qdrant_status, ollama=ollama_status)


@router.get("/patients")
def get_patients():
    logger.info("Fetching patient list")
    try:
        patients = list_patients()
        logger.info(f"Found {len(patients)} patients")
        return {"patients": patients, "total": len(patients)}
    except Exception as e:
        logger.error(f"Failed to fetch patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patients/resync")
async def resync_patients(authorization: str = Header(None)):
    """Re-fetch all patients from ALIS API and update Qdrant with latest data (names, events, biomarkers)."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    from app.services.alis_api import fetch_all_patients
    patients = fetch_all_patients(token=authorization)
    if not patients:
        raise HTTPException(status_code=502, detail="No patients returned from ALIS API")
    updated, errors = 0, []
    for p in patients:
        pid = p.get("id")
        if not pid:
            continue
        try:
            fetch_and_store_patient(pid, token=authorization)
            updated += 1
        except Exception as e:
            logger.error(f"Failed to resync patient {pid}: {e}")
            errors.append(pid)
    logger.info(f"Resync complete | updated={updated} | errors={len(errors)}")
    return {"updated": updated, "errors": errors}


@router.post("/chat")
async def chat(
    message: str = Form(...),
    patient_id: Optional[str] = Form(None),
    pc_group: Optional[str] = Form(None),
    stream: Optional[bool] = Form(False),
    authorization = Header(None)
):

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")

    logger.info(f"Chat request | patient_id: {patient_id} | pc_group: {pc_group} | message: {message}")

    # if stream:
    #     logger.info("Streaming response requested")
    #     generator, sources = rag_query_stream(
    #         question=message,
    #         patient_id=patient_id,
    #         pc_group=pc_group,
    #     )
    #     def event_stream():
    #         for chunk in generator:
    #             yield chunk
    #     return StreamingResponse(event_stream(), media_type="text/plain")
    logger.info(f"Authorization header received: {authorization}")
    logger.info(f"Question is {message} with patient id {patient_id}")
    answer, sources = rag_query(
        question=message,
        patient_id=patient_id,
        pc_group=pc_group,
        token = authorization
    )


    logger.info(f"Chat response generated | sources: {sources} and the response is {answer}")
    return ChatResponse(answer=answer, sources=sources)


@router.post("/ingest")
async def ingest(
    patient_id: str = Form(...),
    pc_group: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    logger.info(f"Ingest request | patient_id: {patient_id} | file: {file.filename}")
    file_bytes = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        count = ingest_excel(file_bytes, patient_id, pc_group)
    elif filename.endswith(".pdf"):
        count = ingest_pdf(file_bytes)
    else:
        logger.error(f"Unsupported file type: {filename}")
        raise HTTPException(status_code=415, detail="Only .xlsx and .pdf accepted")

    logger.info(f"Ingested {count} chunks from {filename}")
    return {"ingested": count}