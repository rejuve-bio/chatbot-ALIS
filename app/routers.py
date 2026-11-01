
import logging
from fastapi import APIRouter, Form, HTTPException, UploadFile, File, Header, Query
from typing import Optional, List

from app.schema import ChatResponse, HealthCheckResponse
from app.services.rag import rag_query, ingest_excel, ingest_pdf, fetch_and_store_patient
from app.services.qdrant_service import list_patients, check_qdrant_health
from app.services.llm_service import embed_text

logger = logging.getLogger(__name__)

from app.services.atlas_service import combined_mechanisms

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


def _fetch_patient_from_alis(patient_id: str, authorization: str) -> dict:
    """For a patient the precompute hasn't saved yet (added since its last run)."""
    from app.services.backend_api import BackendAuthError, fetch_patient
    try:
        patient_data = fetch_patient(patient_id, token=authorization, raise_auth_errors=True)
    except BackendAuthError:
        logger.warning(f"Atlas | ALIS API rejected the token for patient {patient_id}")
        raise HTTPException(status_code=401, detail="Token expired or not authorized for this patient")
    if not patient_data:
        logger.error(f"Atlas | could not fetch patient {patient_id} from ALIS API")
        raise HTTPException(status_code=502, detail=f"Could not fetch patient {patient_id} from ALIS API")
    return patient_data


@router.get("/atlas/{patient_id}/risk")
def get_atlas_for_risk(
    patient_id: str,
    disease: Optional[List[str]] = Query(None),
    pc: Optional[List[str]] = Query(None),
    authorization: str = Header(None),
):
    """Intervention Atlas for the given risk(s) of a patient; with neither disease nor pc, for all of them."""
    from app.services.auth import verify_token
    claims = verify_token(authorization)

    logger.info(f"Atlas request | patient_id={patient_id} | disease={disease or 'all risks'} | pc={pc}")

    from app.services.atlas_v2 import build_risk_area

    if pc:
        mechanisms, diseases = combined_mechanisms(pc)
        if not mechanisms:
            raise HTTPException(status_code=404, detail=f"No mechanism data found for pc_group(s): {pc}")
        disease_name = (disease[0] if disease else None) or (diseases[0] if diseases else None)
        logger.info(f"Atlas | patient={patient_id} | explicit pc override")
        return {"patient_id": patient_id, "risk_areas": [build_risk_area(disease_name, pc, mechanisms)]}

    from app.services import atlas_store
    from app.services.auth import check_clinic
    patient_data = atlas_store.get_patient(patient_id)
    if patient_data:
        check_clinic(claims, patient_data, patient_id)
    else:
        patient_data = _fetch_patient_from_alis(patient_id, authorization)

    suffix = "F" if (patient_data.get("gender") or "").lower() == "female" else "M"
    risks_by_name = {r.get("disease_name"): r for r in patient_data.get("risks", []) if r.get("disease_name")}
    if not disease:
        disease = list(risks_by_name)
        if not disease:
            return {"patient_id": patient_id, "risk_areas": []}
    not_found = [d for d in disease if d not in risks_by_name]
    if not_found:
        logger.warning(f"Atlas | disease(s) not found in patient {patient_id}'s risks: {not_found}")
        raise HTTPException(status_code=404, detail=f"Disease(s) not found in this patient's risks: {not_found}")

    def _build_one(disease_name: str) -> dict:
        pc_groups = [f"{pc_num}{suffix}" for pc_num in risks_by_name[disease_name].get("contributing_pcs", [])]
        return build_risk_area(disease_name, pc_groups)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(disease))) as executor:
        risk_areas = list(executor.map(_build_one, disease))

    logger.info(f"Atlas | patient={patient_id} | built {len(risk_areas)} risk area(s)")
    return {"patient_id": patient_id, "risk_areas": risk_areas}


@router.post("/patients/resync")
def resync_patients(authorization: str = Header(None)):
    """Re-fetch all patients from ALIS API and update Qdrant with latest data (names, events, biomarkers)."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    from app.services.backend_api import fetch_all_patients
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
def chat(
    message: str = Form(...),
    patient_id: Optional[str] = Form(None),
    pc_group: Optional[str] = Form(None),
    authorization = Header(None)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")

    logger.info(f"Chat request | patient_id: {patient_id} | pc_group: {pc_group} | message: {message}")

    answer, sources, resource = rag_query(
        question=message,
        patient_id=patient_id,
        pc_group=pc_group,
        token=authorization
    )

    logger.info(f"Chat response generated | sources: {sources} | resource: {resource} | response: {answer}")
    return ChatResponse(answer=answer, sources=sources, resource=resource)


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