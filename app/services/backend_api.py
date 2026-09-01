
import httpx
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ALIS_API_URL = os.getenv("ALIS_API_URL", "http://37.27.231.93:5001")



def fetch_patient(patient_uuid: str, token: str) -> dict | None:
    url = f"{ALIS_API_URL}/patients/{patient_uuid}"
    headers = {"Content-Type": "application/json"}
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token
    logger.info(f"Sending Authorization header: {headers.get('Authorization')}")
    logger.info(f"Fetching patient from ALIS API: {url}")
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        
        if response.status_code == 404:
            logger.warning(f"Patient {patient_uuid} not found in ALIS API")
            return None
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched patient {patient_uuid} | API response: {data}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch patient {patient_uuid}: {e}")
        return None


def fetch_all_patients(token) -> list[dict]:
    url = f"{ALIS_API_URL}/patients"
    headers = {"Content-Type": "application/json"}
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token 
    logger.info(f"Sending Authorization header: {headers.get('Authorization')}")
    logger.info(f"Fetching all patients from ALIS API: {url}")
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        # response has items key with pagination
        items = data.get("items", [])
        logger.info(f"Fetched {len(items)} patients from ALIS API")
        return items
    except Exception as e:
        logger.error(f"Failed to fetch patients from ALIS API: {e}")
        return []


def fetch_biomarker_data_latest(patient_id: str, token: str) -> dict | None:
    url = f"{ALIS_API_URL}/patients/{patient_id}/biomarker-data/latest"
    headers = {"Content-Type": "application/json"}
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token
    logger.info(f"Fetching latest biomarker data for patient {patient_id}")
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        if response.status_code == 404:
            logger.warning(f"No biomarker data found for patient {patient_id}")
            return None
        response.raise_for_status()
        data = response.json()
        # If the API returns a list, take the most recent entry (last item)
        if isinstance(data, list):
            if not data:
                return None
            data = data[-1]
        _q_prefixes = ("BPQ", "DIQ", "MCQ", "KIQ", "HUQ", "OSQ", "PFQ")
        q_codes = {k: v for k, v in data.items() if k.startswith(_q_prefixes)}
        logger.info(f"biomarker-data/latest questionnaire fields for {patient_id}: {q_codes}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch biomarker data for {patient_id}: {e}")
        return None


def fetch_longitudinal(
    patient_id: str,
    token: str,
    biomarkers: list[str] = None,
    pcs: list[str] = None,
) -> dict | None:
    """
    Fetch longitudinal time series data for a patient.
    biomarkers: list of variable codes e.g. ["BPXPLS", "BPXSAR"]
    pcs: list of PC names e.g. ["PC1", "PC2"]
    """
    url = f"{ALIS_API_URL}/patients/{patient_id}/longitudinal"

    params = []
    for b in (biomarkers or []):
        params.append(("biomarkers", b))
    for p in (pcs or []):
        params.append(("pcs", p))

    headers = {"Content-Type": "application/json"}
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token

    logger.info(
        f"Fetching longitudinal data for patient {patient_id} | "
        f"biomarkers={biomarkers} | pcs={pcs}"
    )

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30.0)
        if response.status_code == 404:
            logger.warning(f"No longitudinal data for patient {patient_id}")
            return None
        response.raise_for_status()
        data = response.json()
        logger.info(f"Longitudinal data fetched successfully for {patient_id} {data}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch longitudinal data for {patient_id}: {e}")
        return None