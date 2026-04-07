
import httpx
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ALIS_API_URL = os.getenv("ALIS_API_URL", "http://37.27.231.93:5001")
ALIS_API_TOKEN = os.getenv("ALIS_API_TOKEN", "")


def get_headers():
    headers = {"Content-Type": "application/json"}
    if ALIS_API_TOKEN:
        headers["Authorization"] = f"Bearer {ALIS_API_TOKEN}"
    return headers


def fetch_patient(patient_uuid: str) -> dict | None:
    url = f"{ALIS_API_URL}/patients/{patient_uuid}"
    logger.info(f"Fetching patient from ALIS API: {url}")
    try:
        response = httpx.get(url, headers=get_headers(), timeout=30.0)
        if response.status_code == 404:
            logger.warning(f"Patient {patient_uuid} not found in ALIS API")
            return None
        response.raise_for_status()
        logger.info(f"Successfully fetched patient {patient_uuid}")
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch patient {patient_uuid} from ALIS API: {e}")
        return None


def fetch_all_patients() -> list[dict]:
    url = f"{ALIS_API_URL}/patients"
    logger.info(f"Fetching all patients from ALIS API: {url}")
    try:
        response = httpx.get(url, headers=get_headers(), timeout=30.0)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Fetched {len(data)} patients from ALIS API")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch patients from ALIS API: {e}")
        return []