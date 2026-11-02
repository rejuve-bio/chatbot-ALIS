
import httpx
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ALIS_API_URL = os.getenv("ALIS_API_URL")


ALIS_SERVICE_KEY = os.getenv("ALIS_SERVICE_KEY")


def fetch_all_patient_risks(updated_since: str | None = None) -> list[dict]:
    """Every patient's gender and risks (same shape as GET /patients/{id}), without a user token."""
    if not ALIS_SERVICE_KEY:
        raise RuntimeError("ALIS_SERVICE_KEY is not set")
    url = f"{ALIS_API_URL}/internal/atlas/patients/risks"
    items, page = [], 1
    while True:
        params = {"page": page, "size": 1000}
        if updated_since:
            params["updated_since"] = updated_since
        response = httpx.get(url, params=params, headers={"X-Service-Key": ALIS_SERVICE_KEY}, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        items.extend(data.get("items", []))
        if page >= data.get("total_pages", 1):
            break
        page += 1
    logger.info(f"Fetched risks for {len(items)} patients from ALIS internal API")
    return items


class BackendAuthError(Exception):
    """The ALIS API rejected the caller's token (expired or not allowed)."""


def fetch_patient(patient_uuid: str, token: str, raise_auth_errors: bool = False) -> dict | None:
    url = f"{ALIS_API_URL}/patients/{patient_uuid}"
    headers = {"Content-Type": "application/json"}
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token
    logger.info(f"Fetching patient from ALIS API: {url}")
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)

        if response.status_code == 404:
            logger.warning(f"Patient {patient_uuid} not found in ALIS API")
            return None
        if response.status_code in (401, 403) and raise_auth_errors:
            raise BackendAuthError(f"ALIS API returned {response.status_code} for patient {patient_uuid}")
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched patient {patient_uuid}")
        return data
    except BackendAuthError:
        raise
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
        logger.info(f"Latest biomarker data fetched for {patient_id} | {len(data)} field(s)")
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
        logger.info(f"Longitudinal data fetched successfully for {patient_id}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch longitudinal data for {patient_id}: {e}")
        return None