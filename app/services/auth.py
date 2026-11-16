"""
Checks the ALIS user token ourselves, so /atlas can serve stored data without calling ALIS.

The token is a JWT that ALIS signs with ALIS_JWT_SECRET (HS256). A valid signature proves ALIS
issued it; its claims carry the user's clinic_id, matched against the patient's clinic.
"""

import logging
import os

import jwt
from fastapi import HTTPException

logger = logging.getLogger(__name__)

ALIS_JWT_SECRET = os.getenv("ALIS_JWT_SECRET")


def verify_token(authorization: str) -> dict:
    """The token's claims; 401 when it is missing, forged or expired. Until ALIS_JWT_SECRET is set,
    the token isn't verified (nginx's X-API-Key is then the only gate) and no claims are returned."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")
    token = authorization[7:] if authorization.lower().startswith("bearer ") else authorization
    if not ALIS_JWT_SECRET:
        logger.warning("auth | ALIS_JWT_SECRET is not set; token not verified (X-API-Key only)")
        return {}
    try:
        return jwt.decode(token, ALIS_JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"auth | invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


def check_clinic(claims: dict, patient: dict, patient_id: str) -> None:
    """403 when the patient belongs to a different clinic than the user. Skipped while the stored
    patient has no clinic_id (ALIS's bulk endpoint doesn't send it yet)."""
    clinic = patient.get("clinic_id")
    if clinic and claims and claims.get("clinic_id") != clinic:
        logger.warning(f"auth | clinic {claims.get('clinic_id')} may not see patient {patient_id}")
        raise HTTPException(status_code=403, detail="Not authorized for this patient")
