
from pydantic import BaseModel
from typing import Optional
from enum import Enum


class FileType(str, Enum):
    excel = "excel"
    pdf = "pdf"


class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None  # if clinician is asking about specific patient


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = []  # which chunks were used


class PatientRecord(BaseModel):
    patient_id: str
    param_code: str
    param_name: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    readings: dict          # {"day_1": 15.4, "day_91": 15.89, ...}
    trend: Optional[str] = None       # "declining", "stable", "improving"
    latest_value: Optional[float] = None
    out_of_range: Optional[bool] = False
    pc_group: Optional[str] = None


class PCKnowledgeRecord(BaseModel):
    pc_group: str           # "PC1M", "PC2M", etc.
    risk_window: str        # "early" or "late"
    causes_of_death: list[str] = []
    diseases: list[str] = []
    mechanisms: list[str] = []
    interventions: list[str] = []
    raw_text: str           # full text used for embedding


class HealthCheckResponse(BaseModel):
    status: str
    qdrant: str
    ollama: str