
from pydantic import BaseModel
from typing import Optional


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = []  # which chunks were used


class HealthCheckResponse(BaseModel):
    status: str
    qdrant: str
    ollama: str