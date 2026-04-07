
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import router
from app.services.qdrant_service import init_collections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing Qdrant collections...")
    init_collections()
    logger.info("Qdrant collections ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="LinAge2 Clinical Chatbot",
    description="RAG-based chatbot for clinicians to query patient biomarker and PC interpretation data",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)