
import logging
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

from app.routers import router
from app.services.qdrant_service import init_collections
from app.services.biology_service import init_biology_collections
from app.services.llm_service import preload_embedding_model
from app.services.codebook import load_codebook
load_codebook()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — preloading embedding model...")
    preload_embedding_model()
    logger.info("Embedding model ready")
    logger.info("Starting up — initializing Qdrant collections...")
    init_collections()
    logger.info("Qdrant collections ready")
    logger.info("Initializing biology-evidence collections (creates + populates only what's missing)...")
    init_biology_collections()
    logger.info("Biology-evidence collections ready")
    from app.services.atlas_v2 import warm_treatment_cache
    threading.Thread(target=warm_treatment_cache, name="atlas-warmup", daemon=True).start()
    logger.info("Atlas treatment warm-up started in the background")
    from app.services.atlas_precompute import precompute_loop
    threading.Thread(target=precompute_loop, name="atlas-precompute", daemon=True).start()
    logger.info("Atlas daily precompute started in the background")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="LinAge2 Clinical Chatbot",
    description="RAG-based chatbot for clinicians to query patient biomarker and PC interpretation data",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)