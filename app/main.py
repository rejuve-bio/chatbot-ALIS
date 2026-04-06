
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import router
from app.services.qdrant_service import init_collections


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs on startup
    print("Initializing Qdrant collections...")
    init_collections()
    yield
    # runs on shutdown (nothing to clean up for now)
    print("Shutting down...")


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