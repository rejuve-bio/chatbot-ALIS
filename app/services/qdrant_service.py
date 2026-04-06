
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
PATIENT_COLLECTION = os.getenv("PATIENT_COLLECTION", "patient_biomarkers")
PC_COLLECTION = os.getenv("PC_COLLECTION", "pc_knowledge")

VECTOR_SIZE = 768  # nomic-embed-text output size

client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT)


# ── Collection Setup ───────────────────────────────────────────────────────────

def init_collections():
    """
    Creates collections if they don't exist yet.
    Call this once on startup.
    """
    existing = [c.name for c in client.get_collections().collections]

    if PATIENT_COLLECTION not in existing:
        client.create_collection(
            collection_name=PATIENT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {PATIENT_COLLECTION}")

    if PC_COLLECTION not in existing:
        client.create_collection(
            collection_name=PC_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {PC_COLLECTION}")


# ── Upsert ─────────────────────────────────────────────────────────────────────
def upsert_patient_chunks(chunks: list[dict], vectors: list[list[float]]):
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "patient_id": chunk["patient_id"],
                "param_code": chunk["param_code"],
                "param_name": chunk["param_name"],
                "reference_range": chunk.get("reference_range", ""),
                "readings": chunk.get("readings", {}),
                "pc_group": chunk.get("pc_group", ""),
                "text_summary": chunk.get("text_summary", "")
            }
        ))
    client.upsert(collection_name=PATIENT_COLLECTION, points=points)
    print(f"Upserted {len(points)} patient chunks")
    

def upsert_pc_chunks(chunks: list[dict], vectors: list[list[float]]):
    """
    Upserts PC knowledge chunks from the PDF into Qdrant.
    """
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "pc_group": chunk["pc_group"],
                "risk_window": chunk["risk_window"],
                "causes_of_death": chunk.get("causes_of_death", []),
                "diseases": chunk.get("diseases", []),
                "mechanisms": chunk.get("mechanisms", []),
                "interventions": chunk.get("interventions", []),
                "raw_text": chunk.get("raw_text", "")
            }
        ))

    client.upsert(collection_name=PC_COLLECTION, points=points)
    print(f"Upserted {len(points)} PC knowledge chunks")


# ── Search ─────────────────────────────────────────────────────────────────────

def search_patient_biomarkers(
    query_vector: list[float],
    patient_id: str = None,
    limit: int = 5
) -> list[dict]:
    """
    Semantic search in patient_biomarkers.
    Optionally filter by patient_id.
    """
    search_filter = None
    if patient_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="patient_id",
                    match=MatchValue(value=patient_id)
                )
            ]
        )

    results = client.search(
        collection_name=PATIENT_COLLECTION,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit
    )

    return [hit.payload for hit in results]


def search_pc_knowledge(
    query_vector: list[float],
    pc_group: str = None,
    limit: int = 3
) -> list[dict]:
    """
    Semantic search in pc_knowledge.
    Optionally filter by pc_group.
    """
    search_filter = None
    if pc_group:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="pc_group",
                    match=MatchValue(value=pc_group.upper())
                )
            ]
        )

    results = client.search(
        collection_name=PC_COLLECTION,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit
    )

    return [hit.payload for hit in results]


# ── Patients List ──────────────────────────────────────────────────────────────

def list_patients() -> list[str]:
    """
    Scrolls through patient_biomarkers and returns unique patient IDs.
    """
    results, _ = client.scroll(
        collection_name=PATIENT_COLLECTION,
        limit=1000,
        with_payload=True
    )

    patient_ids = list(set(
        hit.payload["patient_id"]
        for hit in results
        if "patient_id" in hit.payload
    ))

    return sorted(patient_ids)


# ── Health ─────────────────────────────────────────────────────────────────────

def check_qdrant_health() -> str:
    try:
        client.get_collections()
        return "ok"
    except Exception as e:
        return f"unreachable: {str(e)}"