
import os
import time
import uuid
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

load_dotenv()

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_HOST", "http://localhost:6333")
PATIENT_COLLECTION = os.getenv("PATIENT_COLLECTION", "patient_data")
PC_COLLECTION = os.getenv("PC_COLLECTION", "pc_knowledge")
VECTOR_SIZE = 1024

logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
client = QdrantClient(url=QDRANT_URL)


def _with_retries(fn, description: str, retries: int = 3):
    """Retries a Qdrant call on transient connection drops (seen in practice under load)."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            logger.warning(f"{description} attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(1 * attempt)
    logger.error(f"{description} failed after {retries} attempts: {last_error}")
    raise last_error


def _populate_pc_collection():
    try:
        from data.pc_chunks import PC_CHUNKS
        from app.services.llm_service import embed_batch
        texts = [chunk["raw_text"] for chunk in PC_CHUNKS]
        vectors = embed_batch(texts)
        upsert_pc_chunks(PC_CHUNKS, vectors)
        logger.info(f"PC collection auto-populated with {len(PC_CHUNKS)} chunks")
    except Exception as e:
        logger.error(f"Failed to auto-populate PC collection: {e}")


def init_collections():
    logger.info("Initializing Qdrant collections")
    existing = [c.name for c in client.get_collections().collections]

    if PATIENT_COLLECTION not in existing:
        client.create_collection(
            collection_name=PATIENT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Created collection: {PATIENT_COLLECTION}")
    else:
        logger.info(f"Collection already exists: {PATIENT_COLLECTION}")

    if PC_COLLECTION not in existing:
        client.create_collection(
            collection_name=PC_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Created collection: {PC_COLLECTION}")
    else:
        logger.info(f"Collection already exists: {PC_COLLECTION}")

    pc_count = client.count(collection_name=PC_COLLECTION).count
    if pc_count == 0:
        logger.info("PC collection is empty — auto-populating from built-in chunks")
        _populate_pc_collection()
    else:
        logger.info(f"PC collection already has {pc_count} points — skipping population")


def get_patient_count() -> int:
    return client.count(collection_name=PATIENT_COLLECTION).count


def get_collection_count(collection_name: str) -> int:
    return client.count(collection_name=collection_name).count


def upsert_patient(patient_uuid: str, text_summary: str, vector: list[float], payload: dict):
    logger.info(f"Upserting patient {patient_uuid} into Qdrant")
    client.upsert(
        collection_name=PATIENT_COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, patient_uuid)),
            vector=vector,
            payload={"patient_id": patient_uuid, **payload}
        )]
    )
    logger.info(f"Patient {patient_uuid} upserted successfully")


def search_patient(patient_uuid: str, query_vector: list[float]) -> dict | None:
    logger.info(f"Searching Qdrant for patient {patient_uuid}")
    results = client.query_points(
        collection_name=PATIENT_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(
                key="patient_id",
                match=MatchValue(value=patient_uuid)
            )]
        ),
        limit=1,
        with_payload=True
    )
    if results.points:
        logger.info(f"Patient {patient_uuid} found in Qdrant")
        logger.debug(f"Patient {patient_uuid} search results: {results.points[0].payload}")
        return results.points[0].payload
    logger.info(f"Patient {patient_uuid} not found in Qdrant")
    return None


def upsert_pc_chunks(chunks: list[dict], vectors: list[list[float]]):
    logger.info(f"Upserting {len(chunks)} PC knowledge chunks")
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
    logger.info(f"Upserted {len(points)} PC knowledge chunks successfully")


def upsert_patient_chunks(chunks: list[dict], vectors: list[list[float]]):
    logger.info(f"Upserting {len(chunks)} patient biomarker chunks")
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "patient_id": chunk["patient_id"],
                "param_code": chunk.get("param_code"),
                "param_name": chunk.get("param_name"),
                "reference_range": chunk.get("reference_range"),
                "readings": chunk.get("readings", {}),
                "pc_group": chunk.get("pc_group"),
                "text_summary": chunk.get("text_summary", "")
            }
        ))
    client.upsert(collection_name=PATIENT_COLLECTION, points=points)
    logger.info(f"Upserted {len(points)} patient biomarker chunks successfully")


def search_pc_knowledge(query_vector: list[float], pc_group: str = None, limit: int = 3) -> list[dict]:
    logger.info(f"Searching PC knowledge | pc_group filter: {pc_group}")
    search_filter = None
    if pc_group:
        search_filter = Filter(
            must=[FieldCondition(key="pc_group", match=MatchValue(value=pc_group.upper()))]
        )
    results = client.query_points(
        collection_name=PC_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True
    )
    logger.info(f"PC knowledge query_points returned {results} results")
    return [hit.payload for hit in results.points]


def list_patients() -> list[dict]:
    logger.info("Listing all patients from Qdrant")
    results, _ = client.scroll(
        collection_name=PATIENT_COLLECTION,
        limit=1000,
        with_payload=True,
    )
    seen_ids: set[str] = set()
    patients = []
    for hit in results:
        pid = hit.payload.get("patient_id")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            patients.append({"id": pid, "payload": hit.payload})
    logger.info(f"Found {len(patients)} patients")
    return patients

# ---------------------------------------------------------------------------

def create_collection(collection_name: str, vector_size: int = VECTOR_SIZE):
    """Create a Qdrant collection by name if it doesn't already exist. Safe to call repeatedly."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Created collection: {collection_name}")
    else:
        logger.info(f"Collection already exists: {collection_name}")


def delete_collection(collection_name: str):
    """Drop a collection entirely if it exists. Safe to call repeatedly."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    else:
        logger.info(f"Collection not found, nothing to delete: {collection_name}")


def upsert_chunks(collection_name: str, chunks: list[dict], vectors: list[list[float]]):
    """
    Generic dump function — upserts any list of chunk dicts + matching vectors
    into any collection. The chunk dict becomes the point's payload as-is,
    so callers control their own schema (pc_chunks, intervention chunks, etc.).
    """
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=chunk)
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Upserted {len(points)} chunks into '{collection_name}'")


def search_chunks(
    collection_name: str,
    query_vector: list[float],
    filters: dict | None = None,
    limit: int = 5,
    min_score: float | None = None,
) -> list[dict]:
    """
    Generic dynamic retrieval — pass whichever filter fields you have
    (e.g. {"disease": "Cancer"}, {"pc_group": "PC24"}, {"compound": "Metformin"}).
    None/empty filters means an unfiltered similarity search over the whole collection.

    min_score: if set, drops hits below this cosine similarity — top-k alone
    always returns *something* even for irrelevant queries, so callers that
    need to distinguish "actually relevant" from "just the closest available"
    (e.g. deciding whether to cite a source at all) should set this.
    """
    search_filter = None
    if filters:
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
            if value is not None
        ]
        if conditions:
            search_filter = Filter(must=conditions)

    results = _with_retries(
        lambda: client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        ),
        f"search_chunks({collection_name})",
    )
    points = results.points
    if min_score is not None:
        points = [p for p in points if p.score >= min_score]
    logger.info(f"search_chunks | collection={collection_name} | filters={filters} | "
                f"hits={len(points)} | min_score={min_score}")
    return [hit.payload for hit in points]


def filter_chunks(
    collection_name: str,
    filters: dict,
    limit: int = 20,
) -> list[dict]:
    """
    Exact-match lookup — no embedding/vector needed. Use this for structured
    lookups like "find the chunk where name == 'PRKAA1'", as opposed to
    search_chunks which does semantic similarity search.
    """
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
        if value is not None
    ]
    scroll_filter = Filter(must=conditions) if conditions else None

    points, _ = _with_retries(
        lambda: client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        ),
        f"filter_chunks({collection_name})",
    )
    payloads = [p.payload for p in points]
    logger.info(f"filter_chunks | collection={collection_name} | filters={filters} | hits={len(payloads)}")
    return payloads


def check_qdrant_health() -> str:
    try:
        client.get_collections()
        logger.info("Qdrant health check passed")
        return "ok"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return f"unreachable: {str(e)}"