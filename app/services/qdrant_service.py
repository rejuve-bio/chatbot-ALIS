
import os
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
VECTOR_SIZE = 768

logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
client = QdrantClient(url=QDRANT_URL)


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
    results = client.search(
        collection_name=PATIENT_COLLECTION,
        query_vector=query_vector,
        query_filter=Filter(
            must=[FieldCondition(
                key="patient_id",
                match=MatchValue(value=patient_uuid)
            )]
        ),
        limit=1,
        with_payload=True
    )
    if results:
        logger.info(f"Patient {patient_uuid} found in Qdrant")
        return results[0].payload
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


def search_pc_knowledge(query_vector: list[float], pc_group: str = None, limit: int = 3) -> list[dict]:
    logger.info(f"Searching PC knowledge | pc_group filter: {pc_group}")
    search_filter = None
    if pc_group:
        search_filter = Filter(
            must=[FieldCondition(key="pc_group", match=MatchValue(value=pc_group.upper()))]
        )
    results = client.search(
        collection_name=PC_COLLECTION,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True
    )
    logger.info(f"PC knowledge search returned {len(results)} results")
    return [hit.payload for hit in results]


def list_patients() -> list[str]:
    logger.info("Listing all patients from Qdrant")
    results, _ = client.scroll(
        collection_name=PATIENT_COLLECTION,
        limit=1000,
        with_payload=True
    )
    patients = sorted(set(
        hit.payload["patient_id"]
        for hit in results
        if "patient_id" in hit.payload
    ))
    logger.info(f"Found {len(patients)} patients")
    return patients


def check_qdrant_health() -> str:
    try:
        client.get_collections()
        logger.info("Qdrant health check passed")
        return "ok"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return f"unreachable: {str(e)}"