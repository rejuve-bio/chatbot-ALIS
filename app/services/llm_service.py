import httpx
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

LLM_HOST = os.getenv("LLM_HOST", "http://202.181.159.222:8002")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

LLM_HOST_2 = os.getenv("LLM_HOST_2")
LLM_API_KEY_2 = os.getenv("LLM_API_KEY_2", "")


_EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading local embedding model {_EMBEDDING_MODEL_NAME} (first use — one-time cost)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME, device="cpu")
        logger.info("Local embedding model loaded")
    return _embedding_model


def preload_embedding_model() -> None:
    """Loads the model at startup instead of on the first real request."""
    _get_embedding_model()


def embed_text(text: str, retries: int = 3) -> list[float]:
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return vectors.tolist()


def _clean_response(text: str) -> str:
    import re
    text = text.replace("\\n", "\n")
    result = re.sub(r"\n{3,}", "\n\n", text.strip())
    return result.strip()


def _fix_longitudinal_markdown(text: str) -> str:
    import re
    text = text.replace("\\n", "\n")
    text = re.sub(r'\n?(##\s)', r'\n\n\1', text)
    text = re.sub(r'\n?(-\s\d{4}-)', r'\n\n\1', text)
    text = re.sub(r'\n?(Trend:)', r'\n\n\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def call_llm(
    prompt: str,
    system_prompt: str = None,
    raw_markdown: bool = False,
    history: list[dict] = None,
    use_secondary: bool = False,
) -> str:
    """use_secondary=True routes to LLM_HOST_2 — for a call meant to run concurrently with another on the default host."""
    host = LLM_HOST_2 if use_secondary else LLM_HOST
    api_key = LLM_API_KEY_2 if use_secondary else LLM_API_KEY

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.post(
        f"{host}/v1/chat/completions",
        headers=headers,
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=120.0
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if raw_markdown:
        return _fix_longitudinal_markdown(content)
    return _clean_response(content)
