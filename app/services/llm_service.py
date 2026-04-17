import httpx
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def embed_text(text: str) -> list[float]:
    response = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60.0
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for i, text in enumerate(texts):
        vector = embed_text(text)
        embeddings.append(vector)
        if i % 10 == 0:
            print(f"Embedded {i}/{len(texts)}")
    return embeddings


def _clean_response(text: str) -> str:
    """
    Strip unintended markdown (headers, bold, bullets) while preserving
    proper markdown table syntax so the frontend can render it.
    """
    import re
    lines = text.strip().splitlines()
    cleaned = []
    for line in lines:
        # preserve markdown table rows but strip bold/italic inside cells
        if re.match(r"^\s*\|", line):
            line = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", line)
            cleaned.append(line)
            continue
        # strip leading markdown headers (#, ##, ###)
        line = re.sub(r"^#{1,6}\s*", "", line)
        # strip bold/italic markers
        line = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", line)
        # convert bullet/dash list items to plain lines
        line = re.sub(r"^\s*[-*]\s+", "", line)
        # strip numbered list markers (1. 2. etc)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        cleaned.append(line)
    # collapse multiple blank lines into one
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


def call_llm(prompt: str, system_prompt: str = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=120.0
    )
    response.raise_for_status()
    return _clean_response(response.json()["message"]["content"])


def stream_llm(prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    with httpx.stream(
        "POST",
        f"{OLLAMA_HOST}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": True},
        timeout=120.0
    ) as response:
        for line in response.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                if not chunk.get("done"):
                    yield chunk["message"]["content"]