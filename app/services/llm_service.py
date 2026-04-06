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
    return response.json()["message"]["content"]


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