import httpx
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_HOST = os.getenv("EMBEDDING_HOST", "http://202.181.159.222:11434")
LLM_HOST = os.getenv("LLM_HOST", "http://202.181.159.222:8002")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
EMBEDDING_MODEL = "mxbai-embed-large"


def embed_text(text: str) -> list[float]:
    response = httpx.post(
        f"{EMBEDDING_HOST}/api/embed",
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


def call_llm(prompt: str, system_prompt: str = None, raw_markdown: bool = False, history: list[dict] = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    response = httpx.post(
        f"{LLM_HOST}/v1/chat/completions",
        headers=headers,
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=120.0
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if raw_markdown:
        return _fix_longitudinal_markdown(content)
    return _clean_response(content)
