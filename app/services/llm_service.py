import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_HOST = os.getenv("EMBEDDING_HOST", "http://202.181.159.222:11434")
LLM_HOST = os.getenv("LLM_HOST", "http://202.181.159.222:8001")
LLM_MODEL = os.getenv("LLM_MODEL")
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
    lines = text.strip().splitlines()
    cleaned = []
    for line in lines:
        if re.match(r"^\s*\|", line):
            line = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", line)
            cleaned.append(line)
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        cleaned.append(line)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


def _fix_longitudinal_markdown(text: str) -> str:
    import re
    text = text.replace("\\n", "\n")
    text = re.sub(r'\n?(##\s)', r'\n\n\1', text)
    text = re.sub(r'\n?(-\s\d{4}-)', r'\n\n\1', text)
    text = re.sub(r'\n?(Trend:)', r'\n\n\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def call_llm(prompt: str, system_prompt: str = None, raw_markdown: bool = False) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = httpx.post(
        f"{LLM_HOST}/v1/chat/completions",
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=120.0
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if raw_markdown:
        return _fix_longitudinal_markdown(content)
    return _clean_response(content)


def stream_llm(prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    with httpx.stream(
        "POST",
        f"{LLM_HOST}/v1/chat/completions",
        json={"model": LLM_MODEL, "messages": messages, "stream": True},
        timeout=120.0
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta