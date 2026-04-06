import ollama
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

client = ollama.Client(host=OLLAMA_HOST)



def embed_text(text: str) -> list[float]:
    response = client.embed(model=EMBEDDING_MODEL, input=text)
    return response.embeddings[0]

def embed_batch(texts: list[str]) -> list[list[float]]:
    return [embed_text(t) for t in texts]


def call_llm(prompt: str, system_prompt: str = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]


def stream_llm(prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    stream = client.chat(model=LLM_MODEL, messages=messages, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]