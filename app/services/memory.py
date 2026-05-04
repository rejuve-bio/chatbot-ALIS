import hashlib
from collections import deque
from typing import Optional

# user_key -> deque of (question, answer) tuples, max 3 entries
_store: dict[str, deque] = {}
MAX_TURNS = 3


def _user_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def get_history(token: str) -> list[dict]:
    """Return conversation history as a flat list of {role, content} dicts."""
    key = _user_key(token)
    turns = _store.get(key, deque())
    messages = []
    for q, a in turns:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages


def save_turn(token: str, question: str, answer: str) -> None:
    key = _user_key(token)
    if key not in _store:
        _store[key] = deque(maxlen=MAX_TURNS)
    _store[key].append((question, answer))


def clear_history(token: str) -> None:
    key = _user_key(token)
    _store.pop(key, None)
