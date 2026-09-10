from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Minimal in-memory state for the single-turn LangGraph workflow."""

    user_message: str
    messages: list[dict[str, str]]
    final_response: str
    metadata: dict[str, Any]


def make_initial_state(user_message: str) -> AgentState:
    clean_message = user_message.strip()
    if not clean_message:
        raise ValueError("User message must be a non-empty string.")
    return {
        "user_message": clean_message,
        "messages": [{"role": "user", "content": clean_message}],
        "final_response": "",
        "metadata": {},
    }
