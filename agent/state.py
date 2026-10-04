from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Minimal in-memory state for the single-turn LangGraph workflow."""

    user_request: str
    user_message: str
    messages: list[dict[str, Any]]
    final_response: str
    metadata: dict[str, Any]
    discovered_tools: list[dict[str, Any]]
    selected_tool: str | None
    tool_arguments: dict[str, Any]
    tool_result: Any
    tool_status: str | None
    tool_error: str | None


def make_initial_state(user_request: str) -> AgentState:
    clean_message = user_request.strip() if isinstance(user_request, str) else ""
    if not clean_message:
        raise ValueError("User request must be a non-empty string.")
    return {
        "user_request": clean_message,
        "user_message": clean_message,
        "messages": [{"role": "user", "content": clean_message}],
        "final_response": "",
        "metadata": {},
        "discovered_tools": [],
        "selected_tool": None,
        "tool_arguments": {},
        "tool_result": None,
        "tool_status": None,
        "tool_error": None,
    }
