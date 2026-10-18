from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """In-memory state for the LangGraph workflow with context and safety guardrails support."""

    user_request: str
    user_message: str
    messages: list[dict[str, Any]]
    conversation_history: list[dict[str, Any]]
    context_entities: dict[str, list[dict[str, Any]]]
    final_response: str
    metadata: dict[str, Any]
    discovered_tools: list[dict[str, Any]]
    selected_tool: str | None
    tool_arguments: dict[str, Any]
    tool_result: Any
    tool_status: str | None
    tool_error: str | None
    steps: list[dict[str, Any]]
    intermediate_steps: list[dict[str, Any]]
    step_count: int
    max_steps: int
    confirmed: bool


def make_initial_state(
    user_request: str,
    *,
    max_steps: int = 5,
    conversation_history: list[dict[str, Any]] | None = None,
    confirmed: bool = False,
) -> AgentState:
    clean_message = user_request.strip() if isinstance(user_request, str) else ""
    if not clean_message:
        raise ValueError("User request must be a non-empty string.")

    # Context management: enforce safe context limits on conversation history
    history = list(conversation_history or [])
    if len(history) > 10:
        history = history[-10:]

    messages = list(history) + [{"role": "user", "content": clean_message}]
    meta: dict[str, Any] = {}
    if confirmed:
        meta["confirmed"] = True

    return {
        "user_request": clean_message,
        "user_message": clean_message,
        "messages": messages,
        "conversation_history": history,
        "context_entities": {"tasks": [], "events": [], "notes": []},
        "final_response": "",
        "metadata": meta,
        "discovered_tools": [],
        "selected_tool": None,
        "tool_arguments": {},
        "tool_result": None,
        "tool_status": None,
        "tool_error": None,
        "steps": [],
        "intermediate_steps": [],
        "step_count": 0,
        "max_steps": max_steps,
        "confirmed": confirmed,
    }
