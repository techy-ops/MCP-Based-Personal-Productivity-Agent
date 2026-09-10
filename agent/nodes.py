from __future__ import annotations

from llm import LLMClient

from .exceptions import AgentConfigurationError, AgentExecutionError
from .state import AgentState

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful personal productivity assistant. Be concise, practical, and do not claim "
    "actions were performed unless they were."
)


def llm_node(state: AgentState, *, llm_client: LLMClient, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> AgentState:
    if llm_client is None:
        raise AgentConfigurationError("LLM client is required for the agent workflow.")

    user_message = state.get("user_message", "").strip()
    if not user_message:
        raise ValueError("User message must be a non-empty string.")

    try:
        response = llm_client.generate(user_message, system_message=system_prompt)
    except Exception as exc:  # pragma: no cover - exercised by agent-level tests
        raise AgentExecutionError("The agent LLM node failed while generating a response.") from exc

    conversation = list(state.get("messages", []))
    conversation.append({"role": "assistant", "content": response.content})

    metadata = dict(state.get("metadata", {}))
    metadata.update({"model": response.model, "provider": llm_client.config.provider})

    return {
        "user_message": user_message,
        "messages": conversation,
        "final_response": response.content,
        "metadata": metadata,
    }
