from __future__ import annotations

from typing import Any, Sequence

import pytest

from agent import Agent, AgentConfigurationError, AgentExecutionError, build_agent
from llm import LLMClient, LLMConfig, LLMResponse


class FakeProvider:
    def __init__(self, response: LLMResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        timeout: float,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "timeout": timeout,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def valid_config() -> LLMConfig:
    return LLMConfig(provider="openai", model="test-model", api_key="test-key", timeout=12.0)


def test_agent_graph_builds_and_compiles():
    provider = FakeProvider(LLMResponse("Mock productivity response.", "test-model"))
    agent = build_agent(LLMClient(valid_config(), provider=provider))

    assert agent.graph is not None
    assert "llm_node" in agent.graph.nodes
    assert "__start__" in agent.graph.nodes
    assert any(edge.target == "__end__" for edge in agent.graph.get_graph().edges)


def test_agent_invokes_llm_and_returns_final_response():
    provider = FakeProvider(LLMResponse("Mock productivity response.", "test-model"))
    agent = build_agent(LLMClient(valid_config(), provider=provider))

    result = agent.invoke("Help me organize my tasks.")

    assert result["final_response"] == "Mock productivity response."
    assert result["user_message"] == "Help me organize my tasks."
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "Mock productivity response."
    assert provider.calls[0]["messages"][0]["role"] == "system"


def test_agent_rejects_empty_or_invalid_input():
    provider = FakeProvider(LLMResponse("Mock productivity response.", "test-model"))
    agent = build_agent(LLMClient(valid_config(), provider=provider))

    with pytest.raises(ValueError, match="non-empty"):
        agent.invoke("   ")

    with pytest.raises(ValueError, match="non-empty"):
        agent.invoke("")


def test_agent_keeps_state_isolated_across_invocations():
    provider = FakeProvider(LLMResponse("Mock productivity response.", "test-model"))
    agent = build_agent(LLMClient(valid_config(), provider=provider))

    first = agent.invoke("Message A")
    second = agent.invoke("Message B")

    assert first["user_message"] == "Message A"
    assert second["user_message"] == "Message B"
    assert first["messages"] != second["messages"]
    assert len(provider.calls) == 2
    assert provider.calls[0]["messages"][-1]["content"] == "Message A"
    assert provider.calls[1]["messages"][-1]["content"] == "Message B"


def test_agent_propagates_llm_errors_safely():
    provider = FakeProvider(RuntimeError("LLM exploded"))
    agent = build_agent(LLMClient(valid_config(), provider=provider))

    with pytest.raises(AgentExecutionError, match="failed"):
        agent.invoke("This should fail.")


def test_agent_requires_llm_client_configuration():
    with pytest.raises(AgentConfigurationError, match="LLM client"):
        Agent(None)
