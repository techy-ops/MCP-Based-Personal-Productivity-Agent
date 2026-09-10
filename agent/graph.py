from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from llm import LLMClient

from .exceptions import AgentConfigurationError, AgentExecutionError
from .nodes import DEFAULT_SYSTEM_PROMPT, llm_node
from .state import AgentState, make_initial_state


class Agent:
    """Minimal single-turn LangGraph-based productivity agent."""

    def __init__(self, llm_client: LLMClient | None, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        if llm_client is None:
            raise AgentConfigurationError("LLM client is required to build the agent.")
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node(
            "llm_node",
            lambda state: llm_node(state, llm_client=self.llm_client, system_prompt=self.system_prompt),
        )
        workflow.add_edge(START, "llm_node")
        workflow.add_edge("llm_node", END)
        return workflow.compile()

    def invoke(self, user_message: str) -> AgentState:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("User message must be a non-empty string.")

        initial_state = make_initial_state(user_message)
        try:
            result = self.graph.invoke(initial_state)
        except Exception as exc:
            raise AgentExecutionError("The agent failed while generating a response.") from exc

        if not isinstance(result, dict) or not result.get("final_response"):
            raise AgentExecutionError("The agent did not produce a final response.")
        return result


def build_agent(llm_client: LLMClient | None, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Agent:
    return Agent(llm_client, system_prompt=system_prompt)
