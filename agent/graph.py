from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from langgraph.graph import END, START, StateGraph

from llm import LLMClient
from mcp_client import MCPClient

from .exceptions import AgentConfigurationError, AgentError, AgentExecutionError
from .nodes import (
    DEFAULT_SYSTEM_PROMPT,
    discovery_node,
    final_response_node,
    mcp_invocation_node,
    selection_node,
    validation_node,
)
from .state import AgentState, make_initial_state


class Agent:
    """MCP-aware single-turn LangGraph-based productivity agent."""

    def __init__(
        self,
        llm_client: LLMClient | None,
        mcp_client: MCPClient | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if llm_client is None:
            raise AgentConfigurationError("LLM client is required to build the agent.")
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        self.system_prompt = system_prompt
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        # 1. Discovery node
        async def _discovery_step(state: AgentState) -> AgentState:
            return await discovery_node(state, mcp_client=self.mcp_client)

        workflow.add_node("discovery_node", _discovery_step)

        # 2. LLM tool selection / decision node (named llm_node for Phase 3.2 compatibility)
        async def _llm_step(state: AgentState) -> AgentState:
            return selection_node(state, llm_client=self.llm_client, system_prompt=self.system_prompt)

        workflow.add_node("llm_node", _llm_step)

        # 3. Tool validation node
        async def _validation_step(state: AgentState) -> AgentState:
            return validation_node(state)

        workflow.add_node("validation_node", _validation_step)

        # 4. MCP invocation node
        async def _invocation_step(state: AgentState) -> AgentState:
            return await mcp_invocation_node(state, mcp_client=self.mcp_client)

        workflow.add_node("mcp_invocation_node", _invocation_step)

        # 5. Final response synthesis node
        async def _final_response_step(state: AgentState) -> AgentState:
            return final_response_node(state, llm_client=self.llm_client, system_prompt=self.system_prompt)

        workflow.add_node("final_response_node", _final_response_step)

        # Edges and conditional routing
        workflow.add_edge(START, "discovery_node")
        workflow.add_edge("discovery_node", "llm_node")

        def _route_after_selection(state: AgentState) -> str:
            if state.get("selected_tool"):
                return "validation_node"
            return "final_response_node"

        workflow.add_conditional_edges(
            "llm_node",
            _route_after_selection,
            {
                "validation_node": "validation_node",
                "final_response_node": "final_response_node",
            },
        )

        def _route_after_validation(state: AgentState) -> str:
            if state.get("tool_status") == "validated":
                return "mcp_invocation_node"
            return "final_response_node"

        workflow.add_conditional_edges(
            "validation_node",
            _route_after_validation,
            {
                "mcp_invocation_node": "mcp_invocation_node",
                "final_response_node": "final_response_node",
            },
        )

        workflow.add_edge("mcp_invocation_node", "final_response_node")
        workflow.add_edge("final_response_node", END)

        return workflow.compile()

    async def ainvoke(self, user_message: str) -> AgentState:
        """Asynchronously execute the single-turn MCP-aware agent workflow."""
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("User message must be a non-empty string.")

        initial_state = make_initial_state(user_message)
        try:
            result = await self.graph.ainvoke(initial_state)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentExecutionError("The agent failed while generating a response.") from exc

        if not isinstance(result, dict) or not result.get("final_response"):
            raise AgentExecutionError("The agent did not produce a final response.")
        return result

    def invoke(self, user_message: str) -> AgentState:
        """Synchronously execute the agent workflow with automatic event loop handling."""
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("User message must be a non-empty string.")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, self.ainvoke(user_message)).result()
        else:
            return asyncio.run(self.ainvoke(user_message))

    async def close(self) -> None:
        """Close associated MCP client connection if present."""
        if self.mcp_client is not None:
            await self.mcp_client.close()

    async def __aenter__(self) -> "Agent":
        if self.mcp_client is not None and not self.mcp_client.is_connected:
            await self.mcp_client.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()


def build_agent(
    llm_client: LLMClient | None,
    mcp_client: MCPClient | None = None,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Agent:
    return Agent(llm_client, mcp_client=mcp_client, system_prompt=system_prompt)
