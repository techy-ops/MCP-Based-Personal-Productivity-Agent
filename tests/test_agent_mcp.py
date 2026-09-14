from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence
from unittest.mock import AsyncMock

import pytest

from agent import (
    Agent,
    AgentConfigurationError,
    AgentExecutionError,
    AgentToolDiscoveryError,
    build_agent,
)
from agent.validation import validate_tool_call
from llm import LLMClient, LLMConfig, LLMResponse
from mcp_client.exceptions import (
    MCPConnectionError,
    MCPToolDiscoveryError,
    MCPToolInvocationError,
)


@dataclass
class MockMCPTool:
    name: str
    description: str = ""
    inputSchema: dict[str, Any] | None = None


@dataclass
class MockTextContent:
    type: str = "text"
    text: str = ""


@dataclass
class MockCallToolResult:
    content: list[MockTextContent]
    isError: bool = False
    structuredContent: dict[str, Any] | None = None


class MockMCPClient:
    """Deterministic mock of the MCPClient for testing agent integration."""

    def __init__(self, tools: list[MockMCPTool] | None = None) -> None:
        self.tools = tools if tools is not None else [
            MockMCPTool(
                name="create_task",
                description="Create a task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "priority": {"type": "string"},
                    },
                    "required": ["title"],
                },
            ),
            MockMCPTool(
                name="list_tasks",
                description="List tasks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "priority": {"type": "string"},
                    },
                },
            ),
            MockMCPTool(
                name="create_event",
                description="Create an event",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                    },
                    "required": ["title", "start_time", "end_time"],
                },
            ),
            MockMCPTool(
                name="create_note",
                description="Create a note",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["title", "content"],
                },
            ),
            MockMCPTool(
                name="search_notes",
                description="Search notes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            ),
        ]
        self.is_connected = False
        self.connect_called = 0
        self.close_called = 0
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.next_call_result: Any = None
        self.next_call_exception: Exception | None = None
        self.discovery_exception: Exception | None = None

    async def connect(self) -> "MockMCPClient":
        self.connect_called += 1
        self.is_connected = True
        return self

    async def close(self) -> None:
        self.close_called += 1
        self.is_connected = False

    async def list_tools(self) -> list[MockMCPTool]:
        if self.discovery_exception:
            raise self.discovery_exception
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.tool_calls.append((name, arguments or {}))
        if self.next_call_exception:
            raise self.next_call_exception
        if self.next_call_result is not None:
            return self.next_call_result
        return MockCallToolResult(
            content=[MockTextContent(text=json.dumps({"success": True, "data": {"id": 1, **(arguments or {})}}))]
        )


class SequenceFakeProvider:
    """Fake LLM provider returning a sequence of canned responses."""

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self.responses = list(responses)
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
        self.calls.append({
            "messages": list(messages),
            "model": model,
            "timeout": timeout,
        })
        if not self.responses:
            return LLMResponse("Default fallback response.", model)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def make_client(responses: list[LLMResponse | Exception]) -> tuple[LLMClient, SequenceFakeProvider]:
    provider = SequenceFakeProvider(responses)
    config = LLMConfig(provider="openai", model="test-model", api_key="test-key", timeout=10.0)
    return LLMClient(config, provider=provider), provider


# ==============================================================================
# TEST CATEGORY A — TOOL DISCOVERY
# ==============================================================================

@pytest.mark.asyncio
async def test_tool_discovery_populates_state_metadata():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse("Direct answer.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Hello")

    assert len(result["discovered_tools"]) == 5
    names = [t["name"] for t in result["discovered_tools"]]
    assert "create_task" in names
    assert "create_event" in names
    assert "create_note" in names
    assert "list_tasks" in names
    assert "search_notes" in names


@pytest.mark.asyncio
async def test_tool_discovery_preserves_descriptions_and_schemas():
    mcp = MockMCPClient()
    llm_client, _ = make_client([LLMResponse("Hi", "test-model")])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Hello")

    task_tool = next(t for t in result["discovered_tools"] if t["name"] == "create_task")
    assert task_tool["description"] == "Create a task"
    assert "properties" in task_tool["input_schema"]
    assert "required" in task_tool["input_schema"]
    assert "title" in task_tool["input_schema"]["required"]


@pytest.mark.asyncio
async def test_tool_discovery_does_not_depend_on_hardcoded_17_tools():
    custom_tools = [
        MockMCPTool(name="custom_tool_alpha", description="Alpha", inputSchema={"type": "object"}),
        MockMCPTool(name="custom_tool_beta", description="Beta", inputSchema={"type": "object"}),
    ]
    mcp = MockMCPClient(tools=custom_tools)
    llm_client, _ = make_client([LLMResponse("Hi", "test-model")])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Hello")

    assert len(result["discovered_tools"]) == 2
    assert {t["name"] for t in result["discovered_tools"]} == {"custom_tool_alpha", "custom_tool_beta"}


@pytest.mark.asyncio
async def test_tool_discovery_failure_handled_cleanly():
    mcp = MockMCPClient()
    mcp.discovery_exception = MCPToolDiscoveryError("Server crashed during discovery")
    llm_client, _ = make_client([])
    agent = build_agent(llm_client, mcp_client=mcp)

    with pytest.raises(AgentToolDiscoveryError, match="Failed to discover tools"):
        await agent.ainvoke("Do something")


# ==============================================================================
# TEST CATEGORY B — TOOL SELECTION
# ==============================================================================

@pytest.mark.asyncio
async def test_llm_selects_create_task():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Buy groceries", "priority": "high"}}), "test-model"),
        LLMResponse("I have created the task 'Buy groceries' for you.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create a high priority task to buy groceries.")

    assert result["selected_tool"] == "create_task"
    assert result["tool_arguments"] == {"title": "Buy groceries", "priority": "high"}
    assert result["tool_status"] == "success"
    assert len(mcp.tool_calls) == 1
    assert mcp.tool_calls[0] == ("create_task", {"title": "Buy groceries", "priority": "high"})
    assert "Buy groceries" in result["final_response"]


@pytest.mark.asyncio
async def test_llm_selects_create_event():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({
            "tool_name": "create_event",
            "arguments": {
                "title": "Dentist",
                "start_time": "2026-09-20T10:00:00",
                "end_time": "2026-09-20T11:00:00",
            },
        }), "test-model"),
        LLMResponse("Scheduled Dentist appointment.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Schedule Dentist on Sep 20 from 10 to 11.")

    assert result["selected_tool"] == "create_event"
    assert result["tool_status"] == "success"
    assert len(mcp.tool_calls) == 1
    assert mcp.tool_calls[0][0] == "create_event"


@pytest.mark.asyncio
async def test_llm_selects_create_note():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({
            "tool_name": "create_note",
            "arguments": {"title": "Ideas", "content": "Brainstorming topic"},
        }), "test-model"),
        LLMResponse("Note saved.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Take a note titled Ideas with brainstorming topic.")

    assert result["selected_tool"] == "create_note"
    assert result["tool_status"] == "success"
    assert mcp.tool_calls[0][0] == "create_note"


@pytest.mark.asyncio
async def test_llm_selects_list_tasks():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "list_tasks", "arguments": {"status": "pending"}}), "test-model"),
        LLMResponse("You have 1 pending task.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Show my pending tasks.")

    assert result["selected_tool"] == "list_tasks"
    assert result["tool_arguments"] == {"status": "pending"}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_llm_selects_search_notes():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "search_notes", "arguments": {"query": "project plan"}}), "test-model"),
        LLMResponse("Found 2 notes matching project plan.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Search for notes mentioning project plan.")

    assert result["selected_tool"] == "search_notes"
    assert result["tool_arguments"] == {"query": "project plan"}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_llm_chooses_no_tool_when_appropriate():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse("Hello! How can I help you today?", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Hello")

    assert result["selected_tool"] is None
    assert result["tool_status"] == "no_tool"
    assert len(mcp.tool_calls) == 0
    assert result["final_response"] == "Hello! How can I help you today?"


@pytest.mark.asyncio
async def test_llm_selects_unknown_tool_is_rejected():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "delete_everything", "arguments": {}}), "test-model"),
        LLMResponse("I cannot perform this operation because the requested tool is unavailable.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Delete everything now!")

    assert result["selected_tool"] == "delete_everything"
    assert result["tool_status"] == "validation_error"
    assert "not among the available discovered tools" in result["tool_error"]
    assert len(mcp.tool_calls) == 0


# ==============================================================================
# TEST CATEGORY C — ARGUMENT VALIDATION
# ==============================================================================

def test_validation_layer_directly():
    discovered = [
        {
            "name": "create_task",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string"},
                    "due_days": {"type": "integer"},
                },
                "required": ["title"],
            },
        }
    ]

    # Valid
    valid, err = validate_tool_call("create_task", {"title": "Task 1", "priority": "low"}, discovered)
    assert valid is True
    assert err is None

    # Missing required argument
    valid, err = validate_tool_call("create_task", {"priority": "low"}, discovered)
    assert valid is False
    assert "Missing required argument 'title'" in err

    # Empty required string argument
    valid, err = validate_tool_call("create_task", {"title": "   "}, discovered)
    assert valid is False
    assert "cannot be empty" in err

    # Unsupported argument
    valid, err = validate_tool_call("create_task", {"title": "Task", "unsupported_param": 123}, discovered)
    assert valid is False
    assert "Unsupported argument 'unsupported_param'" in err

    # Invalid argument type (boolean for integer)
    valid, err = validate_tool_call("create_task", {"title": "Task", "due_days": True}, discovered)
    assert valid is False
    assert "expected integer, received boolean" in err

    # Invalid arguments container (non-dict)
    valid, err = validate_tool_call("create_task", ["title"], discovered)
    assert valid is False
    assert "must be provided as a dictionary" in err

    # Unknown tool
    valid, err = validate_tool_call("nonexistent", {}, discovered)
    assert valid is False
    assert "not among the available discovered tools" in err


@pytest.mark.asyncio
async def test_invalid_arguments_never_reach_mcp():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        # Missing required 'title'
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"priority": "high"}}), "test-model"),
        LLMResponse("I couldn't create the task because a title is required.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create a high priority task without a name.")

    assert result["tool_status"] == "validation_error"
    assert "Missing required argument" in result["tool_error"]
    assert len(mcp.tool_calls) == 0


# ==============================================================================
# TEST CATEGORY D & E — MCP INVOCATION & RESULT HANDLING
# ==============================================================================

@pytest.mark.asyncio
async def test_mcp_invocation_not_found_handling():
    mcp = MockMCPClient()
    mcp.next_call_exception = MCPToolInvocationError("Task with id 999 not found.")
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Existing"}}), "test-model"),
        LLMResponse("Task 999 was not found.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Find task 999")

    assert result["tool_status"] == "not_found"
    assert "not found" in result["tool_error"]
    assert len(mcp.tool_calls) == 1


@pytest.mark.asyncio
async def test_mcp_invocation_server_error_handling():
    mcp = MockMCPClient()
    mcp.next_call_exception = MCPToolInvocationError("Internal database lock failure.")
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task"}}), "test-model"),
        LLMResponse("I encountered an internal error while creating the task.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create task")

    assert result["tool_status"] == "server_error"
    assert "database lock" in result["tool_error"]


@pytest.mark.asyncio
async def test_mcp_invocation_connection_error_handling():
    mcp = MockMCPClient()
    mcp.next_call_exception = MCPConnectionError("MCP connection dropped.")
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task"}}), "test-model"),
        LLMResponse("Could not connect to the backend.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create task")

    assert result["tool_status"] == "connection_error"


# ==============================================================================
# TEST CATEGORY F — GRAPH & LIFECYCLE TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_agent_sync_invoke_wrapper_matches_ainvoke():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse("Sync answer.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    # Testing sync invoke()
    result = agent.invoke("Hello sync")
    assert result["final_response"] == "Sync answer."
    assert result["user_message"] == "Hello sync"


@pytest.mark.asyncio
async def test_agent_repeated_invocation_state_isolation():
    mcp = MockMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task One"}}), "test-model"),
        LLMResponse("Created Task One.", "test-model"),
        LLMResponse(json.dumps({"tool_name": "create_note", "arguments": {"title": "Note Two", "content": "Content"}}), "test-model"),
        LLMResponse("Created Note Two.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)

    res1 = await agent.ainvoke("First request")
    res2 = await agent.ainvoke("Second request")

    assert res1["selected_tool"] == "create_task"
    assert res2["selected_tool"] == "create_note"
    assert res1["user_message"] == "First request"
    assert res2["user_message"] == "Second request"
    assert len(mcp.tool_calls) == 2
