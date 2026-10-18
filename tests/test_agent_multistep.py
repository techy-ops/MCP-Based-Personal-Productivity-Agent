from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence
import pytest

from agent import (
    Agent,
    AgentConfigurationError,
    AgentExecutionError,
    AgentToolDiscoveryError,
    AgentToolValidationError,
    build_agent,
)
from app.config import BASE_DIR
from llm import LLMClient, LLMConfig, LLMResponse
from mcp_client import MCPClient
from mcp_client.exceptions import (
    MCPConnectionError,
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


def make_full_17_tool_inventory() -> list[MockMCPTool]:
    """Provide definitions for all 17 MCP tools matching the unified server contract."""
    return [
        # Tasks (6 tools)
        MockMCPTool(
            name="create_task",
            description="Create a task",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string"},
                    "priority": {"type": "string"},
                    "due_date": {"type": "string"},
                },
                "required": ["title"],
            },
        ),
        MockMCPTool(
            name="get_task",
            description="Get task by ID",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
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
            name="update_task",
            description="Update task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string"},
                    "priority": {"type": "string"},
                    "due_date": {"type": "string"},
                },
                "required": ["task_id"],
            },
        ),
        MockMCPTool(
            name="complete_task",
            description="Complete task",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        ),
        MockMCPTool(
            name="delete_task",
            description="Delete task",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        ),
        # Calendar (5 tools)
        MockMCPTool(
            name="create_event",
            description="Create an event",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["title", "start_time", "end_time"],
            },
        ),
        MockMCPTool(
            name="get_event",
            description="Get event by ID",
            inputSchema={
                "type": "object",
                "properties": {"event_id": {"type": "integer"}},
                "required": ["event_id"],
            },
        ),
        MockMCPTool(
            name="list_events",
            description="List events",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
            },
        ),
        MockMCPTool(
            name="update_event",
            description="Update event",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["event_id"],
            },
        ),
        MockMCPTool(
            name="delete_event",
            description="Delete event",
            inputSchema={
                "type": "object",
                "properties": {"event_id": {"type": "integer"}},
                "required": ["event_id"],
            },
        ),
        # Notes (6 tools)
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
            name="get_note",
            description="Get note by ID",
            inputSchema={
                "type": "object",
                "properties": {"note_id": {"type": "integer"}},
                "required": ["note_id"],
            },
        ),
        MockMCPTool(
            name="list_notes",
            description="List notes",
            inputSchema={"type": "object", "properties": {}},
        ),
        MockMCPTool(
            name="update_note",
            description="Update note",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["note_id"],
            },
        ),
        MockMCPTool(
            name="delete_note",
            description="Delete note",
            inputSchema={
                "type": "object",
                "properties": {"note_id": {"type": "integer"}},
                "required": ["note_id"],
            },
        ),
        MockMCPTool(
            name="search_notes",
            description="Search notes by query string",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]


class MockMultiStepMCPClient:
    """Mock MCPClient for testing multi-step workflows."""

    def __init__(self, tools: list[MockMCPTool] | None = None) -> None:
        self.tools = tools if tools is not None else make_full_17_tool_inventory()
        self.is_connected = False
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.call_results: dict[str, Any] = {}
        self.call_exceptions: dict[str, Exception] = {}
        self.next_call_exception: Exception | None = None

    async def connect(self) -> "MockMultiStepMCPClient":
        self.is_connected = True
        return self

    async def close(self) -> None:
        self.is_connected = False

    async def list_tools(self) -> list[MockMCPTool]:
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        args = arguments or {}
        self.tool_calls.append((name, args))

        if self.next_call_exception:
            exc = self.next_call_exception
            self.next_call_exception = None
            raise exc

        if name in self.call_exceptions:
            raise self.call_exceptions[name]

        if name in self.call_results:
            result = self.call_results[name]
            if callable(result):
                return result(args)
            return result

        # Default standard responses
        if name == "create_task":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": {"id": len(self.tool_calls), **args}}))]
            )
        elif name == "complete_task":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": {"id": args.get("task_id", 1), "status": "completed"}}))]
            )
        elif name == "list_tasks":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": [
                    {"id": 1, "title": "First Task", "status": "pending", "priority": "high"},
                    {"id": 2, "title": "Second Task", "status": "in_progress", "priority": "medium"},
                ]}))]
            )
        elif name == "get_task":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": {"id": args.get("task_id", 1), "title": "Found Task", "status": "pending"}}))]
            )
        elif name == "create_event":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": {"id": len(self.tool_calls), **args}}))]
            )
        elif name == "list_events":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": []}))]
            )
        elif name == "create_note":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": {"id": len(self.tool_calls), **args}}))]
            )
        elif name == "search_notes":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": [
                    {"id": 10, "title": f"Note for {args.get('query')}", "content": "Important spec details"}
                ]}))]
            )
        elif name == "update_note":
            return MockCallToolResult(
                content=[MockTextContent(text=json.dumps({"success": True, "data": {"id": args.get("note_id", 1), **args}}))]
            )

        return MockCallToolResult(content=[MockTextContent(text=json.dumps({"success": True, "data": args}))])


class SequenceFakeProvider:
    """Fake LLM provider returning a sequence of mock responses."""

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
        self.calls.append({"messages": list(messages), "model": model, "timeout": timeout})
        if not self.responses:
            raise RuntimeError(f"FakeProvider ran out of mock responses. Call count: {len(self.calls)}")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def make_client(responses: list[LLMResponse | Exception]) -> tuple[LLMClient, SequenceFakeProvider]:
    provider = SequenceFakeProvider(responses)
    config = LLMConfig(provider="openai", model="test-model", api_key="test-key", timeout=10.0)
    return LLMClient(config, provider=provider), provider


# ==============================================================================
# CATEGORY 1: MULTI-STEP TASK CHAINING ACROSS DOMAINS
# ==============================================================================

@pytest.mark.asyncio
async def test_multistep_task_chaining_create_and_complete():
    """Verify 2-step task workflow: create task -> complete created task -> final response."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        # Step 1: LLM proposes create_task
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Deploy Release", "priority": "high"}}), "test-model"),
        # Step 2: LLM observes create_task output and proposes complete_task
        LLMResponse(json.dumps({"tool_name": "complete_task", "arguments": {"task_id": 1}}), "test-model"),
        # Step 3: LLM observes completion and produces final response
        LLMResponse("Created task 'Deploy Release' (ID 1) and marked it complete.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create task Deploy Release and immediately complete it.")

    assert len(mcp.tool_calls) == 2
    assert mcp.tool_calls[0] == ("create_task", {"title": "Deploy Release", "priority": "high"})
    assert mcp.tool_calls[1] == ("complete_task", {"task_id": 1})

    assert result["step_count"] == 2
    assert len(result["steps"]) == 2
    assert result["steps"][0]["tool_name"] == "create_task"
    assert result["steps"][0]["tool_status"] == "success"
    assert result["steps"][1]["tool_name"] == "complete_task"
    assert result["steps"][1]["tool_status"] == "success"

    # Backward-compatible fields reflect the latest step
    assert result["selected_tool"] == "complete_task"
    assert result["tool_status"] == "success"
    assert "Created task 'Deploy Release'" in result["final_response"]


@pytest.mark.asyncio
async def test_multistep_calendar_chaining_list_and_create():
    """Verify 2-step calendar workflow: check events -> create event -> final response."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "list_events", "arguments": {"start_date": "2026-10-15T09:00:00", "end_date": "2026-10-15T17:00:00"}}), "test-model"),
        LLMResponse(json.dumps({
            "tool_name": "create_event",
            "arguments": {
                "title": "Strategy Sync",
                "start_time": "2026-10-15T14:00:00",
                "end_time": "2026-10-15T15:00:00",
            },
        }), "test-model"),
        LLMResponse("Checked your schedule (no conflicts) and booked 'Strategy Sync' at 2 PM.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Check my calendar for Oct 15 and schedule Strategy Sync at 2 PM.")

    assert len(mcp.tool_calls) == 2
    assert result["step_count"] == 2
    assert [s["tool_name"] for s in result["steps"]] == ["list_events", "create_event"]
    assert result["tool_status"] == "success"
    assert "Strategy Sync" in result["final_response"]


@pytest.mark.asyncio
async def test_multistep_notes_chaining_search_and_update():
    """Verify 2-step notes workflow: search notes -> update matching note."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "search_notes", "arguments": {"query": "roadmap"}}), "test-model"),
        LLMResponse(json.dumps({"tool_name": "update_note", "arguments": {"note_id": 10, "content": "Updated roadmap details"}}), "test-model"),
        LLMResponse("Found note 10 for roadmap and updated its content.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Find the roadmap note and update its content.")

    assert len(mcp.tool_calls) == 2
    assert result["steps"][0]["tool_name"] == "search_notes"
    assert result["steps"][1]["tool_name"] == "update_note"
    assert result["steps"][1]["tool_arguments"] == {"note_id": 10, "content": "Updated roadmap details"}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_multistep_cross_domain_3step_workflow():
    """Verify 3-step cross-domain workflow: Notes -> Tasks -> Calendar."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        # Step 1: Search notes for spec
        LLMResponse(json.dumps({"tool_name": "search_notes", "arguments": {"query": "Q4 plan"}}), "test-model"),
        # Step 2: Create task using note topic
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Execute Q4 Plan", "priority": "high"}}), "test-model"),
        # Step 3: Schedule calendar meeting for execution
        LLMResponse(json.dumps({
            "tool_name": "create_event",
            "arguments": {
                "title": "Q4 Plan Review",
                "start_time": "2026-10-18T10:00:00",
                "end_time": "2026-10-18T11:00:00",
            },
        }), "test-model"),
        # Step 4: Final response
        LLMResponse("Found Q4 plan note, created high-priority task, and scheduled review meeting for Oct 18.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Search note Q4 plan, create a high-priority task for it, and schedule review for Oct 18 at 10 AM.")

    assert len(mcp.tool_calls) == 3
    assert [c[0] for c in mcp.tool_calls] == ["search_notes", "create_task", "create_event"]
    assert result["step_count"] == 3
    assert len(result["steps"]) == 3
    assert result["tool_status"] == "success"
    assert "Found Q4 plan note" in result["final_response"]


@pytest.mark.asyncio
async def test_multistep_cross_domain_tasks_to_notes():
    """Verify 2-step cross-domain workflow: Tasks -> Notes."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "list_tasks", "arguments": {"status": "pending"}}), "test-model"),
        LLMResponse(json.dumps({"tool_name": "create_note", "arguments": {"title": "Pending Tasks Report", "content": "Tasks 1 and 2 are currently pending."}}), "test-model"),
        LLMResponse("Retrieved pending tasks and saved summary note.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("List my pending tasks and save a summary note.")

    assert len(mcp.tool_calls) == 2
    assert [c[0] for c in mcp.tool_calls] == ["list_tasks", "create_note"]
    assert result["step_count"] == 2
    assert result["tool_status"] == "success"


# ==============================================================================
# CATEGORY 2: LOOP TERMINATION & STEP LIMIT CONTROLS
# ==============================================================================

@pytest.mark.asyncio
async def test_multistep_enforces_max_steps_limit():
    """Verify agent terminates multi-step loop when max_steps is reached."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        # Step 1
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task 1"}}), "test-model"),
        # Step 2
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task 2"}}), "test-model"),
        # Step 3 (synthesis after max_steps reached)
        LLMResponse("Reached the maximum limit of 2 steps. Created Task 1 and Task 2.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp, max_steps=2)
    result = await agent.ainvoke("Create tasks in a loop.")

    assert len(mcp.tool_calls) == 2
    assert result["step_count"] == 2
    assert result["metadata"]["max_steps_reached"] is True
    assert "Reached the maximum limit" in result["final_response"]


@pytest.mark.asyncio
async def test_multistep_override_max_steps_per_invocation():
    """Verify max_steps can be overridden per individual invocation."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Single Step Task"}}), "test-model"),
        LLMResponse("Completed single step as capped by max_steps=1.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp, max_steps=5)
    result = await agent.ainvoke("Create task", max_steps=1)

    assert len(mcp.tool_calls) == 1
    assert result["step_count"] == 1
    assert result["metadata"]["max_steps_reached"] is True


# ==============================================================================
# CATEGORY 3: DYNAMIC NORMALIZATION & OBSERVATION FEEDBACK
# ==============================================================================

@pytest.mark.asyncio
async def test_multistep_normalizes_string_id_in_intermediate_step():
    """Verify intermediate step normalizes string IDs (e.g. '15') to integers before validation."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "search_notes", "arguments": {"query": "bug"}}), "test-model"),
        # Note the string note_id "15"
        LLMResponse(json.dumps({"tool_name": "get_note", "arguments": {"note_id": "15"}}), "test-model"),
        LLMResponse("Retrieved details for note 15.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Find note about bug and fetch its details.")

    assert len(mcp.tool_calls) == 2
    assert mcp.tool_calls[1] == ("get_note", {"note_id": 15})
    assert result["steps"][1]["tool_arguments"]["note_id"] == 15
    assert result["tool_status"] == "success"


# ==============================================================================
# CATEGORY 4: ERROR HANDLING & RESILIENCE IN MULTI-STEP
# ==============================================================================

@pytest.mark.asyncio
async def test_multistep_error_recovery_not_found_to_list():
    """Verify that when a tool encounters not_found, the agent can recover with an alternative step."""
    mcp = MockMultiStepMCPClient()
    mcp.call_exceptions["get_task"] = MCPToolInvocationError("Task with id 999 not found.")
    llm_client, _ = make_client([
        # Step 1: Tries to get missing task
        LLMResponse(json.dumps({"tool_name": "get_task", "arguments": {"task_id": 999}}), "test-model"),
        # Step 2: Observes not_found, falls back to list_tasks
        LLMResponse(json.dumps({"tool_name": "list_tasks", "arguments": {}}), "test-model"),
        # Step 3: Concludes
        LLMResponse("Task 999 was not found, but I found your active tasks.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Get task 999 or show my tasks.")

    assert len(mcp.tool_calls) == 2
    assert result["step_count"] == 2
    assert result["steps"][0]["tool_status"] == "not_found"
    assert result["steps"][1]["tool_status"] == "success"
    assert "Task 999 was not found" in result["final_response"]


@pytest.mark.asyncio
async def test_multistep_fails_closed_on_invalid_second_step():
    """Verify that an invalid second step fails closed and never executes on MCP."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        # Step 1: Valid task creation
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Valid Task"}}), "test-model"),
        # Step 2: Invalid task creation (invalid priority)
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Bad Task", "priority": "ultra_high"}}), "test-model"),
        # Step 3: Truthful synthesis explaining the validation error
        LLMResponse("Created 'Valid Task', but could not create 'Bad Task' because 'ultra_high' is an invalid priority.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create Valid Task then create Bad Task with ultra high priority.")

    # Only step 1 reached MCP
    assert len(mcp.tool_calls) == 1
    assert mcp.tool_calls[0][0] == "create_task"
    assert result["tool_status"] == "validation_error"
    assert "priority must be one of" in result["tool_error"]


@pytest.mark.asyncio
async def test_multistep_stops_on_connection_error():
    """Verify that a connection drop mid-workflow stops further execution safely."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        # Step 1: Valid task creation
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task One"}}), "test-model"),
        # Step 2: Note creation where MCP drops
        LLMResponse(json.dumps({"tool_name": "create_note", "arguments": {"title": "Note", "content": "Content"}}), "test-model"),
        # Synthesis explaining connection failure
        LLMResponse("Created Task One, but encountered a connection error before saving the note.", "test-model"),
    ])
    # Connection error on second tool
    mcp.call_exceptions["create_note"] = MCPConnectionError("MCP connection severed.")

    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create Task One and a note.")

    assert result["tool_status"] == "connection_error"
    assert len(result["steps"]) == 2
    assert result["steps"][0]["tool_status"] == "success"
    assert result["steps"][1]["tool_status"] == "connection_error"


# ==============================================================================
# CATEGORY 5: STATE ISOLATION & TRACE INTEGRITY
# ==============================================================================

@pytest.mark.asyncio
async def test_multistep_state_isolation_between_invocations():
    """Verify multi-step execution traces and step counts do not leak across invocations."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        # Invocation 1: 2 steps
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task A"}}), "test-model"),
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task B"}}), "test-model"),
        LLMResponse("Created Task A and Task B.", "test-model"),
        # Invocation 2: 1 step
        LLMResponse(json.dumps({"tool_name": "list_tasks", "arguments": {}}), "test-model"),
        LLMResponse("Listed tasks.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)

    res1 = await agent.ainvoke("Create Task A and Task B")
    res2 = await agent.ainvoke("List tasks")

    assert res1["step_count"] == 2
    assert len(res1["steps"]) == 2
    assert res1["steps"][0]["tool_arguments"]["title"] == "Task A"
    assert res1["steps"][1]["tool_arguments"]["title"] == "Task B"

    assert res2["step_count"] == 1
    assert len(res2["steps"]) == 1
    assert res2["steps"][0]["tool_name"] == "list_tasks"
    assert res2["steps"] != res1["steps"]


# ==============================================================================
# CATEGORY 6: LIVE MULTI-STEP MCP INTEGRATION TESTS (REAL MCP + SQLITE)
# ==============================================================================

class LiveSequenceFakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
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
        self.calls.append({"messages": list(messages), "model": model})
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse("Live multi-step default response.", model)


@pytest.fixture
def live_mcp_client(tmp_path):
    db_path = tmp_path / "live_multistep_verification.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path.as_posix()}"}
    client = MCPClient(
        server_path=BASE_DIR / "mcp_servers" / "unified_server.py",
        env=env,
    )
    return client


@pytest.mark.asyncio
async def test_live_multistep_task_lifecycle(live_mcp_client):
    """Verify live multi-step task lifecycle over real stdio MCP transport: create -> get -> complete."""
    provider = LiveSequenceFakeProvider([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Live MultiStep Task", "priority": "high"}}), "live-model"),
        LLMResponse(json.dumps({"tool_name": "get_task", "arguments": {"task_id": 1}}), "live-model"),
        LLMResponse(json.dumps({"tool_name": "complete_task", "arguments": {"task_id": 1}}), "live-model"),
        LLMResponse("Live multi-step lifecycle verified: created, verified, and completed task 1.", "live-model"),
    ])
    config = LLMConfig(provider="openai", model="live-model", api_key="test-key", timeout=15.0)
    llm_client = LLMClient(config, provider=provider)
    agent = build_agent(llm_client, mcp_client=live_mcp_client)

    try:
        result = await agent.ainvoke("Execute task lifecycle for Live MultiStep Task.")
        assert result["step_count"] == 3
        assert len(result["steps"]) == 3
        assert [s["tool_name"] for s in result["steps"]] == ["create_task", "get_task", "complete_task"]
        assert all(s["tool_status"] == "success" for s in result["steps"])
        assert result["steps"][2]["tool_result"]["data"]["status"] == "completed"
        assert "Live multi-step lifecycle verified" in result["final_response"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_live_multistep_cross_domain_notes_and_tasks(live_mcp_client):
    """Verify live cross-domain multi-step workflow over real stdio MCP transport: create note -> search note -> create task."""
    provider = LiveSequenceFakeProvider([
        LLMResponse(json.dumps({"tool_name": "create_note", "arguments": {"title": "Live Architecture Note", "content": "Implement Phase 3.5 multi-step"}}), "live-model"),
        LLMResponse(json.dumps({"tool_name": "search_notes", "arguments": {"query": "Architecture"}}), "live-model"),
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Implement Phase 3.5 multi-step", "priority": "high"}}), "live-model"),
        LLMResponse("Created architecture note, searched it, and created task 'Implement Phase 3.5 multi-step'.", "live-model"),
    ])
    config = LLMConfig(provider="openai", model="live-model", api_key="test-key", timeout=15.0)
    llm_client = LLMClient(config, provider=provider)
    agent = build_agent(llm_client, mcp_client=live_mcp_client)

    try:
        result = await agent.ainvoke("Write note Live Architecture Note, search for it, and create task from it.")
        assert result["step_count"] == 3
        assert [s["tool_name"] for s in result["steps"]] == ["create_note", "search_notes", "create_task"]
        assert all(s["tool_status"] == "success" for s in result["steps"])
        assert result["steps"][2]["tool_result"]["data"]["title"] == "Implement Phase 3.5 multi-step"
    finally:
        await agent.close()


def test_multistep_sync_invoke_wrapper():
    """Verify synchronous agent.invoke wrapper supports multi-step execution."""
    mcp = MockMultiStepMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Sync Task", "priority": "medium"}}), "test-model"),
        LLMResponse(json.dumps({"tool_name": "complete_task", "arguments": {"task_id": 1}}), "test-model"),
        LLMResponse("Synchronously created and completed task 1.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = agent.invoke("Sync create and complete task.")

    assert result["step_count"] == 2
    assert len(result["steps"]) == 2
    assert result["steps"][0]["tool_name"] == "create_task"
    assert result["steps"][1]["tool_name"] == "complete_task"
    assert "Synchronously created and completed task 1." in result["final_response"]


@pytest.mark.asyncio
async def test_live_multistep_tasks_and_calendar(live_mcp_client):
    """Verify live cross-domain Tasks + Calendar multi-step workflow over real stdio MCP transport."""
    provider = LiveSequenceFakeProvider([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Spec Review Task", "priority": "high"}}), "live-model"),
        LLMResponse(json.dumps({
            "tool_name": "create_event",
            "arguments": {
                "title": "Spec Review Session",
                "start_time": "2026-10-20T14:00:00",
                "end_time": "2026-10-20T15:00:00",
            },
        }), "live-model"),
        LLMResponse("Created spec review task and scheduled calendar session for Oct 20.", "live-model"),
    ])
    config = LLMConfig(provider="openai", model="live-model", api_key="test-key", timeout=15.0)
    llm_client = LLMClient(config, provider=provider)
    agent = build_agent(llm_client, mcp_client=live_mcp_client)

    try:
        result = await agent.ainvoke("Create a high priority task Spec Review Task and schedule review meeting on Oct 20 at 2 PM.")
        assert result["step_count"] == 2
        assert len(result["steps"]) == 2
        assert result["steps"][0]["tool_name"] == "create_task"
        assert result["steps"][1]["tool_name"] == "create_event"
        assert all(s["tool_status"] == "success" for s in result["steps"])
        assert result["steps"][0]["tool_result"]["data"]["title"] == "Spec Review Task"
        assert result["steps"][1]["tool_result"]["data"]["title"] == "Spec Review Session"
    finally:
        await agent.close()
