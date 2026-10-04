from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Sequence
import pytest

from agent import (
    Agent,
    AgentConfigurationError,
    AgentExecutionError,
    AgentToolDiscoveryError,
    build_agent,
)
from agent.validation import validate_tool_call
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
            description="Create calendar event",
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
            description="Get calendar event",
            inputSchema={
                "type": "object",
                "properties": {"event_id": {"type": "integer"}},
                "required": ["event_id"],
            },
        ),
        MockMCPTool(
            name="list_events",
            description="List calendar events",
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
            description="Update calendar event",
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
            description="Delete calendar event",
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
            description="List all notes",
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
            description="Search notes by query",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]


class MockNLPMCPClient:
    """Mock MCP client recording invocations and returning structured data."""

    def __init__(self, tools: list[MockMCPTool] | None = None) -> None:
        self.tools = tools if tools is not None else make_full_17_tool_inventory()
        self.is_connected = False
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.next_call_result: Any = None
        self.next_call_exception: Exception | None = None

    async def connect(self) -> "MockNLPMCPClient":
        self.is_connected = True
        return self

    async def close(self) -> None:
        self.is_connected = False

    async def list_tools(self) -> list[MockMCPTool]:
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


class FakeProvider:
    """Deterministic LLM provider returning programmed responses."""

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
            return LLMResponse("Default fallback response.", model)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def make_client(responses: list[LLMResponse | Exception]) -> tuple[LLMClient, FakeProvider]:
    provider = FakeProvider(responses)
    config = LLMConfig(provider="openai", model="test-model", api_key="test-key", timeout=10.0)
    return LLMClient(config, provider=provider), provider


# ==============================================================================
# CATEGORY A & B — NATURAL LANGUAGE INTENT & ARGUMENT EXTRACTION
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_intent_create_task_tomorrow():
    """Natural-language request to create a task tomorrow selects create_task and extracts args."""
    mcp = MockNLPMCPClient()
    llm_client, provider = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "create_task",
                "arguments": {
                    "title": "Finish my project report",
                    "due_date": "2026-10-05T18:00:00",
                },
            }),
            "test-model",
        ),
        LLMResponse("Done — I created the task 'Finish my project report' due tomorrow.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke(user_request="Create a task to finish my project report tomorrow.")

    assert result["selected_tool"] == "create_task"
    assert result["tool_arguments"]["title"] == "Finish my project report"
    assert result["tool_arguments"]["due_date"] == "2026-10-05T18:00:00"
    assert result["tool_status"] == "success"
    assert "Finish my project report" in result["final_response"]
    assert len(mcp.tool_calls) == 1
    # Check that system prompt in selection step included reference context
    selection_call = provider.calls[0]
    assert any("Reference Context" in m["content"] for m in selection_call["messages"] if m["role"] == "system")


@pytest.mark.asyncio
async def test_nl_intent_schedule_meeting():
    """Natural-language meeting request selects create_event with start and end times."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "create_event",
                "arguments": {
                    "title": "Meeting with Rahul",
                    "start_time": "2026-10-05T15:00:00",
                    "end_time": "2026-10-05T16:00:00",
                },
            }),
            "test-model",
        ),
        LLMResponse("Scheduled 'Meeting with Rahul' tomorrow from 3:00 PM to 4:00 PM.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Schedule a meeting with Rahul tomorrow at 3 PM.")

    assert result["selected_tool"] == "create_event"
    assert result["tool_arguments"]["title"] == "Meeting with Rahul"
    assert result["tool_arguments"]["start_time"] == "2026-10-05T15:00:00"
    assert result["tool_arguments"]["end_time"] == "2026-10-05T16:00:00"
    assert result["tool_status"] == "success"
    assert "Scheduled 'Meeting with Rahul'" in result["final_response"]


@pytest.mark.asyncio
async def test_nl_coerces_string_ids_to_integers():
    """String task IDs from natural language are coerced to integers for tool arguments."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "complete_task",
                "arguments": {"task_id": "12"},
            }),
            "test-model",
        ),
        LLMResponse("Marked task 12 as complete.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Mark task 12 as complete.")

    assert result["selected_tool"] == "complete_task"
    assert result["tool_arguments"] == {"task_id": 12}
    assert isinstance(result["tool_arguments"]["task_id"], int)
    assert result["tool_status"] == "success"
    assert mcp.tool_calls[0] == ("complete_task", {"task_id": 12})


# ==============================================================================
# CATEGORY C — TASKS CRUD OPERATIONS
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_task_crud_create():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "create_task",
                "arguments": {"title": "Review PR #2", "priority": "high"},
            }),
            "test-model",
        ),
        LLMResponse("Created high priority task 'Review PR #2'.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create a high priority task to review PR #2.")

    assert result["selected_tool"] == "create_task"
    assert result["tool_arguments"]["priority"] == "high"
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_task_crud_list():
    mcp = MockNLPMCPClient()
    mcp.next_call_result = MockCallToolResult(
        content=[MockTextContent(text=json.dumps({
            "success": True,
            "data": [
                {"id": 1, "title": "Report", "status": "pending", "priority": "high"},
                {"id": 2, "title": "Tests", "status": "pending", "priority": "medium"},
            ],
        }))]
    )
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "list_tasks", "arguments": {"status": "pending"}}),
            "test-model",
        ),
        LLMResponse("You have 2 pending tasks: 1. Report (high priority), 2. Tests (medium priority).", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Show my pending tasks.")

    assert result["selected_tool"] == "list_tasks"
    assert result["tool_arguments"] == {"status": "pending"}
    assert result["tool_status"] == "success"
    assert "pending tasks" in result["final_response"]


@pytest.mark.asyncio
async def test_nl_task_crud_update():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "update_task",
                "arguments": {"task_id": 5, "priority": "high"},
            }),
            "test-model",
        ),
        LLMResponse("Updated task 5 priority to high.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Change priority of task 5 to high.")

    assert result["selected_tool"] == "update_task"
    assert result["tool_arguments"] == {"task_id": 5, "priority": "high"}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_task_crud_delete():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "delete_task", "arguments": {"task_id": 12}}),
            "test-model",
        ),
        LLMResponse("Task 12 has been successfully deleted.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Delete task 12.")

    assert result["selected_tool"] == "delete_task"
    assert result["tool_arguments"] == {"task_id": 12}
    assert result["tool_status"] == "success"
    assert "deleted" in result["final_response"]


# ==============================================================================
# CATEGORY D — CALENDAR CRUD OPERATIONS
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_calendar_crud_list_events_for_friday():
    mcp = MockNLPMCPClient()
    mcp.next_call_result = MockCallToolResult(
        content=[MockTextContent(text=json.dumps({
            "success": True,
            "data": [
                {
                    "id": 10,
                    "title": "Weekly Sync",
                    "start_time": "2026-10-09T10:00:00",
                    "end_time": "2026-10-09T11:00:00",
                    "location": "Room B",
                }
            ],
        }))]
    )
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "list_events",
                "arguments": {
                    "start_date": "2026-10-09T00:00:00",
                    "end_date": "2026-10-09T23:59:59",
                },
            }),
            "test-model",
        ),
        LLMResponse("On Friday, you have 1 event: Weekly Sync from 10:00 AM to 11:00 AM in Room B.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Show my calendar events for Friday.")

    assert result["selected_tool"] == "list_events"
    assert result["tool_status"] == "success"
    assert "Weekly Sync" in result["final_response"]


@pytest.mark.asyncio
async def test_nl_calendar_crud_update_event():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "update_event",
                "arguments": {
                    "event_id": 8,
                    "start_time": "2026-10-05T16:00:00",
                    "end_time": "2026-10-05T17:00:00",
                },
            }),
            "test-model",
        ),
        LLMResponse("Rescheduled event 8 to 4:00 PM.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Move event 8 to 4 PM tomorrow.")

    assert result["selected_tool"] == "update_event"
    assert result["tool_arguments"]["event_id"] == 8
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_calendar_crud_delete_event():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "delete_event", "arguments": {"event_id": 8}}),
            "test-model",
        ),
        LLMResponse("Event 8 has been cancelled and removed.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Cancel event 8.")

    assert result["selected_tool"] == "delete_event"
    assert result["tool_arguments"] == {"event_id": 8}
    assert result["tool_status"] == "success"


# ==============================================================================
# CATEGORY E — NOTES CRUD & SEARCH
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_notes_crud_create():
    mcp = MockNLPMCPClient()
    llp_args = {"title": "Meeting Notes", "content": "Architecture discussion notes"}
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_note", "arguments": llp_args}), "test-model"),
        LLMResponse("Created note 'Meeting Notes'.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create a note about today's meeting with architecture discussion notes.")

    assert result["selected_tool"] == "create_note"
    assert result["tool_arguments"] == llp_args
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_notes_search():
    mcp = MockNLPMCPClient()
    mcp.next_call_result = MockCallToolResult(
        content=[MockTextContent(text=json.dumps({
            "success": True,
            "data": [{"id": 3, "title": "System Architecture", "content": "Diagrams and specs"}],
        }))]
    )
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "search_notes", "arguments": {"query": "project architecture"}}),
            "test-model",
        ),
        LLMResponse("Found 1 note matching 'project architecture': System Architecture.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Search my notes for project architecture.")

    assert result["selected_tool"] == "search_notes"
    assert result["tool_arguments"] == {"query": "project architecture"}
    assert result["tool_status"] == "success"
    assert "System Architecture" in result["final_response"]


@pytest.mark.asyncio
async def test_nl_notes_delete():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "delete_note", "arguments": {"note_id": 3}}), "test-model"),
        LLMResponse("Note 3 has been deleted.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Delete note 3.")

    assert result["selected_tool"] == "delete_note"
    assert result["tool_arguments"] == {"note_id": 3}
    assert result["tool_status"] == "success"


# ==============================================================================
# CATEGORY F & G — INVALID ARGUMENTS & MALFORMED OUTPUT HANDLING
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_missing_required_argument_rejected_before_mcp():
    """Missing required start_time for create_event fails validation before reaching MCP."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "create_event", "arguments": {"title": "Incomplete Event"}}),
            "test-model",
        ),
        LLMResponse("Could not schedule event because start time and end time are required.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Schedule Incomplete Event.")

    assert result["tool_status"] == "validation_error"
    assert "start_time" in result["tool_error"]
    assert len(mcp.tool_calls) == 0


@pytest.mark.asyncio
async def test_nl_invalid_date_format_rejected():
    """Invalid date/time string fails validation before reaching MCP."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "create_event",
                "arguments": {
                    "title": "Bad Date Event",
                    "start_time": "invalid-datetime-format",
                    "end_time": "2026-10-05T12:00:00",
                },
            }),
            "test-model",
        ),
        LLMResponse("Invalid date format provided.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Schedule event at invalid date.")

    assert result["tool_status"] == "validation_error"
    assert "ISO 8601" in result["tool_error"]
    assert len(mcp.tool_calls) == 0


@pytest.mark.asyncio
async def test_nl_end_time_before_start_time_rejected():
    """End time chronologically before start time fails validation before reaching MCP."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "create_event",
                "arguments": {
                    "title": "Reverse Time",
                    "start_time": "2026-10-05T15:00:00",
                    "end_time": "2026-10-05T14:00:00",
                },
            }),
            "test-model",
        ),
        LLMResponse("End time must be after start time.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Schedule Reverse Time from 3 to 2.")

    assert result["tool_status"] == "validation_error"
    assert "must be after 'start_time'" in result["tool_error"]
    assert len(mcp.tool_calls) == 0


@pytest.mark.asyncio
async def test_nl_non_positive_id_rejected():
    """Negative or zero ID values fail validation before reaching MCP."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "complete_task", "arguments": {"task_id": -5}}),
            "test-model",
        ),
        LLMResponse("Task ID must be a positive integer.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Complete task -5.")

    assert result["tool_status"] == "validation_error"
    assert "positive integer" in result["tool_error"]
    assert len(mcp.tool_calls) == 0


@pytest.mark.asyncio
async def test_nl_malformed_llm_output_handled_gracefully():
    """When the LLM produces malformed or non-JSON output, agent treats as conversational without crashing."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse("I am not sure how to perform that action yet.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Do some undefined magic.")

    assert result["selected_tool"] is None
    assert result["tool_status"] == "no_tool"
    assert result["final_response"] == "I am not sure how to perform that action yet."
    assert len(mcp.tool_calls) == 0


# ==============================================================================
# CATEGORY H & I — RESOURCE NOT FOUND & MCP FAILURES
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_resource_not_found_propagated():
    """Resource not found error from MCP is safely handled and truthfully explained."""
    mcp = MockNLPMCPClient()
    mcp.next_call_exception = MCPToolInvocationError("Task with id 999 was not found.")
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "complete_task", "arguments": {"task_id": 999}}), "test-model"),
        LLMResponse("Task 999 was not found in your task list.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Mark task 999 as complete.")

    assert result["tool_status"] == "not_found"
    assert "not found" in result["tool_error"].lower()
    assert "Task 999 was not found" in result["final_response"]


@pytest.mark.asyncio
async def test_nl_calendar_conflict_propagated():
    """Calendar conflict error is categorized as conflict_error and truthfully explained."""
    mcp = MockNLPMCPClient()
    mcp.next_call_exception = MCPToolInvocationError("Calendar conflict: another event already exists during this time.")
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "create_event",
                "arguments": {
                    "title": "Double Booked",
                    "start_time": "2026-10-05T10:00:00",
                    "end_time": "2026-10-05T11:00:00",
                },
            }),
            "test-model",
        ),
        LLMResponse("Could not schedule 'Double Booked' due to a scheduling conflict.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Schedule Double Booked tomorrow at 10 AM.")

    assert result["tool_status"] == "conflict_error"
    assert "conflict" in result["tool_error"].lower()
    assert "conflict" in result["final_response"].lower()


@pytest.mark.asyncio
async def test_nl_mcp_connection_failure():
    """MCP connection error propagates cleanly and produces a helpful error response."""
    mcp = MockNLPMCPClient()
    mcp.next_call_exception = MCPConnectionError("Unified MCP server connection lost.")
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "list_notes", "arguments": {}}), "test-model"),
        LLMResponse("Unable to retrieve notes right now due to a connection error with the backend.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Show all my notes.")

    assert result["tool_status"] == "connection_error"
    assert "connection" in result["final_response"].lower()


# ==============================================================================
# CATEGORY J & K — LLM FAILURES & RESPONSE GENERATION
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_llm_failure_during_selection_raises_agent_error():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([RuntimeError("OpenAI API network timeout")])
    agent = build_agent(llm_client, mcp_client=mcp)

    with pytest.raises(AgentExecutionError, match="tool-selection"):
        await agent.ainvoke("Create a task")


@pytest.mark.asyncio
async def test_nl_llm_failure_during_final_response_raises_agent_error():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task 1"}}), "test-model"),
        RuntimeError("OpenAI API rate limit during synthesis"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)

    with pytest.raises(AgentExecutionError, match="final response"):
        await agent.ainvoke("Create a task titled Task 1")


@pytest.mark.asyncio
async def test_nl_conversational_response_for_greetings():
    """Greetings and non-tool queries are handled conversationally without tool invocation."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse("Good morning! How can I assist you with your tasks or calendar today?", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Good morning!")

    assert result["selected_tool"] is None
    assert result["tool_status"] == "no_tool"
    assert len(mcp.tool_calls) == 0
    assert "Good morning" in result["final_response"]


# ==============================================================================
# CATEGORY L — STATE ISOLATION BETWEEN SEPARATE REQUESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_nl_state_isolation_between_requests():
    """Separate natural language invocations maintain isolated state and messages."""
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "create_task", "arguments": {"title": "Task A"}}), "test-model"),
        LLMResponse("Created Task A.", "test-model"),
        LLMResponse(json.dumps({"tool_name": "create_note", "arguments": {"title": "Note B", "content": "Text"}}), "test-model"),
        LLMResponse("Created Note B.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)

    res1 = await agent.ainvoke("Create task Task A")
    res2 = await agent.ainvoke("Create note Note B with Text")

    assert res1["user_request"] == "Create task Task A"
    assert res2["user_request"] == "Create note Note B with Text"
    assert res1["selected_tool"] == "create_task"
    assert res2["selected_tool"] == "create_note"
    assert len(res1["messages"]) == 2
    assert len(res2["messages"]) == 2
    assert res1["messages"][0]["content"] == "Create task Task A"
    assert res2["messages"][0]["content"] == "Create note Note B with Text"


# ==============================================================================
# CATEGORY M — LIVE REPRESENTATIVE MCP INVOCATION (NATURAL LANGUAGE)
# ==============================================================================

@pytest.fixture
def isolated_live_nl_client(tmp_path):
    db_path = tmp_path / "live_nl_verification.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path.as_posix()}"}
    client = MCPClient(
        server_path=BASE_DIR / "mcp_servers" / "unified_server.py",
        env=env,
    )
    return client


@pytest.mark.asyncio
async def test_live_nl_task_creation_and_completion(isolated_live_nl_client):
    """Verify end-to-end natural language task creation and completion with real unified MCP server."""
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({
                "tool_name": "create_task",
                "arguments": {
                    "title": "Natural Language Live Task",
                    "priority": "high",
                },
            }),
            "test-model",
        ),
        LLMResponse("Created task 'Natural Language Live Task' with high priority.", "test-model"),
        LLMResponse(
            json.dumps({
                "tool_name": "complete_task",
                "arguments": {"task_id": "1"},
            }),
            "test-model",
        ),
        LLMResponse("Task 1 has been completed.", "test-model"),
    ])

    agent = build_agent(llm_client, mcp_client=isolated_live_nl_client)
    try:
        # Create
        res_create = await agent.ainvoke("Create a high priority task titled Natural Language Live Task")
        assert res_create["selected_tool"] == "create_task"
        assert res_create["tool_status"] == "success"
        assert res_create["tool_result"]["success"] is True
        task_id = res_create["tool_result"]["data"]["id"]
        assert task_id == 1

        # Complete
        res_complete = await agent.ainvoke("Mark task 1 as complete")
        assert res_complete["selected_tool"] == "complete_task"
        assert res_complete["tool_status"] == "success"
        assert res_complete["tool_result"]["success"] is True
        assert res_complete["tool_result"]["data"]["status"] == "completed"
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_live_nl_calendar_and_notes_workflows(isolated_live_nl_client):
    """Verify live calendar scheduling and notes creation/retrieval via natural language."""
    llm_client, _ = make_client([
        # 1. Create calendar event
        LLMResponse(
            json.dumps({
                "tool_name": "create_event",
                "arguments": {
                    "title": "Architecture Sprint Review",
                    "start_time": "2026-10-15T14:00:00",
                    "end_time": "2026-10-15T15:00:00",
                    "location": "Room 401",
                },
            }),
            "test-model",
        ),
        LLMResponse("Scheduled 'Architecture Sprint Review' on Oct 15 at 2:00 PM.", "test-model"),
        # 2. Create note
        LLMResponse(
            json.dumps({
                "tool_name": "create_note",
                "arguments": {
                    "title": "Architecture Sprint Decisions",
                    "content": "Decided to adopt unified tool schemas for Phase 3.4.",
                },
            }),
            "test-model",
        ),
        LLMResponse("Created note 'Architecture Sprint Decisions'.", "test-model"),
        # 3. Search notes
        LLMResponse(
            json.dumps({
                "tool_name": "search_notes",
                "arguments": {"query": "Sprint Decisions"},
            }),
            "test-model",
        ),
        LLMResponse("Found note: Architecture Sprint Decisions.", "test-model"),
    ])

    agent = build_agent(llm_client, mcp_client=isolated_live_nl_client)
    try:
        # Event creation
        res_event = await agent.ainvoke("Schedule Architecture Sprint Review on Oct 15 from 2 to 3 PM in Room 401.")
        assert res_event["selected_tool"] == "create_event"
        assert res_event["tool_status"] == "success"
        assert res_event["tool_result"]["data"]["title"] == "Architecture Sprint Review"

        # Note creation
        res_note = await agent.ainvoke("Save a note titled Architecture Sprint Decisions with content: Decided to adopt unified tool schemas.")
        assert res_note["selected_tool"] == "create_note"
        assert res_note["tool_status"] == "success"
        assert res_note["tool_result"]["data"]["title"] == "Architecture Sprint Decisions"

        # Note search
        res_search = await agent.ainvoke("Search my notes for Sprint Decisions.")
        assert res_search["selected_tool"] == "search_notes"
        assert res_search["tool_status"] == "success"
        assert len(res_search["tool_result"]["data"]) == 1
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_nl_get_task():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "get_task", "arguments": {"task_id": 10}}), "test-model"),
        LLMResponse("Task 10 details: Title 'Deploy service', Status 'pending'.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Show details for task 10.")

    assert result["selected_tool"] == "get_task"
    assert result["tool_arguments"] == {"task_id": 10}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_get_event():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "get_event", "arguments": {"event_id": 4}}), "test-model"),
        LLMResponse("Event 4 details: Title 'Team Sync'.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Show details for event 4.")

    assert result["selected_tool"] == "get_event"
    assert result["tool_arguments"] == {"event_id": 4}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_get_note():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "get_note", "arguments": {"note_id": 7}}), "test-model"),
        LLMResponse("Note 7 details: Title 'Ideas'.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Read note 7.")

    assert result["selected_tool"] == "get_note"
    assert result["tool_arguments"] == {"note_id": 7}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_list_notes():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(json.dumps({"tool_name": "list_notes", "arguments": {}}), "test-model"),
        LLMResponse("You have 2 notes saved.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Show all my notes.")

    assert result["selected_tool"] == "list_notes"
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_update_note():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "update_note", "arguments": {"note_id": 7, "content": "Updated content"}}),
            "test-model",
        ),
        LLMResponse("Updated note 7 with new content.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Update note 7 content to Updated content.")

    assert result["selected_tool"] == "update_note"
    assert result["tool_arguments"] == {"note_id": 7, "content": "Updated content"}
    assert result["tool_status"] == "success"


@pytest.mark.asyncio
async def test_nl_invalid_status_enum_rejected():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "create_task", "arguments": {"title": "Task", "status": "invalid_status"}}),
            "test-model",
        ),
        LLMResponse("Invalid status provided.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create task with invalid status.")

    assert result["tool_status"] == "validation_error"
    assert "status must be one of" in result["tool_error"]


@pytest.mark.asyncio
async def test_nl_invalid_priority_enum_rejected():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "create_task", "arguments": {"title": "Task", "priority": "super_urgent"}}),
            "test-model",
        ),
        LLMResponse("Invalid priority provided.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create task with super urgent priority.")

    assert result["tool_status"] == "validation_error"
    assert "priority must be one of" in result["tool_error"]


@pytest.mark.asyncio
async def test_nl_missing_note_content_rejected():
    mcp = MockNLPMCPClient()
    llm_client, _ = make_client([
        LLMResponse(
            json.dumps({"tool_name": "create_note", "arguments": {"title": "Title Only"}}),
            "test-model",
        ),
        LLMResponse("Content is required to create a note.", "test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=mcp)
    result = await agent.ainvoke("Create a note titled Title Only without content.")

    assert result["tool_status"] == "validation_error"
    assert "content" in result["tool_error"]
