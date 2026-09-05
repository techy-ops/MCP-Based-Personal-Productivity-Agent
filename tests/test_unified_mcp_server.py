import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.calendar_service as calendar_service
import app.services.note_service as note_service
import app.services.task_service as task_service


@pytest.fixture
def unified_mcp_server(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(calendar_service, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(note_service, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(task_service, "SessionLocal", TestSessionLocal)

    from mcp_servers.unified_server import server
    yield server


async def invoke_tool(server, tool_name, **kwargs):
    result = await server.call_tool(tool_name, kwargs)
    assert result and len(result) > 0
    return json.loads(result[0].text)


@pytest.mark.asyncio
async def test_unified_server_imports_and_initializes(unified_mcp_server):
    assert unified_mcp_server is not None
    tools = await unified_mcp_server.list_tools()
    assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_unified_server_registers_exactly_all_domain_tools(unified_mcp_server):
    tools = await unified_mcp_server.list_tools()
    names = {tool.name for tool in tools}
    expected = {
        "create_task",
        "get_task",
        "list_tasks",
        "update_task",
        "complete_task",
        "delete_task",
        "create_event",
        "get_event",
        "list_events",
        "update_event",
        "delete_event",
        "create_note",
        "get_note",
        "list_notes",
        "update_note",
        "delete_note",
        "search_notes",
    }
    assert len(names) == 17
    assert names == expected


@pytest.mark.asyncio
async def test_unified_task_tool_schemas(unified_mcp_server):
    tools = await unified_mcp_server.list_tools()
    tool_map = {tool.name: tool for tool in tools}
    for name in ["create_task", "get_task", "list_tasks", "update_task", "complete_task", "delete_task"]:
        assert tool_map[name].description
        assert "inputSchema" in tool_map[name].__dict__

    assert "title" in tool_map["create_task"].inputSchema["properties"]
    assert "task_id" in tool_map["get_task"].inputSchema["properties"]
    assert "status" in tool_map["list_tasks"].inputSchema["properties"]
    assert "priority" in tool_map["list_tasks"].inputSchema["properties"]
    assert "task_id" in tool_map["update_task"].inputSchema["properties"]
    assert "task_id" in tool_map["complete_task"].inputSchema["properties"]
    assert "task_id" in tool_map["delete_task"].inputSchema["properties"]


@pytest.mark.asyncio
async def test_unified_task_tools_can_each_be_invoked(unified_mcp_server):
    created = await invoke_tool(unified_mcp_server, "create_task", title="Unified task", priority="high")
    task_id = created["data"]["id"]
    assert created["success"] is True
    assert (await invoke_tool(unified_mcp_server, "get_task", task_id=task_id))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "list_tasks"))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "update_task", task_id=task_id, title="Updated task"))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "complete_task", task_id=task_id))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "delete_task", task_id=task_id))["success"] is True


@pytest.mark.asyncio
async def test_unified_task_workflow(unified_mcp_server):
    created = await invoke_tool(unified_mcp_server, "create_task", title="Workflow task", priority="medium")
    task_id = created["data"]["id"]

    retrieved = await invoke_tool(unified_mcp_server, "get_task", task_id=task_id)
    assert retrieved["data"]["title"] == "Workflow task"

    updated = await invoke_tool(unified_mcp_server, "update_task", task_id=task_id, title="Updated workflow task")
    assert updated["success"] is True
    assert updated["data"]["title"] == "Updated workflow task"

    retrieved_updated = await invoke_tool(unified_mcp_server, "get_task", task_id=task_id)
    assert retrieved_updated["data"]["title"] == "Updated workflow task"

    completed = await invoke_tool(unified_mcp_server, "complete_task", task_id=task_id)
    assert completed["success"] is True
    assert completed["data"]["status"] == "completed"

    listed = await invoke_tool(unified_mcp_server, "list_tasks", status="completed")
    assert listed["success"] is True
    assert any(item["id"] == task_id for item in listed["data"])

    deleted = await invoke_tool(unified_mcp_server, "delete_task", task_id=task_id)
    assert deleted["success"] is True
    assert deleted["data"] is True

    missing = await invoke_tool(unified_mcp_server, "get_task", task_id=task_id)
    assert missing["success"] is False
    assert "not found" in missing["error"].lower()


@pytest.mark.asyncio
async def test_unified_task_error_handling(unified_mcp_server):
    missing_get = await invoke_tool(unified_mcp_server, "get_task", task_id=999)
    assert missing_get["success"] is False
    assert "not found" in missing_get["error"].lower()

    missing_update = await invoke_tool(unified_mcp_server, "update_task", task_id=999, title="Missing")
    assert missing_update["success"] is False
    assert "not found" in missing_update["error"].lower()

    missing_delete = await invoke_tool(unified_mcp_server, "delete_task", task_id=999)
    assert missing_delete["success"] is False
    assert "not found" in missing_delete["error"].lower()

    invalid_create = await invoke_tool(unified_mcp_server, "create_task", title="", priority="high")
    assert invalid_create["success"] is False
    assert "title" in invalid_create["error"].lower()


@pytest.mark.asyncio
async def test_unified_calendar_workflow(unified_mcp_server):
    created = await invoke_tool(
        unified_mcp_server,
        "create_event",
        title="Unified calendar event",
        start_time="2026-09-20T09:00:00",
        end_time="2026-09-20T10:00:00",
        location="Conference room",
    )
    event_id = created["data"]["id"]
    assert created["success"] is True

    retrieved = await invoke_tool(unified_mcp_server, "get_event", event_id=event_id)
    assert retrieved["data"]["title"] == "Unified calendar event"

    listed = await invoke_tool(
        unified_mcp_server,
        "list_events",
        start_date="2026-09-20T00:00:00",
        end_date="2026-09-20T23:59:59",
    )
    assert listed["success"] is True
    assert any(item["id"] == event_id for item in listed["data"])

    updated = await invoke_tool(
        unified_mcp_server,
        "update_event",
        event_id=event_id,
        title="Updated unified event",
    )
    assert updated["success"] is True
    assert updated["data"]["title"] == "Updated unified event"

    retrieved_updated = await invoke_tool(unified_mcp_server, "get_event", event_id=event_id)
    assert retrieved_updated["data"]["title"] == "Updated unified event"

    deleted = await invoke_tool(unified_mcp_server, "delete_event", event_id=event_id)
    assert deleted["success"] is True
    assert deleted["data"] is True

    missing = await invoke_tool(unified_mcp_server, "get_event", event_id=event_id)
    assert missing["success"] is False
    assert "not found" in missing["error"].lower()


@pytest.mark.asyncio
async def test_unified_notes_workflow(unified_mcp_server):
    created = await invoke_tool(
        unified_mcp_server,
        "create_note",
        title="Unified note",
        content="Initial unified content",
    )
    note_id = created["data"]["id"]
    assert created["success"] is True

    retrieved = await invoke_tool(unified_mcp_server, "get_note", note_id=note_id)
    assert retrieved["data"]["content"] == "Initial unified content"

    listed = await invoke_tool(unified_mcp_server, "list_notes")
    assert listed["success"] is True
    assert any(item["id"] == note_id for item in listed["data"])

    updated = await invoke_tool(
        unified_mcp_server,
        "update_note",
        note_id=note_id,
        content="Updated unified content",
    )
    assert updated["success"] is True
    assert updated["data"]["content"] == "Updated unified content"

    retrieved_updated = await invoke_tool(unified_mcp_server, "get_note", note_id=note_id)
    assert retrieved_updated["data"]["content"] == "Updated unified content"

    searched = await invoke_tool(unified_mcp_server, "search_notes", query="updated unified")
    assert searched["success"] is True
    assert any(item["id"] == note_id for item in searched["data"])

    deleted = await invoke_tool(unified_mcp_server, "delete_note", note_id=note_id)
    assert deleted["success"] is True
    assert deleted["data"] is True

    missing = await invoke_tool(unified_mcp_server, "get_note", note_id=note_id)
    assert missing["success"] is False
    assert "not found" in missing["error"].lower()


@pytest.mark.asyncio
async def test_unified_cross_domain_workflow(unified_mcp_server):
    task = await invoke_tool(unified_mcp_server, "create_task", title="Cross-domain task")
    event = await invoke_tool(
        unified_mcp_server,
        "create_event",
        title="Cross-domain event",
        start_time="2026-09-21T09:00:00",
        end_time="2026-09-21T10:00:00",
    )
    note = await invoke_tool(unified_mcp_server, "create_note", title="Cross-domain note", content="Shared server")

    assert (await invoke_tool(unified_mcp_server, "get_task", task_id=task["data"]["id"]))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "get_event", event_id=event["data"]["id"]))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "get_note", note_id=note["data"]["id"]))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "list_tasks"))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "list_events"))["success"] is True
    assert (await invoke_tool(unified_mcp_server, "list_notes"))["success"] is True


@pytest.mark.asyncio
async def test_unified_calendar_and_notes_error_handling(unified_mcp_server):
    missing_event = await invoke_tool(unified_mcp_server, "get_event", event_id=999)
    assert missing_event["success"] is False
    assert "not found" in missing_event["error"].lower()

    missing_event_update = await invoke_tool(unified_mcp_server, "update_event", event_id=999, title="Missing")
    assert missing_event_update["success"] is False
    assert "not found" in missing_event_update["error"].lower()

    missing_event_delete = await invoke_tool(unified_mcp_server, "delete_event", event_id=999)
    assert missing_event_delete["success"] is False
    assert "not found" in missing_event_delete["error"].lower()

    missing_note = await invoke_tool(unified_mcp_server, "get_note", note_id=999)
    assert missing_note["success"] is False
    assert "not found" in missing_note["error"].lower()

    missing_note_update = await invoke_tool(unified_mcp_server, "update_note", note_id=999, title="Missing")
    assert missing_note_update["success"] is False
    assert "not found" in missing_note_update["error"].lower()

    missing_note_delete = await invoke_tool(unified_mcp_server, "delete_note", note_id=999)
    assert missing_note_delete["success"] is False
    assert "not found" in missing_note_delete["error"].lower()

    invalid_search = await invoke_tool(unified_mcp_server, "search_notes", query=" ")
    assert invalid_search["success"] is False
    assert "search query" in invalid_search["error"].lower()


@pytest.mark.asyncio
async def test_unified_calendar_and_notes_schemas(unified_mcp_server):
    tools = await unified_mcp_server.list_tools()
    tool_map = {tool.name: tool for tool in tools}
    for name in [
        "create_event",
        "get_event",
        "list_events",
        "update_event",
        "delete_event",
        "create_note",
        "get_note",
        "list_notes",
        "update_note",
        "delete_note",
        "search_notes",
    ]:
        assert tool_map[name].description
        assert "inputSchema" in tool_map[name].__dict__

    assert "title" in tool_map["create_event"].inputSchema["properties"]
    assert "start_time" in tool_map["create_event"].inputSchema["properties"]
    assert "event_id" in tool_map["get_event"].inputSchema["properties"]
    assert "start_date" in tool_map["list_events"].inputSchema["properties"]
    assert "event_id" in tool_map["update_event"].inputSchema["properties"]
    assert "event_id" in tool_map["delete_event"].inputSchema["properties"]
    assert "title" in tool_map["create_note"].inputSchema["properties"]
    assert "content" in tool_map["create_note"].inputSchema["properties"]
    assert "note_id" in tool_map["get_note"].inputSchema["properties"]
    assert "note_id" in tool_map["update_note"].inputSchema["properties"]
    assert "note_id" in tool_map["delete_note"].inputSchema["properties"]
    assert "query" in tool_map["search_notes"].inputSchema["properties"]


@pytest.mark.asyncio
async def test_unified_empty_data_behavior(unified_mcp_server):
    tasks = await invoke_tool(unified_mcp_server, "list_tasks")
    events = await invoke_tool(unified_mcp_server, "list_events")
    notes = await invoke_tool(unified_mcp_server, "list_notes")
    search = await invoke_tool(unified_mcp_server, "search_notes", query="no matches")

    assert tasks["success"] is True and tasks["data"] == []
    assert events["success"] is True and events["data"] == []
    assert notes["success"] is True and notes["data"] == []
    assert search["success"] is True and search["data"] == []


@pytest.mark.asyncio
async def test_unified_calendar_conflict_error_preserves_service_behavior(unified_mcp_server):
    first = await invoke_tool(
        unified_mcp_server,
        "create_event",
        title="Existing event",
        start_time="2026-09-24T09:00:00",
        end_time="2026-09-24T10:00:00",
    )
    assert first["success"] is True

    conflict = await invoke_tool(
        unified_mcp_server,
        "create_event",
        title="Overlapping event",
        start_time="2026-09-24T09:30:00",
        end_time="2026-09-24T10:30:00",
    )
    assert conflict["success"] is False
    assert "conflict" in conflict["error"].lower()


@pytest.mark.asyncio
async def test_unified_response_contracts(unified_mcp_server):
    task = await invoke_tool(unified_mcp_server, "create_task", title="Response task")
    event = await invoke_tool(
        unified_mcp_server,
        "create_event",
        title="Response event",
        start_time="2026-09-25T09:00:00",
        end_time="2026-09-25T10:00:00",
    )
    note = await invoke_tool(unified_mcp_server, "create_note", title="Response note", content="Response content")

    for payload in [task, event, note]:
        assert payload["success"] is True
        assert set(payload) == {"success", "data", "message"}
        assert isinstance(payload["data"], dict)
        assert isinstance(payload["message"], str)

    failures = [
        await invoke_tool(unified_mcp_server, "get_task", task_id=999),
        await invoke_tool(unified_mcp_server, "get_event", event_id=999),
        await invoke_tool(unified_mcp_server, "get_note", note_id=999),
    ]
    for payload in failures:
        assert payload["success"] is False
        assert set(payload) == {"success", "error"}
        assert isinstance(payload["error"], str)


@pytest.mark.asyncio
async def test_unified_repeated_reads_are_stable(unified_mcp_server):
    task = await invoke_tool(unified_mcp_server, "create_task", title="Repeated task")
    event = await invoke_tool(
        unified_mcp_server,
        "create_event",
        title="Repeated event",
        start_time="2026-09-26T09:00:00",
        end_time="2026-09-26T10:00:00",
    )
    note = await invoke_tool(unified_mcp_server, "create_note", title="Repeated note", content="Repeated content")

    for _ in range(5):
        assert (await invoke_tool(unified_mcp_server, "get_task", task_id=task["data"]["id"]))["success"] is True
        assert (await invoke_tool(unified_mcp_server, "list_tasks"))["success"] is True
        assert (await invoke_tool(unified_mcp_server, "get_event", event_id=event["data"]["id"]))["success"] is True
        assert (await invoke_tool(unified_mcp_server, "list_events"))["success"] is True
        assert (await invoke_tool(unified_mcp_server, "get_note", note_id=note["data"]["id"]))["success"] is True
        assert (await invoke_tool(unified_mcp_server, "list_notes"))["success"] is True
        assert (await invoke_tool(unified_mcp_server, "search_notes", query="repeated"))["success"] is True
