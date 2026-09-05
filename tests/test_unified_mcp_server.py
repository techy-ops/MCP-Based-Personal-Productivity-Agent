import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.task_service as task_service


@pytest.fixture
def unified_mcp_server(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
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
async def test_unified_server_registers_exactly_task_tools(unified_mcp_server):
    tools = await unified_mcp_server.list_tools()
    names = {tool.name for tool in tools}
    assert names == {
        "create_task",
        "get_task",
        "list_tasks",
        "update_task",
        "complete_task",
        "delete_task",
    }
    assert not names.intersection(
        {
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
    )


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
