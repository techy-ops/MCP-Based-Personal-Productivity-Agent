import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.note_service as note_service


@pytest.fixture
def notes_mcp_server(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(note_service, "SessionLocal", TestSessionLocal)

    from mcp_servers.notes_server import server
    yield server


async def invoke_tool(server, tool_name, **kwargs):
    result = await server.call_tool(tool_name, kwargs)
    assert result and len(result) > 0
    return json.loads(result[0].text)


@pytest.mark.asyncio
async def test_notes_mcp_server_imports_and_initializes(notes_mcp_server):
    assert notes_mcp_server is not None
    tools = await notes_mcp_server.list_tools()
    assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_notes_mcp_server_registers_only_phase_231_tools(notes_mcp_server):
    tools = await notes_mcp_server.list_tools()
    names = {tool.name for tool in tools}
    assert names == {"create_note", "get_note", "list_notes"}
    assert not names.intersection({"update_note", "delete_note", "search_notes"})


@pytest.mark.asyncio
async def test_create_note_tool_success(notes_mcp_server):
    payload = await invoke_tool(
        notes_mcp_server,
        "create_note",
        title="Deployment checklist",
        content="Verify migrations before release.",
    )
    assert payload["success"] is True
    assert payload["data"]["title"] == "Deployment checklist"
    assert payload["data"]["content"] == "Verify migrations before release."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("title", "content", "expected"),
    [("", "Content", "title"), ("Title", "", "content")],
)
async def test_create_note_tool_invalid_input(notes_mcp_server, title, content, expected):
    payload = await invoke_tool(notes_mcp_server, "create_note", title=title, content=content)
    assert payload["success"] is False
    assert expected in payload["error"].lower()


@pytest.mark.asyncio
async def test_get_note_tool_success(notes_mcp_server):
    created = await invoke_tool(notes_mcp_server, "create_note", title="Reference", content="Useful details")
    payload = await invoke_tool(notes_mcp_server, "get_note", note_id=created["data"]["id"])
    assert payload["success"] is True
    assert payload["data"]["title"] == "Reference"
    assert payload["data"]["content"] == "Useful details"


@pytest.mark.asyncio
async def test_get_note_tool_missing_note(notes_mcp_server):
    payload = await invoke_tool(notes_mcp_server, "get_note", note_id=999)
    assert payload["success"] is False
    assert "not found" in payload["error"].lower()


@pytest.mark.asyncio
async def test_get_note_tool_invalid_id(notes_mcp_server):
    payload = await invoke_tool(notes_mcp_server, "get_note", note_id=0)
    assert payload["success"] is False
    assert "note_id" in payload["error"].lower()


@pytest.mark.asyncio
async def test_list_notes_tool_success(notes_mcp_server):
    await invoke_tool(notes_mcp_server, "create_note", title="First", content="Alpha")
    await invoke_tool(notes_mcp_server, "create_note", title="Second", content="Beta")
    payload = await invoke_tool(notes_mcp_server, "list_notes")
    assert payload["success"] is True
    assert {item["title"] for item in payload["data"]} == {"First", "Second"}


@pytest.mark.asyncio
async def test_list_notes_tool_empty_result(notes_mcp_server):
    payload = await invoke_tool(notes_mcp_server, "list_notes")
    assert payload == {
        "success": True,
        "data": [],
        "message": "Notes retrieved successfully",
    }


@pytest.mark.asyncio
async def test_notes_mcp_end_to_end_workflow(notes_mcp_server):
    note_a = await invoke_tool(notes_mcp_server, "create_note", title="Project A", content="A details")
    note_b = await invoke_tool(notes_mcp_server, "create_note", title="Project B", content="B details")

    retrieved = await invoke_tool(notes_mcp_server, "get_note", note_id=note_a["data"]["id"])
    assert retrieved["success"] is True
    assert retrieved["data"]["content"] == "A details"

    listed = await invoke_tool(notes_mcp_server, "list_notes")
    assert listed["success"] is True
    assert {item["id"] for item in listed["data"]} == {note_a["data"]["id"], note_b["data"]["id"]}

    missing = await invoke_tool(notes_mcp_server, "get_note", note_id=999)
    assert missing["success"] is False
    assert "not found" in missing["error"].lower()


@pytest.mark.asyncio
async def test_notes_mcp_tool_descriptions_and_schemas(notes_mcp_server):
    tools = await notes_mcp_server.list_tools()
    tool_map = {tool.name: tool for tool in tools}
    for name in ["create_note", "get_note", "list_notes"]:
        assert tool_map[name].description
        assert "inputSchema" in tool_map[name].__dict__

    assert "title" in tool_map["create_note"].inputSchema["properties"]
    assert "content" in tool_map["create_note"].inputSchema["properties"]
    assert "note_id" in tool_map["get_note"].inputSchema["properties"]
