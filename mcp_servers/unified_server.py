from fastmcp import FastMCP

from mcp_servers.task_server import (
    complete_task,
    create_task,
    delete_task,
    get_task,
    list_tasks,
    update_task,
)
from mcp_servers.calendar_server import (
    create_event,
    delete_event,
    get_event,
    list_events,
    update_event,
)
from mcp_servers.notes_server import (
    create_note,
    delete_note,
    get_note,
    list_notes,
    search_notes,
    update_note,
)

server = FastMCP("unified_productivity_server")


for tool in (
    create_task,
    get_task,
    list_tasks,
    update_task,
    complete_task,
    delete_task,
    create_event,
    get_event,
    list_events,
    update_event,
    delete_event,
    create_note,
    get_note,
    list_notes,
    update_note,
    delete_note,
    search_notes,
):
    server.tool()(tool)


if __name__ == "__main__":
    server.run(transport="stdio")
