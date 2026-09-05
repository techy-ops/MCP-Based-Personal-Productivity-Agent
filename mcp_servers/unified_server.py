from fastmcp import FastMCP

from mcp_servers.task_server import (
    complete_task,
    create_task,
    delete_task,
    get_task,
    list_tasks,
    update_task,
)

server = FastMCP("unified_productivity_server")


for task_tool in (create_task, get_task, list_tasks, update_task, complete_task, delete_task):
    server.tool()(task_tool)


if __name__ == "__main__":
    server.run(transport="stdio")
