from __future__ import annotations

from datetime import datetime
from typing import Any

from fastmcp import FastMCP

from app.schemas.task import VALID_TASK_PRIORITIES, VALID_TASK_STATUSES
from app.services import task_service
from app.utils.validators import DatabaseError, NotFoundError, ValidationError

server = FastMCP("task_server")


def _validate_task_id(task_id: int) -> int:
    if not isinstance(task_id, int) or task_id <= 0:
        raise ValidationError("task_id must be a positive integer.")
    return task_id


def _validate_task_payload(
    *,
    title: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> None:
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ValidationError("title cannot be empty.")
    if status is not None:
        normalized = str(status).strip()
        if normalized not in VALID_TASK_STATUSES:
            raise ValidationError("status must be one of: pending, in_progress, completed")
    if priority is not None:
        normalized = str(priority).strip()
        if normalized not in VALID_TASK_PRIORITIES:
            raise ValidationError("priority must be one of: low, medium, high")


def _serialize_task(task: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy task model into a JSON-friendly dictionary."""
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "status": task.status,
        "priority": task.priority,
        "due_date": task.due_date.isoformat() if task.due_date is not None else None,
        "created_at": task.created_at.isoformat() if task.created_at is not None else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at is not None else None,
    }


def _success_response(data: Any, message: str) -> dict[str, Any]:
    return {"success": True, "data": data, "message": message}


def _error_response(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@server.tool(
    name="create_task",
    description=(
        "Create a new productivity task with a title, optional description, status, priority, and due date. "
        "Use this when a workflow needs to add a new task to the personal productivity system."
    ),
)
def create_task(
    title: str,
    description: str | None = None,
    status: str = "pending",
    priority: str = "medium",
    due_date: str | datetime | None = None,
) -> dict[str, Any]:
    """Create a task by delegating to the existing Phase 1 task service."""
    try:
        _validate_task_payload(title=title, status=status, priority=priority)
        task = task_service.create_task(
            title=title,
            description=description,
            status=status,
            priority=priority,
            due_date=_coerce_datetime(due_date),
        )
        return _success_response(_serialize_task(task), "Task created successfully")
    except (ValidationError, ValueError, TypeError) as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to create task: {exc}")


@server.tool(
    name="get_task",
    description=(
        "Retrieve a single task by ID. Use this when you need the full task details for a known task record."
    ),
)
def get_task(task_id: int) -> dict[str, Any]:
    """Fetch one task by ID using the Phase 1 service layer."""
    try:
        validated_id = _validate_task_id(task_id)
        task = task_service.get_task(validated_id)
        return _success_response(_serialize_task(task), "Task retrieved successfully")
    except NotFoundError as exc:
        return _error_response(str(exc))
    except ValidationError as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to retrieve task: {exc}")


@server.tool(
    name="list_tasks",
    description=(
        "List productivity tasks, optionally filtered by status or priority. Use this to inspect the current task backlog."
    ),
)
def list_tasks(status: str | None = None, priority: str | None = None) -> dict[str, Any]:
    """List tasks according to the existing service filters."""
    try:
        _validate_task_payload(status=status, priority=priority)
        tasks = task_service.list_tasks(status=status, priority=priority)
        return _success_response([_serialize_task(task) for task in tasks], "Tasks retrieved successfully")
    except ValidationError as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to list tasks: {exc}")


@server.tool(
    name="update_task",
    description=(
        "Update an existing task by ID. Accepts the fields supported by the task model, such as title, description, status, priority, and due date."
    ),
)
def update_task(
    task_id: int,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    due_date: str | datetime | None = None,
) -> dict[str, Any]:
    """Update an existing task using the Phase 1 task service implementation."""
    payload: dict[str, Any] = {}
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description
    if status is not None:
        payload["status"] = status
    if priority is not None:
        payload["priority"] = priority
    if due_date is not None:
        payload["due_date"] = _coerce_datetime(due_date)

    try:
        validated_id = _validate_task_id(task_id)
        _validate_task_payload(title=title, status=status, priority=priority)
        task = task_service.update_task(validated_id, **payload)
        return _success_response(_serialize_task(task), "Task updated successfully")
    except NotFoundError as exc:
        return _error_response(str(exc))
    except (ValidationError, ValueError, TypeError) as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to update task: {exc}")


@server.tool(
    name="complete_task",
    description=(
        "Mark an existing task as completed. Use this once a task has been finished or delivered."
    ),
)
def complete_task(task_id: int) -> dict[str, Any]:
    """Complete a task using the Phase 1 service function."""
    try:
        validated_id = _validate_task_id(task_id)
        task = task_service.complete_task(validated_id)
        return _success_response(_serialize_task(task), "Task marked as completed")
    except NotFoundError as exc:
        return _error_response(str(exc))
    except (ValidationError, ValueError, TypeError) as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to complete task: {exc}")


@server.tool(
    name="delete_task",
    description=(
        "Delete a task permanently by ID. Use this when the task is no longer needed or should be removed from the backlog."
    ),
)
def delete_task(task_id: int) -> dict[str, Any]:
    """Delete a task through the Phase 1 task service."""
    try:
        validated_id = _validate_task_id(task_id)
        deleted = task_service.delete_task(validated_id)
        return _success_response(deleted, "Task deleted successfully")
    except NotFoundError as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to delete task: {exc}")


if __name__ == "__main__":
    server.run(transport="stdio")
