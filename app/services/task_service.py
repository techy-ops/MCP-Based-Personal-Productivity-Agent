from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.database.models import Task
from app.schemas.task import TaskCreate, TaskUpdate, VALID_TASK_PRIORITIES, VALID_TASK_STATUSES
from app.utils.validators import DatabaseError, NotFoundError, ValidationError


def _normalize_task_payload(payload: dict | None = None, *, update: bool = False) -> dict:
    """Validate task payloads using the task schema."""
    if payload is None:
        payload = {}
    try:
        model = TaskUpdate(**payload) if update else TaskCreate(**payload)
    except PydanticValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else "Invalid task data."
        raise ValidationError(message) from exc
    return model.model_dump(exclude_unset=True)


def _get_task_or_raise(session: Session, task_id: int) -> Task:
    task = session.get(Task, task_id)
    if task is None:
        raise NotFoundError(f"Task with id {task_id} was not found.")
    return task


def create_task(
    *,
    title: str,
    description: str | None = None,
    status: str = "pending",
    priority: str = "medium",
    due_date: datetime | None = None,
) -> Task:
    """Create a new task after validating its data and business rules."""
    payload = _normalize_task_payload(
        {
            "title": title,
            "description": description,
            "status": status,
            "priority": priority,
            "due_date": due_date,
        }
    )

    session = SessionLocal()
    try:
        task = Task(**payload)
        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to create task.") from exc
    finally:
        session.close()


def get_task(task_id: int) -> Task:
    """Fetch a single task by ID."""
    session = SessionLocal()
    try:
        return _get_task_or_raise(session, task_id)
    finally:
        session.close()


def list_tasks(*, status: str | None = None, priority: str | None = None) -> list[Task]:
    """List tasks with optional status and priority filters."""
    if status is not None and status not in VALID_TASK_STATUSES:
        raise ValidationError("status must be one of: pending, in_progress, completed")
    if priority is not None and priority not in VALID_TASK_PRIORITIES:
        raise ValidationError("priority must be one of: low, medium, high")

    session = SessionLocal()
    try:
        stmt = select(Task)
        if status is not None:
            stmt = stmt.where(Task.status == status)
        if priority is not None:
            stmt = stmt.where(Task.priority == priority)
        stmt = stmt.order_by(Task.updated_at.desc(), Task.created_at.desc())
        return list(session.execute(stmt).scalars().all())
    finally:
        session.close()


def update_task(task_id: int, **kwargs) -> Task:
    """Update a task using explicit fields with validation."""
    session = SessionLocal()
    try:
        task = _get_task_or_raise(session, task_id)
        payload = _normalize_task_payload(kwargs, update=True)
        for field, value in payload.items():
            setattr(task, field, value)
        task.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(task)
        return task
    except ValidationError:
        raise
    except NotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to update task.") from exc
    finally:
        session.close()


def complete_task(task_id: int) -> Task:
    """Mark a task as completed."""
    return update_task(task_id, status="completed")


def delete_task(task_id: int) -> bool:
    """Delete a task permanently."""
    session = SessionLocal()
    try:
        task = _get_task_or_raise(session, task_id)
        session.delete(task)
        session.commit()
        return True
    except NotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to delete task.") from exc
    finally:
        session.close()
