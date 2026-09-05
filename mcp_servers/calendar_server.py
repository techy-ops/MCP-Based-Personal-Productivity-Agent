from __future__ import annotations

from datetime import datetime
from typing import Any

from fastmcp import FastMCP

from app.services import calendar_service
from app.utils.validators import DatabaseError, NotFoundError, ValidationError

server = FastMCP("calendar_server")


def _validate_event_id(event_id: int) -> int:
    if not isinstance(event_id, int) or event_id <= 0:
        raise ValidationError("event_id must be a positive integer.")
    return event_id


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _serialize_event(event: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy calendar event model into a JSON-friendly dictionary."""
    return {
        "id": event.id,
        "title": event.title,
        "description": event.description,
        "start_time": event.start_time.isoformat() if event.start_time is not None else None,
        "end_time": event.end_time.isoformat() if event.end_time is not None else None,
        "location": event.location,
        "created_at": event.created_at.isoformat() if event.created_at is not None else None,
        "updated_at": event.updated_at.isoformat() if event.updated_at is not None else None,
    }


def _success_response(data: Any, message: str) -> dict[str, Any]:
    return {"success": True, "data": data, "message": message}


def _error_response(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


@server.tool(
    name="create_event",
    description=(
        "Create a new calendar event with a title, optional description, valid start and end times, and optional location. "
        "Use this when a meeting, appointment, or time-block needs to be added to the personal productivity calendar. "
        "The existing calendar service enforces chronological validation and overlap conflict detection."
    ),
)
def create_event(
    title: str,
    description: str | None = None,
    start_time: str | datetime | None = None,
    end_time: str | datetime | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Create a calendar event by delegating to the Phase 1 calendar service."""
    try:
        if title is None or not str(title).strip():
            raise ValidationError("title cannot be empty")
        if start_time is None:
            raise ValidationError("start_time is required.")
        if end_time is None:
            raise ValidationError("end_time is required.")

        event = calendar_service.create_event(
            title=str(title).strip(),
            description=description,
            start_time=_coerce_datetime(start_time),
            end_time=_coerce_datetime(end_time),
            location=location,
        )
        return _success_response(_serialize_event(event), "Calendar event created successfully")
    except NotFoundError as exc:
        return _error_response(str(exc))
    except (ValidationError, ValueError, TypeError) as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to create calendar event: {exc}")


@server.tool(
    name="get_event",
    description=(
        "Retrieve a single calendar event by its event ID. Use this when you need the full details of an existing calendar record."
    ),
)
def get_event(event_id: int) -> dict[str, Any]:
    """Fetch a single calendar event using the Phase 1 service layer."""
    try:
        validated_id = _validate_event_id(event_id)
        event = calendar_service.get_event(validated_id)
        return _success_response(_serialize_event(event), "Calendar event retrieved successfully")
    except NotFoundError as exc:
        return _error_response(str(exc))
    except (ValidationError, ValueError, TypeError) as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to retrieve calendar event: {exc}")


if __name__ == "__main__":
    server.run(transport="stdio")
