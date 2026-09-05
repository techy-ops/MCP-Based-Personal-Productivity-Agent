from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from app.services import note_service
from app.utils.validators import DatabaseError, NotFoundError, ValidationError

server = FastMCP("notes_server")


def _validate_note_id(note_id: int) -> int:
    if not isinstance(note_id, int) or note_id <= 0:
        raise ValidationError("note_id must be a positive integer.")
    return note_id


def _serialize_note(note: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy note model into a JSON-friendly dictionary."""
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "created_at": note.created_at.isoformat() if note.created_at is not None else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at is not None else None,
    }


def _success_response(data: Any, message: str) -> dict[str, Any]:
    return {"success": True, "data": data, "message": message}


def _error_response(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


@server.tool(
    name="create_note",
    description=(
        "Create a new note containing a title and content for storing reusable personal information, ideas, or reference material."
    ),
)
def create_note(title: str, content: str) -> dict[str, Any]:
    """Create a note by delegating to the existing Phase 1 note service."""
    try:
        if title is None or not isinstance(title, str) or not title.strip():
            raise ValidationError("title cannot be empty")
        if content is None or not isinstance(content, str) or not content.strip():
            raise ValidationError("content cannot be empty")
        note = note_service.create_note(title=title, content=content)
        return _success_response(_serialize_note(note), "Note created successfully")
    except (ValidationError, ValueError, TypeError) as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to create note: {exc}")


@server.tool(
    name="get_note",
    description="Retrieve a single note by its note ID when the full stored title and content are needed.",
)
def get_note(note_id: int) -> dict[str, Any]:
    """Fetch a single note using the Phase 1 note service layer."""
    try:
        validated_id = _validate_note_id(note_id)
        note = note_service.get_note(validated_id)
        return _success_response(_serialize_note(note), "Note retrieved successfully")
    except NotFoundError as exc:
        return _error_response(str(exc))
    except (ValidationError, ValueError, TypeError) as exc:
        return _error_response(str(exc))
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to retrieve note: {exc}")


@server.tool(
    name="list_notes",
    description="Retrieve all available notes ordered by the existing Notes service from most recently updated.",
)
def list_notes() -> dict[str, Any]:
    """List notes through the Phase 1 note service layer."""
    try:
        notes = note_service.list_notes()
        return _success_response([_serialize_note(note) for note in notes], "Notes retrieved successfully")
    except DatabaseError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # pragma: no cover - safety fallback
        return _error_response(f"Unable to list notes: {exc}")


if __name__ == "__main__":
    server.run(transport="stdio")
