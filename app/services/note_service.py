from __future__ import annotations

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.database.models import Note
from app.schemas.note import NoteCreate, NoteUpdate
from app.utils.validators import DatabaseError, NotFoundError, ValidationError


def _normalize_note_payload(payload: dict | None = None, *, update: bool = False) -> dict:
    """Validate note payloads using the note schema."""
    if payload is None:
        payload = {}
    try:
        model = NoteUpdate(**payload) if update else NoteCreate(**payload)
    except PydanticValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else "Invalid note data."
        raise ValidationError(message) from exc
    return model.model_dump(exclude_unset=True)


def _get_note_or_raise(session: Session, note_id: int) -> Note:
    note = session.get(Note, note_id)
    if note is None:
        raise NotFoundError(f"Note with id {note_id} was not found.")
    return note


def create_note(*, title: str, content: str) -> Note:
    """Create a new note."""
    payload = _normalize_note_payload({"title": title, "content": content})
    session = SessionLocal()
    try:
        note = Note(**payload)
        session.add(note)
        session.commit()
        session.refresh(note)
        return note
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to create note.") from exc
    finally:
        session.close()


def get_note(note_id: int) -> Note:
    """Fetch a single note by ID."""
    session = SessionLocal()
    try:
        return _get_note_or_raise(session, note_id)
    finally:
        session.close()


def list_notes() -> list[Note]:
    """List notes, ordered by most recently updated."""
    session = SessionLocal()
    try:
        stmt = select(Note).order_by(Note.updated_at.desc(), Note.created_at.desc())
        return list(session.execute(stmt).scalars().all())
    finally:
        session.close()


def update_note(note_id: int, **kwargs) -> Note:
    """Update an existing note."""
    session = SessionLocal()
    try:
        note = _get_note_or_raise(session, note_id)
        payload = _normalize_note_payload(kwargs, update=True)
        for field, value in payload.items():
            setattr(note, field, value)
        session.commit()
        session.refresh(note)
        return note
    except ValidationError:
        raise
    except NotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to update note.") from exc
    finally:
        session.close()


def delete_note(note_id: int) -> bool:
    """Delete a note permanently."""
    session = SessionLocal()
    try:
        note = _get_note_or_raise(session, note_id)
        session.delete(note)
        session.commit()
        return True
    except NotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to delete note.") from exc
    finally:
        session.close()


def search_notes(query: str) -> list[Note]:
    """Search notes by title or content using case-insensitive SQL matching."""
    if query is None or not query.strip():
        raise ValidationError("search query cannot be empty.")
    term = query.strip()
    session = SessionLocal()
    try:
        stmt = select(Note).where(
            or_(Note.title.ilike(f"%{term}%"), Note.content.ilike(f"%{term}%"))
        ).order_by(Note.updated_at.desc(), Note.created_at.desc())
        return list(session.execute(stmt).scalars().all())
    finally:
        session.close()
