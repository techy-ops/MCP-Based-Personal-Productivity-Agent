from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.database.models import CalendarEvent
from app.schemas.calendar import CalendarEventCreate, CalendarEventUpdate
from app.utils.validators import DatabaseError, NotFoundError, ValidationError


def _normalize_event_payload(payload: dict | None = None, *, update: bool = False) -> dict:
    """Validate calendar payloads against the event schema."""
    if payload is None:
        payload = {}
    try:
        model = CalendarEventUpdate(**payload) if update else CalendarEventCreate(**payload)
    except PydanticValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else "Invalid calendar event data."
        raise ValidationError(message) from exc
    return model.model_dump(exclude_unset=True)


def _get_event_or_raise(session: Session, event_id: int) -> CalendarEvent:
    event = session.get(CalendarEvent, event_id)
    if event is None:
        raise NotFoundError(f"Calendar event with id {event_id} was not found.")
    return event


def _has_conflict(session: Session, start_time: datetime, end_time: datetime, *, exclude_id: int | None = None) -> bool:
    """Return True when another event overlaps with the requested time range."""
    stmt = select(CalendarEvent).where(
        CalendarEvent.start_time < end_time,
        CalendarEvent.end_time > start_time,
    )
    if exclude_id is not None:
        stmt = stmt.where(CalendarEvent.id != exclude_id)
    return session.execute(stmt).scalars().first() is not None


def create_event(
    *,
    title: str,
    description: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    location: str | None = None,
) -> CalendarEvent:
    """Create a calendar event with basic conflict detection."""
    if start_time is None:
        raise ValidationError("start_time is required.")
    if end_time is None:
        raise ValidationError("end_time is required.")

    payload = _normalize_event_payload(
        {
            "title": title,
            "description": description,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
        }
    )

    session = SessionLocal()
    try:
        if _has_conflict(session, payload["start_time"], payload["end_time"]):
            raise ValueError("Calendar conflict: another event already exists during this time.")
        event = CalendarEvent(**payload)
        session.add(event)
        session.commit()
        session.refresh(event)
        return event
    except ValueError:
        session.rollback()
        raise
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to create calendar event.") from exc
    finally:
        session.close()


def get_event(event_id: int) -> CalendarEvent:
    """Fetch one calendar event."""
    session = SessionLocal()
    try:
        return _get_event_or_raise(session, event_id)
    finally:
        session.close()


def list_events(*, start_date: datetime | None = None, end_date: datetime | None = None) -> list[CalendarEvent]:
    """List events, optionally filtered by date range."""
    session = SessionLocal()
    try:
        stmt = select(CalendarEvent)
        if start_date is not None:
            stmt = stmt.where(CalendarEvent.start_time >= start_date)
        if end_date is not None:
            stmt = stmt.where(CalendarEvent.end_time <= end_date)
        stmt = stmt.order_by(CalendarEvent.start_time.asc(), CalendarEvent.created_at.desc())
        return list(session.execute(stmt).scalars().all())
    finally:
        session.close()


def update_event(event_id: int, **kwargs) -> CalendarEvent:
    """Update an existing event and ensure no time conflicts form."""
    session = SessionLocal()
    try:
        event = _get_event_or_raise(session, event_id)
        start_time = kwargs.get("start_time", event.start_time)
        end_time = kwargs.get("end_time", event.end_time)
        payload = _normalize_event_payload({**event.__dict__, **kwargs}, update=True)
        if "start_time" in payload:
            start_time = payload["start_time"]
        if "end_time" in payload:
            end_time = payload["end_time"]
        if _has_conflict(session, start_time, end_time, exclude_id=event_id):
            raise ValueError("Calendar conflict: another event already exists during this time.")
        for field, value in payload.items():
            setattr(event, field, value)
        event.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(event)
        return event
    except ValueError:
        session.rollback()
        raise
    except NotFoundError:
        raise
    except ValidationError:
        raise
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to update calendar event.") from exc
    finally:
        session.close()


def delete_event(event_id: int) -> bool:
    """Delete a calendar event permanently."""
    session = SessionLocal()
    try:
        event = _get_event_or_raise(session, event_id)
        session.delete(event)
        session.commit()
        return True
    except NotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - fallback guard
        session.rollback()
        raise DatabaseError("Unable to delete calendar event.") from exc
    finally:
        session.close()
