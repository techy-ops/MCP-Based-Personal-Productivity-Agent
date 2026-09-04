from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.calendar_service as calendar_service
from app.services.calendar_service import create_event, delete_event, get_event, list_events, update_event
from app.utils.validators import NotFoundError, ValidationError


@pytest.fixture
def calendar_db(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(calendar_service, "SessionLocal", TestSessionLocal)
    yield TestSessionLocal


def test_create_event(calendar_db):
    start = datetime(2026, 9, 10, 9, 0)
    end = datetime(2026, 9, 10, 10, 0)
    event = create_event(title="Project meeting", start_time=start, end_time=end)
    assert event.title == "Project meeting"
    assert event.id is not None


def test_get_event(calendar_db):
    start = datetime(2026, 9, 10, 9, 0)
    end = datetime(2026, 9, 10, 10, 0)
    created = create_event(title="Review", start_time=start, end_time=end)
    fetched = get_event(created.id)
    assert fetched.title == "Review"


def test_list_events(calendar_db):
    create_event(title="A", start_time=datetime(2026, 9, 10, 9, 0), end_time=datetime(2026, 9, 10, 10, 0))
    create_event(title="B", start_time=datetime(2026, 9, 11, 9, 0), end_time=datetime(2026, 9, 11, 10, 0))
    events = list_events()
    assert len(events) >= 2


def test_update_event(calendar_db):
    start = datetime(2026, 9, 10, 9, 0)
    end = datetime(2026, 9, 10, 10, 0)
    event = create_event(title="Old title", start_time=start, end_time=end)
    updated = update_event(event.id, title="New title", end_time=end + timedelta(hours=1))
    assert updated.title == "New title"


def test_delete_event(calendar_db):
    start = datetime(2026, 9, 10, 9, 0)
    end = datetime(2026, 9, 10, 10, 0)
    event = create_event(title="Delete me", start_time=start, end_time=end)
    assert delete_event(event.id) is True
    with pytest.raises(NotFoundError):
        get_event(event.id)


def test_invalid_time_range(calendar_db):
    with pytest.raises(ValidationError):
        create_event(title="Bad range", start_time=datetime(2026, 9, 10, 15, 0), end_time=datetime(2026, 9, 10, 14, 0))


def test_calendar_conflict_detection(calendar_db):
    start = datetime(2026, 9, 10, 10, 0)
    end = datetime(2026, 9, 10, 11, 0)
    create_event(title="Existing", start_time=start, end_time=end)

    with pytest.raises(ValueError, match="Calendar conflict"):
        create_event(title="Overlap", start_time=datetime(2026, 9, 10, 10, 30), end_time=datetime(2026, 9, 10, 11, 30))


def test_missing_event(calendar_db):
    with pytest.raises(NotFoundError):
        get_event(999)
