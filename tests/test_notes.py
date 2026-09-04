from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.note_service as note_service
from app.services.note_service import create_note, delete_note, get_note, list_notes, search_notes, update_note
from app.utils.validators import NotFoundError


@pytest.fixture
def note_db(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(note_service, "SessionLocal", TestSessionLocal)
    yield TestSessionLocal


def test_create_note(note_db):
    note = create_note(title="Machine Learning Basics", content="Introduction to supervised learning")
    assert note.title == "Machine Learning Basics"
    assert note.id is not None


def test_get_note(note_db):
    created = create_note(title="Important idea", content="Use smaller batches while training")
    fetched = get_note(created.id)
    assert fetched.content == "Use smaller batches while training"


def test_list_notes(note_db):
    create_note(title="A", content="Alpha")
    create_note(title="B", content="Beta")
    notes = list_notes()
    assert len(notes) >= 2


def test_update_note(note_db):
    note = create_note(title="Draft", content="Old content")
    updated = update_note(note.id, title="Updated", content="New content")
    assert updated.title == "Updated"
    assert updated.content == "New content"


def test_delete_note(note_db):
    note = create_note(title="Delete me", content="Remove me")
    assert delete_note(note.id) is True
    with pytest.raises(NotFoundError):
        get_note(note.id)


def test_keyword_search(note_db):
    create_note(title="Machine Learning Basics", content="Gradient descent is key")
    create_note(title="Python revision", content="Practice loops")
    create_note(title="Project ideas", content="Build a portfolio")

    results = search_notes("learning")
    assert any("Machine Learning Basics" == item.title for item in results)


def test_case_insensitive_search(note_db):
    create_note(title="Database normalization", content="Use foreign keys")
    results = search_notes("NORMALIZATION")
    assert any("Database normalization" == item.title for item in results)


def test_missing_note(note_db):
    with pytest.raises(NotFoundError):
        get_note(999)
