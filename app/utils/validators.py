from __future__ import annotations

from datetime import datetime


class ValidationError(ValueError):
    """Raised when user input fails validation."""


class NotFoundError(LookupError):
    """Raised when a requested record is not found."""


class DatabaseError(RuntimeError):
    """Raised when a database operation fails."""


def validate_required_string(value: str | None, field_name: str) -> str:
    """Validate a required non-empty string field."""
    if value is None or not value.strip():
        raise ValidationError(f"{field_name} cannot be empty.")
    return value.strip()


def validate_datetime(value: datetime | None, field_name: str) -> datetime | None:
    """Validate a datetime value, if one is supplied."""
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ValidationError(f"{field_name} must be a valid datetime.")
    return value
