from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.validators import ValidationError

VALID_TASK_STATUSES = {"pending", "in_progress", "completed"}
VALID_TASK_PRIORITIES = {"low", "medium", "high"}


class TaskCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str = Field(..., min_length=1)
    description: str | None = None
    status: str = Field(default="pending")
    priority: str = Field(default="medium")
    due_date: datetime | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError("title cannot be empty")
        return value.strip()

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in VALID_TASK_STATUSES:
            raise ValueError("status must be one of: pending, in_progress, completed")
        return normalized

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in VALID_TASK_PRIORITIES:
            raise ValueError("priority must be one of: low, medium, high")
        return normalized

    @field_validator("due_date")
    @classmethod
    def validate_due_date(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError("due_date must be a valid datetime")
        return value


class TaskUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    due_date: datetime | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title cannot be empty")
        return value.strip() if value is not None else None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if normalized not in VALID_TASK_STATUSES:
            raise ValueError("status must be one of: pending, in_progress, completed")
        return normalized

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if normalized not in VALID_TASK_PRIORITIES:
            raise ValueError("priority must be one of: low, medium, high")
        return normalized

    @field_validator("due_date")
    @classmethod
    def validate_due_date(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError("due_date must be a valid datetime")
        return value


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str | None = None
    status: str
    priority: str
    due_date: datetime | None = None
    created_at: datetime
    updated_at: datetime
