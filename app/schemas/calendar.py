from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALID_CALENDAR_TITLE = "title cannot be empty"


class CalendarEventCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str = Field(..., min_length=1)
    description: str | None = None
    start_time: datetime
    end_time: datetime
    location: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError(VALID_CALENDAR_TITLE)
        return value.strip()

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, value: datetime, info):
        start_time = info.data.get("start_time")
        if start_time is not None and value <= start_time:
            raise ValueError("end_time must be after start_time")
        return value


class CalendarEventUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str | None = None
    description: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    location: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(VALID_CALENDAR_TITLE)
        return value.strip() if value is not None else None

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, value: datetime | None, info):
        if value is None:
            return value
        start_time = info.data.get("start_time")
        if start_time is None:
            return value
        if value <= start_time:
            raise ValueError("end_time must be after start_time")
        return value


class CalendarEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str | None = None
    start_time: datetime
    end_time: datetime
    location: str | None = None
    created_at: datetime
    updated_at: datetime
