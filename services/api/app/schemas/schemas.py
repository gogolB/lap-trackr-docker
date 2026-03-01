from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.models.models import SessionStatus


# ── Auth ──────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: UUID
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Grading Results ──────────────────────────────────────────────────


class GradingResultOut(BaseModel):
    id: UUID
    session_id: UUID
    workspace_volume: float | None = None
    avg_speed: float | None = None
    max_jerk: float | None = None
    path_length: float | None = None
    economy_of_motion: float | None = None
    total_time: float | None = None
    completed_at: datetime | None = None
    error: str | None = None

    model_config = {"from_attributes": True}


# ── Sessions ─────────────────────────────────────────────────────────


class SessionOut(BaseModel):
    id: UUID
    user_id: UUID
    started_at: datetime
    stopped_at: datetime | None = None
    status: SessionStatus
    on_axis_path: str | None = None
    off_axis_path: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionDetailOut(SessionOut):
    grading_result: GradingResultOut | None = None
