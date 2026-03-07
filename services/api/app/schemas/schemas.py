from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.models.models import ModelStatus, SessionStatus


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


class SessionStartRequest(BaseModel):
    name: str = ""


class SessionOut(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    started_at: datetime
    stopped_at: datetime | None = None
    status: SessionStatus
    on_axis_path: str | None = None
    off_axis_path: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionDetailOut(SessionOut):
    grading_result: GradingResultOut | None = None


# ── ML Models ───────────────────────────────────────────────────────


# ── Calibration ─────────────────────────────────────────────────────


class BoardConfig(BaseModel):
    rows: int
    cols: int
    square_size_mm: float
    marker_size_mm: float
    aruco_dict: str


class CalibrationCaptureResult(BaseModel):
    success: bool
    markers_detected: int = 0
    charuco_corners: int = 0
    coverage_pct: float = 0.0
    total_captures: int = 0
    preview_jpeg_b64: str | None = None


class CalibrationStatus(BaseModel):
    total_captures: int = 0
    board_config: BoardConfig


class CalibrationOut(BaseModel):
    id: UUID
    camera_name: str
    is_default: bool
    session_id: UUID | None = None
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    extrinsic_matrix: list | None = None
    board_rows: int
    board_cols: int
    square_size_mm: float
    marker_size_mm: float
    aruco_dict: str
    reprojection_error: float | None = None
    num_frames_used: int | None = None
    is_global: bool = False
    calibration_path: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── ML Models ───────────────────────────────────────────────────────


# ── Camera Config ──────────────────────────────────────────────────


class CameraConfigOut(BaseModel):
    on_axis_serial: str
    off_axis_serial: str
    on_axis_swap_eyes: bool = False
    off_axis_swap_eyes: bool = False
    on_axis_flip: bool = False
    off_axis_flip: bool = False
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class CameraConfigUpdate(BaseModel):
    on_axis_serial: str | None = None
    off_axis_serial: str | None = None
    on_axis_swap_eyes: bool | None = None
    off_axis_swap_eyes: bool | None = None
    on_axis_flip: bool | None = None
    off_axis_flip: bool | None = None


# ── ML Models ───────────────────────────────────────────────────────


class MLModelOut(BaseModel):
    id: UUID
    slug: str
    name: str
    model_type: str
    description: str | None = None
    version: str | None = None
    download_url: str | None = None
    file_size_bytes: int | None = None
    file_path: str | None = None
    status: ModelStatus
    is_active: bool = False
    is_custom: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class MLModelDownloadProgress(BaseModel):
    model_id: UUID
    status: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    percent: float = 0.0
    error: str | None = None

    model_config = {"protected_namespaces": ()}
