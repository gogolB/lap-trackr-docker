import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ModelStatus(str, enum.Enum):
    available = "available"
    downloading = "downloading"
    ready = "ready"
    active = "active"
    custom = "custom"
    failed = "failed"


class SessionStatus(str, enum.Enum):
    recording = "recording"
    completed = "completed"
    exporting = "exporting"
    export_failed = "export_failed"
    awaiting_init = "awaiting_init"
    grading = "grading"
    graded = "graded"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    sessions: Mapped[list["Session"]] = relationship(
        "Session", back_populates="user", cascade="all, delete-orphan"
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="Untitled Session")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    stopped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.recording, nullable=False
    )
    on_axis_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    off_axis_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped["User"] = relationship("User", back_populates="sessions")
    grading_result: Mapped["GradingResult | None"] = relationship(
        "GradingResult", back_populates="session", uselist=False, cascade="all, delete-orphan"
    )


class GradingResult(Base):
    __tablename__ = "grading_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    workspace_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_jerk: Mapped[float | None] = mapped_column(Float, nullable=True)
    path_length: Mapped[float | None] = mapped_column(Float, nullable=True)
    economy_of_motion: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="grading_result")


class Calibration(Base):
    __tablename__ = "calibrations"
    __table_args__ = (
        # Only one default calibration per camera_name
        Index(
            "ix_calibrations_default_camera",
            "camera_name",
            unique=True,
            postgresql_where=text("is_default = true"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    camera_name: Mapped[str] = mapped_column(String(32), nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True
    )
    fx: Mapped[float] = mapped_column(Float, nullable=False)
    fy: Mapped[float] = mapped_column(Float, nullable=False)
    cx: Mapped[float] = mapped_column(Float, nullable=False)
    cy: Mapped[float] = mapped_column(Float, nullable=False)
    k1: Mapped[float | None] = mapped_column(Float, nullable=True)
    k2: Mapped[float | None] = mapped_column(Float, nullable=True)
    k3: Mapped[float | None] = mapped_column(Float, nullable=True)
    p1: Mapped[float | None] = mapped_column(Float, nullable=True)
    p2: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    extrinsic_matrix: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    board_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    board_cols: Mapped[int] = mapped_column(Integer, nullable=False)
    square_size_mm: Mapped[float] = mapped_column(Float, nullable=False)
    marker_size_mm: Mapped[float] = mapped_column(Float, nullable=False)
    aruco_dict: Mapped[str] = mapped_column(String(32), nullable=False)
    reprojection_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_frames_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_global: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    calibration_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    download_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus), default=ModelStatus.available, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class CameraConfig(Base):
    __tablename__ = "camera_config"
    __table_args__ = (
        CheckConstraint("id = 1", name="single_row_camera_config"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    on_axis_serial: Mapped[str] = mapped_column(String(32), nullable=False)
    off_axis_serial: Mapped[str] = mapped_column(String(32), nullable=False)
    on_axis_swap_eyes: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    off_axis_swap_eyes: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    on_axis_flip: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    off_axis_flip: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=True,
    )
