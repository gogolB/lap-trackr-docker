import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SessionStatus(str, enum.Enum):
    recording = "recording"
    completed = "completed"
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
