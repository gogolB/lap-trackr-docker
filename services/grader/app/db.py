"""Database operations for the grading worker.

Uses SQLAlchemy Core (synchronous) with psycopg2 so the single-threaded
worker can talk to PostgreSQL without an async event loop.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine

from app.config import DATA_DIR, DATABASE_URL

logger = logging.getLogger("grader.db")

# ---------------------------------------------------------------------------
# Engine (lazy singleton)
# ---------------------------------------------------------------------------

_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=3,
        )
        logger.info("Database engine created for %s", DATABASE_URL)
    return _engine


# ---------------------------------------------------------------------------
# Table definitions (must match the API service's schema)
# ---------------------------------------------------------------------------

metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("status", String(32), nullable=False),
    # Other columns exist but are not needed by the worker.
)

grading_results_table = Table(
    "grading_results",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column(
        "session_id",
        UUID(as_uuid=True),
        ForeignKey("sessions.id"),
        nullable=False,
        unique=True,
    ),
    Column("workspace_volume", Float, nullable=True),
    Column("avg_speed", Float, nullable=True),
    Column("max_jerk", Float, nullable=True),
    Column("path_length", Float, nullable=True),
    Column("economy_of_motion", Float, nullable=True),
    Column("total_time", Float, nullable=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    Column("error", Text, nullable=True),
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _get_session_dir(session_id: str) -> str | None:
    """Look up the on_axis_path for a session and return its parent directory."""
    engine = _get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT on_axis_path FROM sessions WHERE id = :sid"),
            {"sid": session_id},
        ).first()
    if row and row[0]:
        return os.path.dirname(row[0])
    return None


def update_session_status(session_id: str, status: str) -> None:
    """Set the ``status`` column on the sessions table."""

    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            sessions_table.update()
            .where(sessions_table.c.id == session_id)
            .values(status=status)
        )
    logger.info("Session %s: status -> %s", session_id, status)


def save_results(session_id: str, results: dict[str, Any]) -> None:
    """Persist grading results to the database and as JSON files.

    * Upserts a row in ``grading_results``.
    * Writes ``metrics.json`` and ``poses.json`` under the session's
      results directory at ``<DATA_DIR>/results/<session_id>/``.
    """

    metrics: dict = results.get("metrics", {})
    poses: list = results.get("poses", [])
    now = datetime.now(timezone.utc)

    engine = _get_engine()
    with engine.begin() as conn:
        # Check whether a row already exists for this session.
        existing = conn.execute(
            grading_results_table.select().where(
                grading_results_table.c.session_id == session_id
            )
        ).first()

        values = {
            "session_id": session_id,
            "workspace_volume": metrics.get("workspace_volume"),
            "avg_speed": metrics.get("avg_speed"),
            "max_jerk": metrics.get("max_jerk"),
            "path_length": metrics.get("path_length"),
            "economy_of_motion": metrics.get("economy_of_motion"),
            "total_time": metrics.get("total_time"),
            "completed_at": now,
            "error": None,
        }

        if existing is not None:
            conn.execute(
                grading_results_table.update()
                .where(grading_results_table.c.session_id == session_id)
                .values(**values)
            )
        else:
            conn.execute(
                grading_results_table.insert().values(
                    id=uuid.uuid4(),
                    **values,
                )
            )

    # --- Write JSON files ---------------------------------------------------
    # Derive the results dir from the session's on_axis_path.
    # The on_axis_path looks like /data/users/<user_id>/<timestamp>/on_axis.svo2
    # We want /data/users/<user_id>/<timestamp>/results/
    session_dir = _get_session_dir(session_id)
    results_dir = os.path.join(session_dir, "results") if session_dir else os.path.join(DATA_DIR, "results", session_id)
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote %s", metrics_path)

    poses_path = os.path.join(results_dir, "poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses, f, indent=2)
    logger.info("Wrote %s", poses_path)


def save_error(session_id: str, error_message: str) -> None:
    """Record an error against a grading session."""

    now = datetime.now(timezone.utc)
    engine = _get_engine()

    with engine.begin() as conn:
        existing = conn.execute(
            grading_results_table.select().where(
                grading_results_table.c.session_id == session_id
            )
        ).first()

        if existing is not None:
            conn.execute(
                grading_results_table.update()
                .where(grading_results_table.c.session_id == session_id)
                .values(error=error_message, completed_at=now)
            )
        else:
            conn.execute(
                grading_results_table.insert().values(
                    id=uuid.uuid4(),
                    session_id=session_id,
                    error=error_message,
                    completed_at=now,
                )
            )

    logger.info("Saved error for session %s", session_id)
