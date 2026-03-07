"""Tip initialization endpoints -- serves sample frames and manages tip positions."""

import json
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.models import Session, SessionStatus, User

router = APIRouter(prefix="/sessions", tags=["tip-init"])
logger = logging.getLogger("api.tip_init")


class TipInitUpdate(BaseModel):
    tips: dict[str, list[dict]]


@router.get("/{session_id}/tip-init")
async def get_tip_init(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return tip detection data and sample frame filenames."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.on_axis_path:
        raise HTTPException(status_code=404, detail="No session files found")

    session_dir = Path(session.on_axis_path).parent

    # Load tip detections
    det_path = session_dir / "tip_detections.json"
    detections: dict = {}
    if det_path.exists():
        detections = json.loads(det_path.read_text())

    # Find sample frame filenames
    sample_frames = sorted(
        f.name for f in session_dir.glob("*_sample_*.jpg")
    )

    return {
        "detections": detections,
        "sample_frames": sample_frames,
    }


@router.get("/{session_id}/sample-frame/{filename}")
async def get_sample_frame(
    session_id: UUID,
    filename: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    token: str | None = None,
):
    """Serve a sample frame JPEG."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.on_axis_path:
        raise HTTPException(status_code=404, detail="No session files found")

    # Validate filename to prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    session_dir = Path(session.on_axis_path).parent
    frame_path = session_dir / filename

    if not frame_path.exists() or not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Sample frame not found")

    return FileResponse(str(frame_path), media_type="image/jpeg")


@router.put("/{session_id}/tip-init")
async def update_tip_init(
    session_id: UUID,
    body: TipInitUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save user-confirmed tip positions and advance status to completed."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.awaiting_init:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session is not awaiting initialization (current status: {session.status.value})",
        )

    if not session.on_axis_path:
        raise HTTPException(status_code=404, detail="No session files found")

    session_dir = Path(session.on_axis_path).parent

    # Save tip_init.json
    init_path = session_dir / "tip_init.json"
    init_path.write_text(json.dumps(body.tips, indent=2))
    logger.info("Saved tip init to %s", init_path)

    # Advance status
    session.status = SessionStatus.completed
    await db.commit()
    await db.refresh(session)

    return {"status": "completed"}
