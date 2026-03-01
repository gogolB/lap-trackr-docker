import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.models import GradingResult, Session, User
from app.schemas.schemas import GradingResultOut

router = APIRouter(prefix="/results", tags=["results"])


async def _get_session_for_user(
    session_id: UUID, current_user: User, db: AsyncSession
) -> Session:
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return session


def _results_dir(session: Session) -> Path:
    if session.on_axis_path:
        return Path(session.on_axis_path).parent / "results"
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No recording path associated with this session",
    )


@router.get("/{session_id}", response_model=GradingResultOut)
async def get_grading_result(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_for_user(session_id, current_user, db)

    result = await db.execute(
        select(GradingResult).where(GradingResult.session_id == session.id)
    )
    grading_result = result.scalar_one_or_none()
    if grading_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Grading result not found for this session",
        )
    return grading_result


@router.get("/{session_id}/metrics")
async def get_metrics(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_for_user(session_id, current_user, db)
    results_dir = _results_dir(session)
    metrics_path = results_dir / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="metrics.json not found",
        )

    with open(metrics_path, "r") as f:
        return json.load(f)


@router.get("/{session_id}/poses")
async def get_poses(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_for_user(session_id, current_user, db)
    results_dir = _results_dir(session)
    poses_path = results_dir / "poses.json"

    if not poses_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="poses.json not found",
        )

    with open(poses_path, "r") as f:
        return json.load(f)
