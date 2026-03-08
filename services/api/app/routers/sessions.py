import asyncio
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx
import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.models import CameraConfig, Session, SessionStatus, User
from app.schemas.schemas import SessionDetailOut, SessionOut, SessionStartRequest

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger("api.sessions")
JOB_PROGRESS_KEY_PREFIX = "job_progress:"
EXPORT_CANCEL_KEY_PREFIX = "export_cancel:"
STAGE_FIELD_PREFIX = "stage__"


def _session_dir(session: Session) -> Path | None:
    if not session.on_axis_path:
        return None
    return Path(session.on_axis_path).parent


def _serialize_camera_config(config: CameraConfig | None) -> dict:
    if config is None:
        return {
            "on_axis_serial": "",
            "off_axis_serial": "",
            "on_axis_swap_eyes": False,
            "off_axis_swap_eyes": False,
            "on_axis_rotation": 0,
            "off_axis_rotation": 0,
            "on_axis_flip_h": False,
            "on_axis_flip_v": False,
            "off_axis_flip_h": False,
            "off_axis_flip_v": False,
        }

    return {
        "on_axis_serial": config.on_axis_serial,
        "off_axis_serial": config.off_axis_serial,
        "on_axis_swap_eyes": config.on_axis_swap_eyes,
        "off_axis_swap_eyes": config.off_axis_swap_eyes,
        "on_axis_rotation": config.on_axis_rotation,
        "off_axis_rotation": config.off_axis_rotation,
        "on_axis_flip_h": config.on_axis_flip_h,
        "on_axis_flip_v": config.on_axis_flip_v,
        "off_axis_flip_h": config.off_axis_flip_h,
        "off_axis_flip_v": config.off_axis_flip_v,
    }


def _build_grading_job(session: Session) -> str:
    calibration_path = None
    stereo_calibration_path = None
    tip_init_path = None

    session_dir = _session_dir(session)
    if session_dir:
        calib_file = session_dir / "calibration_on_axis.json"
        if calib_file.exists():
            calibration_path = str(calib_file)
        stereo_file = session_dir / "stereo_calibration.json"
        if stereo_file.exists():
            stereo_calibration_path = str(stereo_file)
        tip_file = session_dir / "tip_init.json"
        if tip_file.exists():
            tip_init_path = str(tip_file)

    return json.dumps(
        {
            "session_id": str(session.id),
            "on_axis_path": session.on_axis_path,
            "off_axis_path": session.off_axis_path,
            "calibration_path": calibration_path,
            "stereo_calibration_path": stereo_calibration_path,
            "tip_init_path": tip_init_path,
        }
    )


def _derive_post_export_status(session: Session) -> SessionStatus:
    if session.status == SessionStatus.graded:
        return SessionStatus.graded

    session_dir = _session_dir(session)
    if session_dir and (session_dir / "tip_init.json").exists():
        return SessionStatus.completed

    return SessionStatus.awaiting_init


def _build_export_job(
    session: Session,
    post_export_status: SessionStatus | None = None,
) -> str:
    target_status = post_export_status or _derive_post_export_status(session)
    return json.dumps(
        {
            "session_id": str(session.id),
            "on_axis_path": session.on_axis_path,
            "off_axis_path": session.off_axis_path,
            "post_export_status": target_status.value,
        }
    )


@router.get("/", response_model=list[SessionOut])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    result = await db.execute(
        select(Session)
        .where(Session.user_id == current_user.id)
        .order_by(Session.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/{session_id}", response_model=SessionDetailOut)
async def get_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session)
        .options(selectinload(Session.grading_result))
        .where(Session.id == session_id, Session.user_id == current_user.id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return session


@router.post("/start", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def start_session(
    body: SessionStartRequest = SessionStartRequest(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = str(
        Path(settings.DATA_DIR) / "users" / str(current_user.id) / timestamp
    )

    # Generate default name if not provided
    session_name = body.name.strip() if body.name else ""
    if not session_name:
        session_name = f"Session {now.strftime('%b %d, %Y %I:%M %p')}"

    session = Session(
        user_id=current_user.id,
        name=session_name,
        started_at=now,
        status=SessionStatus.recording,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Create the session directory on disk
    await asyncio.to_thread(Path(session_dir).mkdir, parents=True, exist_ok=True)

    # Copy global default calibrations into session dir
    for cam in ("on_axis", "off_axis"):
        default_path = Path(settings.CALIBRATION_DIR) / "default" / f"{cam}.json"
        if default_path.exists():
            dest = Path(session_dir) / f"calibration_{cam}.json"
            await asyncio.to_thread(shutil.copy2, str(default_path), str(dest))
            logger.info("Copied default calibration for %s into session dir", cam)

    # Copy stereo calibration if available
    stereo_path = Path(settings.CALIBRATION_DIR) / "default" / "stereo_calibration.json"
    if stereo_path.exists():
        dest = Path(session_dir) / "stereo_calibration.json"
        await asyncio.to_thread(shutil.copy2, str(stereo_path), str(dest))
        logger.info("Copied stereo calibration into session dir")

    # Write session_metadata.json
    config_result = await db.execute(select(CameraConfig).where(CameraConfig.id == 1))
    active_camera_config = config_result.scalar_one_or_none()
    metadata = {
        "session_id": str(session.id),
        "user_id": str(current_user.id),
        "started_at": now.isoformat(),
        "cameras": {},
        "camera_config": _serialize_camera_config(active_camera_config),
        "zed_sdk_version": "unknown",
        "software_version": "1.0.0",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            info_resp = await client.get(f"{settings.CAMERA_SERVICE_URL}/camera-info")
            if info_resp.status_code == 200:
                cam_info = info_resp.json()
                metadata["cameras"] = cam_info.get("cameras", {})
                metadata["zed_sdk_version"] = cam_info.get("zed_sdk_version", "unknown")
    except Exception as exc:
        logger.warning("Could not fetch camera info for metadata: %s", exc)

    metadata_json = json.dumps(metadata, indent=2)
    await asyncio.to_thread(
        (Path(session_dir) / "session_metadata.json").write_text, metadata_json
    )

    # Call the camera service to start recording
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.CAMERA_SERVICE_URL}/record/start",
                json={"session_dir": session_dir},
            )
            resp.raise_for_status()
            camera_data = resp.json()

        session.on_axis_path = camera_data.get("on_axis_path")
        session.off_axis_path = camera_data.get("off_axis_path")
        await db.commit()
        await db.refresh(session)

    except Exception as exc:
        logger.exception("Camera start failed")
        session.status = SessionStatus.failed
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Camera start failed",
        )

    return session


@router.post("/{session_id}/stop", response_model=SessionOut)
async def stop_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        ).with_for_update()
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    if session.status != SessionStatus.recording:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session is not recording (current status: {session.status.value})",
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{settings.CAMERA_SERVICE_URL}/record/stop")
            resp.raise_for_status()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to contact camera service to stop recording",
        )

    session.stopped_at = datetime.now(timezone.utc)

    # Queue export job (SVO2 → MP4 + NPZ)
    export_job = _build_export_job(session, SessionStatus.awaiting_init)
    r = redis.from_url(settings.REDIS_URL)
    try:
        await r.lpush("export_jobs", export_job)
        session.status = SessionStatus.exporting
    except Exception:
        logger.warning("Failed to enqueue export job, setting status to export_failed")
        session.status = SessionStatus.export_failed
    finally:
        await r.aclose()

    await db.commit()
    await db.refresh(session)
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        ).with_for_update()
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    if session.status in (SessionStatus.recording, SessionStatus.exporting, SessionStatus.grading):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete session while status is '{session.status.value}'",
        )

    # Remove session files from disk
    if session.on_axis_path:
        session_dir = str(Path(session.on_axis_path).parent)
        if Path(session_dir).exists():
            shutil.rmtree(session_dir, ignore_errors=True)

    await db.delete(session)
    await db.commit()


@router.post("/{session_id}/grade", response_model=SessionOut)
async def grade_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        ).with_for_update()
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    if session.status != SessionStatus.completed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session must be completed before grading (current status: {session.status.value})",
        )

    job_payload = _build_grading_job(session)

    r = redis.from_url(settings.REDIS_URL)
    try:
        await r.lpush("grading_jobs", job_payload)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to enqueue grading job",
        )
    finally:
        await r.aclose()

    session.status = SessionStatus.grading
    await db.commit()
    await db.refresh(session)
    return session


@router.post("/{session_id}/re-export", response_model=SessionOut)
async def re_export_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-queue export for an existing session, cancelling an active export if needed."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        ).with_for_update()
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    if session.status in (SessionStatus.recording, SessionStatus.grading):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot re-export while status is '{session.status.value}'",
        )
    if not session.on_axis_path and not session.off_axis_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session does not have any recorded camera files",
        )

    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        progress_key = f"{JOB_PROGRESS_KEY_PREFIX}{session.id}"
        cancel_key = f"{EXPORT_CANCEL_KEY_PREFIX}{session.id}"
        if session.status == SessionStatus.exporting:
            await r.set(cancel_key, "1", ex=3600)
        else:
            await r.delete(cancel_key)
        await r.delete(progress_key)
        await r.lpush("export_jobs", _build_export_job(session))
        session.status = SessionStatus.exporting
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to enqueue export job",
        )
    finally:
        await r.aclose()

    await db.commit()
    await db.refresh(session)
    return session


@router.post("/{session_id}/retry", response_model=SessionOut)
async def retry_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retry a failed or export_failed session by re-queuing the appropriate job."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id, Session.user_id == current_user.id
        ).with_for_update()
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    if session.status not in (SessionStatus.failed, SessionStatus.export_failed):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Only failed or export_failed sessions can be retried (current status: {session.status.value})",
        )

    r = redis.from_url(settings.REDIS_URL)
    try:
        if session.status == SessionStatus.export_failed:
            await r.lpush("export_jobs", _build_export_job(session))
            session.status = SessionStatus.exporting
        else:
            await r.lpush("grading_jobs", _build_grading_job(session))
            session.status = SessionStatus.grading
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to enqueue retry job",
        )
    finally:
        await r.aclose()

    await db.commit()
    await db.refresh(session)
    return session


@router.get("/{session_id}/progress")
async def get_session_progress(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return live progress for an in-flight export or grading job."""
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

    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        data = await r.hgetall(f"job_progress:{session_id}")
    finally:
        await r.aclose()

    if not data:
        return {
            "session_id": str(session_id),
            "status": session.status.value,
            "stage": None,
            "current": 0,
            "total": 0,
            "percent": 0,
            "detail": "",
            "updated_at": None,
            "stage_started_at": None,
            "stages": {},
        }

    stages: dict[str, dict] = {}
    for key, value in data.items():
        if not key.startswith(STAGE_FIELD_PREFIX):
            continue
        stage_name = key[len(STAGE_FIELD_PREFIX):]
        try:
            stages[stage_name] = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    return {
        "session_id": str(session_id),
        "status": session.status.value,
        "stage": data.get("stage", ""),
        "current": int(data.get("current", 0)),
        "total": int(data.get("total", 0)),
        "percent": float(data.get("percent", 0)),
        "detail": data.get("detail", ""),
        "updated_at": float(data["updated_at"]) if data.get("updated_at") else None,
        "stage_started_at": float(data["stage_started_at"])
        if data.get("stage_started_at")
        else None,
        "stages": stages,
    }


@router.get("/{session_id}/download")
async def download_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream a ZIP of the session directory (MP4, NPZ, calibration, metadata, results)."""
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

    if not session.on_axis_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No session files found"
        )

    session_dir = Path(session.on_axis_path).parent.resolve()
    allowed_prefix = Path(settings.DATA_DIR).resolve() / "users" / str(current_user.id)
    if not str(session_dir).startswith(str(allowed_prefix)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    if not session_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session directory not found on disk",
        )

    # Build ZIP on disk to avoid OOM on large sessions
    tmp_path = await asyncio.to_thread(_build_session_zip, session_dir)

    timestamp = session.started_at.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"session_{timestamp}.zip"

    def stream_and_cleanup():
        try:
            with open(tmp_path, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(
        stream_and_cleanup(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _build_session_zip(session_dir: Path) -> str:
    """Build ZIP on disk, return temp file path."""
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    try:
        with os.fdopen(fd, "wb") as f:
            with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(session_dir.rglob("*")):
                    if file_path.is_file() and not file_path.is_symlink():
                        arcname = file_path.relative_to(session_dir)
                        zf.write(file_path, arcname)
    except Exception:
        os.unlink(tmp_path)
        raise
    return tmp_path
