"""Calibration router -- proxies capture/compute to camera service, persists to DB."""

import json
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.models import Calibration, User
from app.schemas.schemas import CalibrationCaptureResult, CalibrationOut

router = APIRouter(prefix="/calibration", tags=["calibration"])

CAMERA_URL = settings.CAMERA_SERVICE_URL


# ---------------------------------------------------------------------------
# Proxy endpoints -- forward to camera service
# ---------------------------------------------------------------------------

@router.post("/capture/{camera_name}", response_model=CalibrationCaptureResult)
async def capture_frame(
    camera_name: str,
    current_user: User = Depends(get_current_user),
):
    """Capture a frame, detect ChArUco corners on the camera service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CAMERA_URL}/calibration/capture/{camera_name}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera service error: {exc}")


@router.post("/compute/{camera_name}")
async def compute_calibration(
    camera_name: str,
    save_as_default: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Compute extrinsic calibration and persist it."""
    # Ask camera service to compute
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{CAMERA_URL}/calibration/compute/{camera_name}")
            resp.raise_for_status()
            calibration_data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera service error: {exc}")

    intrinsics = calibration_data["intrinsics"]
    board_config = calibration_data["board_config"]
    quality = calibration_data["quality"]
    disto = intrinsics.get("distortion", [])

    # Write calibration JSON to default location
    calibration_path = None
    if save_as_default:
        default_dir = Path(settings.CALIBRATION_DIR) / "default"
        default_dir.mkdir(parents=True, exist_ok=True)
        cal_file = default_dir / f"{camera_name}.json"
        calibration_data["is_global"] = True
        cal_file.write_text(json.dumps(calibration_data, indent=2))
        calibration_path = str(cal_file)

    # Remove old default for this camera if saving as default
    if save_as_default:
        old = await db.execute(
            select(Calibration).where(
                Calibration.camera_name == camera_name,
                Calibration.is_default == True,
            )
        )
        for row in old.scalars().all():
            await db.delete(row)

    # Persist to DB
    cal = Calibration(
        camera_name=camera_name,
        is_default=save_as_default,
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        k1=disto[0] if len(disto) > 0 else None,
        k2=disto[1] if len(disto) > 1 else None,
        k3=disto[2] if len(disto) > 2 else None,
        p1=disto[3] if len(disto) > 3 else None,
        p2=disto[4] if len(disto) > 4 else None,
        image_width=intrinsics["image_width"],
        image_height=intrinsics["image_height"],
        extrinsic_matrix=calibration_data.get("extrinsic_matrix"),
        board_rows=board_config["rows"],
        board_cols=board_config["cols"],
        square_size_mm=board_config["square_size_mm"],
        marker_size_mm=board_config["marker_size_mm"],
        aruco_dict=board_config["aruco_dict"],
        reprojection_error=quality.get("reprojection_error"),
        num_frames_used=quality.get("num_frames_used"),
        is_global=save_as_default,
        calibration_path=calibration_path,
    )
    db.add(cal)
    await db.commit()
    await db.refresh(cal)

    return calibration_data


@router.post("/reset/{camera_name}")
async def reset_calibration(
    camera_name: str,
    current_user: User = Depends(get_current_user),
):
    """Reset accumulated captures on the camera service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{CAMERA_URL}/calibration/reset/{camera_name}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera service error: {exc}")


@router.get("/status")
async def get_calibration_status(
    current_user: User = Depends(get_current_user),
):
    """Get capture status from camera service and default calibration info from DB."""
    # Get camera-side capture status
    camera_status = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CAMERA_URL}/calibration/status")
            resp.raise_for_status()
            camera_status = resp.json()
    except Exception:
        pass

    return camera_status


@router.get("/defaults", response_model=list[CalibrationOut])
async def list_default_calibrations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all default calibrations."""
    result = await db.execute(
        select(Calibration).where(Calibration.is_default == True)
    )
    return result.scalars().all()


@router.get("/defaults/{camera_name}", response_model=CalibrationOut)
async def get_default_calibration(
    camera_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the default calibration for a specific camera."""
    result = await db.execute(
        select(Calibration).where(
            Calibration.camera_name == camera_name,
            Calibration.is_default == True,
        )
    )
    cal = result.scalar_one_or_none()
    if cal is None:
        raise HTTPException(status_code=404, detail=f"No default calibration for '{camera_name}'")
    return cal


@router.delete("/defaults/{camera_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_default_calibration(
    camera_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete the default calibration for a camera."""
    result = await db.execute(
        select(Calibration).where(
            Calibration.camera_name == camera_name,
            Calibration.is_default == True,
        )
    )
    cal = result.scalar_one_or_none()
    if cal is None:
        raise HTTPException(status_code=404, detail=f"No default calibration for '{camera_name}'")

    # Remove the file on disk
    if cal.calibration_path:
        p = Path(cal.calibration_path)
        if p.exists():
            p.unlink()

    await db.delete(cal)
    await db.commit()
