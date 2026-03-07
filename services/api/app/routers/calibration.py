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

_VALID_CAMERA_NAMES = {"on_axis", "off_axis"}


def _validate_camera_name(camera_name: str) -> None:
    if camera_name not in _VALID_CAMERA_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid camera_name '{camera_name}'. Must be one of: {', '.join(sorted(_VALID_CAMERA_NAMES))}")


# ---------------------------------------------------------------------------
# Proxy endpoints -- forward to camera service
# ---------------------------------------------------------------------------

@router.post("/capture/{camera_name}", response_model=CalibrationCaptureResult)
async def capture_frame(
    camera_name: str,
    current_user: User = Depends(get_current_user),
):
    """Capture a frame, detect ChArUco corners on the camera service."""
    _validate_camera_name(camera_name)
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
    _validate_camera_name(camera_name)
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
    _validate_camera_name(camera_name)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{CAMERA_URL}/calibration/reset/{camera_name}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera service error: {exc}")


@router.post("/capture/stereo")
async def capture_stereo_frame(
    current_user: User = Depends(get_current_user),
):
    """Capture from both cameras simultaneously and detect ChArUco corners."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{CAMERA_URL}/calibration/capture/stereo")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera service error: {exc}")


@router.post("/compute/stereo")
async def compute_stereo_calibration(
    save_as_default: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Compute stereo calibration (per-camera extrinsics + inter-camera transform)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{CAMERA_URL}/calibration/compute/stereo")
            resp.raise_for_status()
            stereo_data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Camera service error: {exc}")

    default_dir = Path(settings.CALIBRATION_DIR) / "default"
    default_dir.mkdir(parents=True, exist_ok=True)

    # Save per-camera calibration files and DB records
    for camera_name in ("on_axis", "off_axis"):
        calibration_data = stereo_data[camera_name]
        intrinsics = calibration_data["intrinsics"]
        board_config = calibration_data["board_config"]
        quality = calibration_data["quality"]
        disto = intrinsics.get("distortion", [])

        calibration_path = None
        if save_as_default:
            calibration_data["is_global"] = True
            cal_file = default_dir / f"{camera_name}.json"
            cal_file.write_text(json.dumps(calibration_data, indent=2))
            calibration_path = str(cal_file)

            # Remove old default for this camera
            old = await db.execute(
                select(Calibration).where(
                    Calibration.camera_name == camera_name,
                    Calibration.is_default == True,
                )
            )
            for row in old.scalars().all():
                await db.delete(row)

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

    # Save stereo calibration JSON
    if save_as_default:
        stereo_file = default_dir / "stereo_calibration.json"
        stereo_file.write_text(json.dumps(stereo_data["stereo"], indent=2))

    await db.commit()
    return stereo_data


@router.post("/reset/stereo")
async def reset_stereo_calibration(
    current_user: User = Depends(get_current_user),
):
    """Reset accumulated captures for both cameras."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{CAMERA_URL}/calibration/reset/stereo")
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
    _validate_camera_name(camera_name)
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
    _validate_camera_name(camera_name)
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
        p = Path(cal.calibration_path).resolve()
        allowed = Path(settings.CALIBRATION_DIR).resolve()
        if p.exists() and str(p).startswith(str(allowed)):
            p.unlink()

    await db.delete(cal)
    await db.commit()
