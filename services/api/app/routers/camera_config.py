import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.models import CameraConfig, User
from app.schemas.schemas import CameraConfigOut, CameraConfigUpdate

router = APIRouter(prefix="/camera-config", tags=["camera-config"])
logger = logging.getLogger("api.camera_config")


async def _get_or_create_config(db: AsyncSession) -> CameraConfig:
    """Get the single camera config row, creating it with defaults if needed."""
    result = await db.execute(select(CameraConfig).where(CameraConfig.id == 1))
    config = result.scalar_one_or_none()
    if config is None:
        config = CameraConfig(
            id=1,
            on_axis_serial="",
            off_axis_serial="",
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


def _build_camera_payload(config: CameraConfig) -> dict:
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


async def push_saved_camera_config(db: AsyncSession) -> dict:
    """Apply the persisted camera config to the running camera service."""
    config = await _get_or_create_config(db)
    payload = _build_camera_payload(config)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.CAMERA_SERVICE_URL}/config/apply",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.error("Failed to apply camera config: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to apply config to camera service: {exc}",
        )


@router.get("/", response_model=CameraConfigOut)
async def get_camera_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    config = await _get_or_create_config(db)
    return config


@router.put("/", response_model=CameraConfigOut)
async def update_camera_config(
    body: CameraConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    config = await _get_or_create_config(db)
    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(config, field, value)
    await db.commit()
    await db.refresh(config)
    return config


@router.post("/apply")
async def apply_camera_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Push the current camera config to the running camera service."""
    return await push_saved_camera_config(db)
