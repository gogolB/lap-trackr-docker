"""Router for ML model management — download, upload, activate, delete."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
import uuid

import aiofiles
import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.models import MLModel, ModelStatus, User
from app.schemas.schemas import MLModelDownloadProgress, MLModelOut

logger = logging.getLogger("api.models")

router = APIRouter(prefix="/models", tags=["models"])

MODELS_DIR = settings.MODELS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


def _progress_key(model_id: str) -> str:
    return f"model_download:{model_id}"


async def _download_model(model_id: str, url: str, dest: str) -> None:
    """Background task: stream-download a model file, updating Redis progress."""
    r = _get_redis()
    key = _progress_key(model_id)

    from app.core.database import async_session  # local import to avoid cycles

    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(600.0)) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0

                async with aiofiles.open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded / total * 100) if total else 0
                        await r.hset(key, mapping={
                            "status": "downloading",
                            "downloaded_bytes": str(downloaded),
                            "total_bytes": str(total),
                            "percent": f"{pct:.1f}",
                        })

        # Mark as ready in DB
        async with async_session() as db:
            await db.execute(
                update(MLModel)
                .where(MLModel.id == model_id)
                .values(status=ModelStatus.ready, file_path=dest)
            )
            await db.commit()

        await r.hset(key, mapping={
            "status": "ready",
            "downloaded_bytes": str(downloaded),
            "total_bytes": str(total),
            "percent": "100.0",
        })
        await r.expire(key, 300)

    except Exception as exc:
        logger.exception("Download failed for model %s", model_id)
        async with async_session() as db:
            await db.execute(
                update(MLModel)
                .where(MLModel.id == model_id)
                .values(status=ModelStatus.failed)
            )
            await db.commit()

        await r.hset(key, mapping={
            "status": "failed",
            "error": str(exc),
        })
        await r.expire(key, 300)

    finally:
        await r.aclose()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[MLModelOut])
async def list_models(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MLModel).order_by(MLModel.created_at))
    return result.scalars().all()


@router.post("/{model_id}/download", response_model=MLModelOut)
async def download_model(
    model_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    model = await db.get(MLModel, model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    if model.status not in (ModelStatus.available, ModelStatus.failed):
        raise HTTPException(400, f"Model status is '{model.status}', cannot start download")
    if not model.download_url:
        raise HTTPException(400, "No download URL for this model")

    # Compute destination path
    dest_dir = os.path.join(MODELS_DIR, model.model_type, model.slug)
    filename = model.download_url.rsplit("/", 1)[-1]
    dest = os.path.join(dest_dir, filename)

    model.status = ModelStatus.downloading
    await db.commit()
    await db.refresh(model)

    asyncio.create_task(_download_model(str(model.id), model.download_url, dest))

    return model


@router.get("/{model_id}/progress", response_model=MLModelDownloadProgress)
async def download_progress(model_id: uuid.UUID, current_user: User = Depends(get_current_user)):
    r = _get_redis()
    key = _progress_key(str(model_id))
    data = await r.hgetall(key)
    await r.aclose()

    if not data:
        return MLModelDownloadProgress(
            model_id=model_id,
            status="unknown",
        )

    return MLModelDownloadProgress(
        model_id=model_id,
        status=data.get("status", "unknown"),
        downloaded_bytes=int(data.get("downloaded_bytes", 0)),
        total_bytes=int(data.get("total_bytes", 0)),
        percent=float(data.get("percent", 0)),
        error=data.get("error"),
    )


@router.post("/{model_id}/activate", response_model=MLModelOut)
async def activate_model(
    model_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(MLModel).where(MLModel.id == model_id).with_for_update()
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(404, "Model not found")
    if model.status not in (ModelStatus.ready, ModelStatus.active, ModelStatus.custom):
        raise HTTPException(400, f"Model status is '{model.status}', cannot activate")

    # Deactivate all non-custom others → set status back to ready
    await db.execute(
        update(MLModel)
        .where(MLModel.is_active == True, MLModel.id != model_id, MLModel.is_custom == False)  # noqa: E712
        .values(is_active=False, status=ModelStatus.ready)
    )
    # Deactivate custom others → keep status as custom
    await db.execute(
        update(MLModel)
        .where(MLModel.is_active == True, MLModel.id != model_id, MLModel.is_custom == True)  # noqa: E712
        .values(is_active=False)
    )

    model.is_active = True
    model.status = ModelStatus.active if not model.is_custom else ModelStatus.custom
    await db.commit()
    await db.refresh(model)
    return model


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    model = await db.get(MLModel, model_id)
    if not model:
        raise HTTPException(404, "Model not found")

    # Delete file from disk
    if model.file_path and os.path.exists(model.file_path):
        parent = os.path.dirname(model.file_path)
        shutil.rmtree(parent, ignore_errors=True)

    if model.is_custom:
        # Remove custom models from DB entirely
        await db.delete(model)
    else:
        # Reset catalog models to available
        model.status = ModelStatus.available
        model.file_path = None
        model.is_active = False

    await db.commit()


@router.post("/upload", response_model=MLModelOut)
async def upload_model(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(400, "Only .pt model files are accepted")

    if len(file.filename) > 200:
        raise HTTPException(400, "Filename too long (max 200 characters)")

    safe_filename = Path(file.filename).name
    if not safe_filename or ".." in safe_filename:
        raise HTTPException(400, "Invalid filename")

    slug = f"custom-{safe_filename.replace('.pt', '').replace(' ', '-').lower()}"

    # Check for duplicate slug
    existing = await db.execute(select(MLModel).where(MLModel.slug == slug))
    if existing.scalar_one_or_none():
        raise HTTPException(400, f"A model with slug '{slug}' already exists")

    dest_dir = os.path.join(MODELS_DIR, "custom", slug)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, safe_filename)

    # Validate resolved path stays within dest_dir
    if not os.path.realpath(dest).startswith(os.path.realpath(dest_dir)):
        raise HTTPException(400, "Invalid filename")

    content = await file.read()
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)

    model = MLModel(
        slug=slug,
        name=safe_filename.replace(".pt", ""),
        model_type="yolo",
        description=f"Custom uploaded YOLO model: {safe_filename}",
        file_path=dest,
        file_size_bytes=len(content),
        status=ModelStatus.custom,
        is_custom=True,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model
