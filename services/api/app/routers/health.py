import time
from typing import Any

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.database import async_session

router = APIRouter(prefix="/health", tags=["health"])


async def _check_db() -> dict[str, Any]:
    """Ping PostgreSQL and return version info."""
    start = time.monotonic()
    try:
        async with async_session() as db:
            row = await db.execute(text("SELECT version()"))
            version = row.scalar()
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {"status": "ok", "latency_ms": latency_ms, "version": version}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_redis() -> dict[str, Any]:
    """Ping Redis and report memory usage."""
    start = time.monotonic()
    try:
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        info = await r.info("memory")
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        await r.aclose()
        return {
            "status": "ok",
            "latency_ms": latency_ms,
            "used_memory_human": info.get("used_memory_human", "unknown"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_camera() -> dict[str, Any]:
    """Call the camera service /status endpoint."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.CAMERA_SERVICE_URL}/status")
            resp.raise_for_status()
            data = resp.json()
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {"status": "ok", "latency_ms": latency_ms, **data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_grader() -> dict[str, Any]:
    """Check if the grader is alive by looking at Redis queue length."""
    try:
        r = aioredis.from_url(settings.REDIS_URL)
        queue_len = await r.llen("grading_jobs")
        await r.aclose()
        return {"status": "ok", "pending_jobs": queue_len}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/system")
async def system_health() -> dict[str, Any]:
    """Aggregate health of all services."""
    import asyncio

    db_task = asyncio.create_task(_check_db())
    redis_task = asyncio.create_task(_check_redis())
    camera_task = asyncio.create_task(_check_camera())
    grader_task = asyncio.create_task(_check_grader())

    db, redis_info, camera, grader = await asyncio.gather(
        db_task, redis_task, camera_task, grader_task
    )

    services = {
        "api": {"status": "ok"},
        "database": db,
        "redis": redis_info,
        "camera": camera,
        "grader": grader,
    }

    all_ok = all(s.get("status") == "ok" for s in services.values())

    return {
        "overall": "healthy" if all_ok else "degraded",
        "services": services,
    }
