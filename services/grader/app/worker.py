"""Main worker loop -- polls Redis for grading jobs and runs the pipeline."""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import traceback

import redis

from app.camera_transform import load_camera_config_from_session_dir
from app.config import REDIS_URL
from app.db import get_camera_config, save_error, save_results, update_session_status
from app.pipeline import grade, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("grader.worker")

QUEUE_KEY = "grading_jobs"

PROGRESS_KEY_PREFIX = "job_progress:"
STAGE_FIELD_PREFIX = "stage__"
PROGRESS_TTL = 3600


def _publish_progress(
    redis_client: redis.Redis,
    session_id: str,
    stage: str,
    current: int,
    total: int,
    detail: str = "",
) -> None:
    key = f"{PROGRESS_KEY_PREFIX}{session_id}"
    pct = round(current / total * 100, 1) if total > 0 else 0
    now = time.time()
    stage_field = f"{STAGE_FIELD_PREFIX}{stage}"
    stage_status = "completed" if total > 0 and current >= total else "running"
    raw_stage_data = redis_client.hget(key, stage_field)
    stage_started_at = now
    if raw_stage_data:
        try:
            parsed = json.loads(raw_stage_data)
            if current > 0 and parsed.get("started_at") is not None:
                stage_started_at = float(parsed["started_at"])
        except (TypeError, ValueError, json.JSONDecodeError):
            stage_started_at = now

    stage_payload = json.dumps(
        {
            "stage": stage,
            "current": current,
            "total": total,
            "percent": pct,
            "detail": detail,
            "status": stage_status,
            "updated_at": now,
            "started_at": stage_started_at,
        }
    )
    redis_client.hset(key, mapping={
        "stage": stage,
        "current": str(current),
        "total": str(total),
        "percent": str(pct),
        "detail": detail,
        "updated_at": str(now),
        "stage_started_at": str(stage_started_at),
        stage_field: stage_payload,
    })
    redis_client.expire(key, PROGRESS_TTL)


_shutdown = False


def _resolve_camera_config(job: dict) -> dict | None:
    from pathlib import Path

    if isinstance(job.get("camera_config"), dict):
        return job["camera_config"]
    path = job.get("on_axis_path") or job.get("off_axis_path")
    if path:
        config = load_camera_config_from_session_dir(Path(path).parent)
        if config is not None:
            return config
    return get_camera_config()


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    logger.info("Received signal %s, will exit after current job", signum)


def main() -> None:
    """Block on the Redis queue and process grading jobs forever."""
    global _shutdown

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Connecting to Redis at %s", REDIS_URL)
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection OK -- waiting for jobs on '%s'", QUEUE_KEY)

    while not _shutdown:
        # BRPOP with timeout allows checking shutdown flag between polls
        result = redis_client.brpop(QUEUE_KEY, timeout=5)
        if result is None:
            continue
        _, raw = result
        try:
            job: dict = json.loads(raw)
            session_id: str = job["session_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Malformed job message, skipping: %s (raw=%r)", exc, raw[:200] if raw else raw)
            continue

        if not job.get("on_axis_path") and not job.get("off_axis_path"):
            logger.error("Job for session %s missing both on_axis_path and off_axis_path, skipping", session_id)
            continue

        logger.info("Received job for session %s", session_id)

        try:
            update_session_status(session_id, "grading")
            logger.info("Session %s: status -> grading", session_id)
            job["camera_config"] = _resolve_camera_config(job)

            def on_progress(stage: str, current: int, total: int, detail: str = "") -> None:
                _publish_progress(redis_client, session_id, stage, current, total, detail)

            results = grade(job, on_progress=on_progress)

            _publish_progress(redis_client, session_id, "complete", 1, 1)
            save_results(session_id, results)
            update_session_status(session_id, "graded")
            logger.info("Session %s: grading complete", session_id)

        except Exception:
            tb = traceback.format_exc()
            logger.error("Session %s FAILED:\n%s", session_id, tb)
            try:
                update_session_status(session_id, "failed")
                save_error(session_id, tb)
            except Exception:
                logger.error(
                    "Could not persist failure for %s:\n%s",
                    session_id,
                    traceback.format_exc(),
                )


if __name__ == "__main__":
    main()
