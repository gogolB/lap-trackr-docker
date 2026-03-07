"""Export worker -- polls Redis for export jobs and converts SVO2 to MP4+NPZ."""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import traceback
from pathlib import Path

import redis

from app.config import REDIS_URL
from app.db import update_session_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("grader.export_worker")

QUEUE_KEY = "export_jobs"

PROGRESS_KEY_PREFIX = "job_progress:"
PROGRESS_TTL = 3600  # 1 hour

_shutdown = False


def _publish_progress(
    redis_client: redis.Redis,
    session_id: str,
    stage: str,
    current: int,
    total: int,
    detail: str = "",
) -> None:
    """Write job progress to a Redis hash for the API to read."""
    key = f"{PROGRESS_KEY_PREFIX}{session_id}"
    now = time.time()
    pct = round(current / total * 100, 1) if total > 0 else 0
    redis_client.hset(key, mapping={
        "stage": stage,
        "current": str(current),
        "total": str(total),
        "percent": str(pct),
        "detail": detail,
        "updated_at": str(now),
    })
    redis_client.expire(key, PROGRESS_TTL)


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    logger.info("Received signal %s, will exit after current job", signum)


def main() -> None:
    """Block on the Redis queue and process export jobs forever."""
    global _shutdown

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Connecting to Redis at %s", REDIS_URL)
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection OK -- waiting for jobs on '%s'", QUEUE_KEY)

    while not _shutdown:
        result = redis_client.brpop(QUEUE_KEY, timeout=5)
        if result is None:
            continue
        _, raw = result
        try:
            job: dict = json.loads(raw)
            session_id: str = job["session_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Malformed export job message, skipping: %s (raw=%r)", exc, raw[:200] if raw else raw)
            continue
        logger.info("Received export job for session %s", session_id)

        try:
            from app.exporter import export_svo2
            from app.color_detector import detect_tips

            all_sample_paths: list[str] = []
            session_dir: str | None = None

            # Count how many cameras to export for progress tracking
            cam_keys = [k for k in ("on_axis_path", "off_axis_path") if job.get(k) and Path(job[k]).exists()]
            cam_total = len(cam_keys)

            for cam_idx, key in enumerate(cam_keys):
                svo_path = job[key]
                cam_label = key.replace("_path", "")

                def on_export_progress(frame: int, total: int, _ci=cam_idx, _ct=cam_total, _cl=cam_label) -> None:
                    _publish_progress(
                        redis_client, session_id,
                        stage=f"exporting ({_cl.replace('_', ' ')} {_ci + 1}/{_ct})",
                        current=frame, total=total,
                        detail=f"{_cl} camera",
                    )

                logger.info("Exporting %s: %s", key, svo_path)
                _publish_progress(redis_client, session_id, stage=f"exporting ({cam_label.replace('_', ' ')})", current=0, total=0, detail="starting")
                result = export_svo2(svo_path, on_progress=on_export_progress)
                logger.info(
                    "  Exported %d frames → %s",
                    result["frame_count"],
                    result["mp4_path"],
                )
                all_sample_paths.extend(result.get("sample_paths", []))
                if session_dir is None:
                    session_dir = str(Path(svo_path).parent)

            # Run color detection on sample frames
            _publish_progress(redis_client, session_id, stage="detecting tips", current=0, total=len(all_sample_paths))
            tip_detections: dict[str, list[dict]] = {}
            if session_dir and all_sample_paths:
                import cv2
                for sample_path in all_sample_paths:
                    if Path(sample_path).exists():
                        bgr = cv2.imread(sample_path)
                        if bgr is not None:
                            tips = detect_tips(bgr)
                            filename = Path(sample_path).name
                            tip_detections[filename] = tips

                # Save tip detections
                det_path = Path(session_dir) / "tip_detections.json"
                det_path.write_text(json.dumps(tip_detections, indent=2))
                logger.info("Saved tip detections to %s", det_path)

            _publish_progress(redis_client, session_id, stage="complete", current=1, total=1)
            update_session_status(session_id, "awaiting_init")
            logger.info("Session %s: export complete, status -> awaiting_init", session_id)

        except Exception:
            tb = traceback.format_exc()
            logger.error("Session %s export FAILED:\n%s", session_id, tb)
            try:
                update_session_status(session_id, "export_failed")
            except Exception:
                logger.error(
                    "Could not persist export failure for %s:\n%s",
                    session_id,
                    traceback.format_exc(),
                )


if __name__ == "__main__":
    main()
