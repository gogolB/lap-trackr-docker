"""Export worker -- polls Redis for export jobs and converts SVO2 to MP4+NPZ."""

from __future__ import annotations

import json
import logging
import sys
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


def main() -> None:
    """Block on the Redis queue and process export jobs forever."""

    logger.info("Connecting to Redis at %s", REDIS_URL)
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection OK -- waiting for jobs on '%s'", QUEUE_KEY)

    while True:
        _, raw = redis_client.brpop(QUEUE_KEY, timeout=0)
        job: dict = json.loads(raw)
        session_id: str = job["session_id"]
        logger.info("Received export job for session %s", session_id)

        try:
            from app.exporter import export_svo2

            for key in ("on_axis_path", "off_axis_path"):
                svo_path = job.get(key)
                if svo_path and Path(svo_path).exists():
                    logger.info("Exporting %s: %s", key, svo_path)
                    result = export_svo2(svo_path)
                    logger.info(
                        "  Exported %d frames → %s",
                        result["frame_count"],
                        result["mp4_path"],
                    )

            update_session_status(session_id, "completed")
            logger.info("Session %s: export complete, status -> completed", session_id)

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
