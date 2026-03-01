"""Main worker loop -- polls Redis for grading jobs and runs the pipeline."""

from __future__ import annotations

import json
import logging
import sys
import traceback

import redis

from app.config import REDIS_URL
from app.db import save_error, save_results, update_session_status
from app.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("grader.worker")

QUEUE_KEY = "grading_jobs"


def main() -> None:
    """Block on the Redis queue and process grading jobs forever."""

    logger.info("Connecting to Redis at %s", REDIS_URL)
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection OK -- waiting for jobs on '%s'", QUEUE_KEY)

    while True:
        # BRPOP blocks until a job is available (timeout=0 means wait forever)
        _, raw = redis_client.brpop(QUEUE_KEY, timeout=0)
        job: dict = json.loads(raw)
        session_id: str = job["session_id"]
        logger.info("Received job for session %s", session_id)

        try:
            update_session_status(session_id, "grading")
            logger.info("Session %s: status -> grading", session_id)

            results = run_pipeline(job)

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
