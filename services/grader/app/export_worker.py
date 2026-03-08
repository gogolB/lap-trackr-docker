"""Export worker -- polls Redis for export jobs and converts SVO2 to MP4+NPZ."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import signal
import sys
import threading
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
EXPORT_CANCEL_KEY_PREFIX = "export_cancel:"
STAGE_FIELD_PREFIX = "stage__"
PROGRESS_TTL = 3600  # 1 hour

_shutdown = False
_progress_lock = threading.Lock()


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
    stage_field = f"{STAGE_FIELD_PREFIX}{stage}"
    stage_status = "completed" if total > 0 and current >= total else "running"

    with _progress_lock:
        raw_stage_data = redis_client.hget(key, stage_field)
        stage_started_at = now
        if raw_stage_data:
            try:
                parsed = json.loads(raw_stage_data)
                if current > 0 and parsed.get("started_at") is not None:
                    stage_started_at = float(parsed["started_at"])
            except (TypeError, ValueError, json.JSONDecodeError):
                stage_started_at = now
        elif current > 0:
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

        redis_client.hset(
            key,
            mapping={
                "stage": stage,
                "current": str(current),
                "total": str(total),
                "percent": str(pct),
                "detail": detail,
                "updated_at": str(now),
                "stage_started_at": str(stage_started_at),
                stage_field: stage_payload,
            },
        )
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
            from app.exporter import ExportCancelledError, export_svo2
            from app.color_detector import detect_tips

            update_session_status(session_id, "exporting")
            all_sample_paths: list[str] = []
            session_dir: str | None = None
            cancel_key = f"{EXPORT_CANCEL_KEY_PREFIX}{session_id}"
            post_export_status = job.get("post_export_status", "awaiting_init")

            cam_keys = [k for k in ("on_axis_path", "off_axis_path") if job.get(k) and Path(job[k]).exists()]
            cam_total = len(cam_keys)
            if cam_total == 0:
                raise FileNotFoundError(f"Session {session_id} has no exportable camera files")

            cancel_event = threading.Event()

            def _should_cancel() -> bool:
                return cancel_event.is_set() or bool(redis_client.exists(cancel_key))

            def _export_camera(cam_idx: int, key: str) -> tuple[str, str, dict]:
                svo_path = job[key]
                cam_label = key.replace("_path", "")
                stage_key = f"export_{cam_label}"
                logger.info("Exporting %s: %s", key, svo_path)
                _publish_progress(
                    redis_client,
                    session_id,
                    stage=stage_key,
                    current=0,
                    total=1,
                    detail=f"Opening {cam_label.replace('_', '-')} camera recording ({cam_idx + 1}/{cam_total})",
                )

                def on_export_progress(frame: int, total: int, _ci=cam_idx, _ct=cam_total, _cl=cam_label) -> None:
                    _publish_progress(
                        redis_client,
                        session_id,
                        stage=f"export_{_cl}",
                        current=frame,
                        total=max(total, frame, 1),
                        detail=f"Exporting {_cl.replace('_', '-')} camera ({_ci + 1}/{_ct})",
                    )

                result = export_svo2(
                    svo_path,
                    on_progress=on_export_progress,
                    should_cancel=_should_cancel,
                )
                _publish_progress(
                    redis_client,
                    session_id,
                    stage=stage_key,
                    current=max(result["frame_count"], 1),
                    total=max(result["frame_count"], 1),
                    detail=f"Finalized {cam_label.replace('_', '-')} outputs",
                )
                logger.info(
                    "  Exported %d frames → %s",
                    result["frame_count"],
                    ", ".join(result.get("mp4_paths", [result["mp4_path"]])),
                )
                return cam_label, svo_path, result

            camera_results: dict[str, tuple[str, dict]] = {}
            first_error: Exception | None = None
            with ThreadPoolExecutor(max_workers=cam_total, thread_name_prefix="export") as executor:
                futures = [
                    executor.submit(_export_camera, cam_idx, key)
                    for cam_idx, key in enumerate(cam_keys)
                ]
                for future in as_completed(futures):
                    try:
                        cam_label, svo_path, result = future.result()
                    except ExportCancelledError as exc:
                        cancel_event.set()
                        if first_error is None:
                            first_error = exc
                    except Exception as exc:
                        cancel_event.set()
                        if first_error is None:
                            first_error = exc
                    else:
                        camera_results[cam_label] = (svo_path, result)

            if first_error is not None:
                raise first_error

            for cam_label in ("on_axis", "off_axis"):
                camera_result = camera_results.get(cam_label)
                if camera_result is None:
                    continue
                svo_path, result = camera_result
                all_sample_paths.extend(result.get("sample_paths", []))
                if session_dir is None:
                    session_dir = str(Path(svo_path).parent)

            # Run color detection on sample frames
            detect_total = max(len(all_sample_paths), 1)
            _publish_progress(
                redis_client,
                session_id,
                stage="detect_tips",
                current=0,
                total=detect_total,
                detail="Detecting initial tip positions",
            )
            tip_detections: dict[str, list[dict]] = {}
            if session_dir and all_sample_paths:
                import cv2
                for idx, sample_path in enumerate(all_sample_paths, start=1):
                    if Path(sample_path).exists():
                        bgr = cv2.imread(sample_path)
                        if bgr is not None:
                            tips = detect_tips(bgr)
                            filename = Path(sample_path).name
                            tip_detections[filename] = tips
                    _publish_progress(
                        redis_client,
                        session_id,
                        stage="detect_tips",
                        current=idx,
                        total=detect_total,
                        detail=Path(sample_path).name,
                    )

                # Save tip detections
                det_path = Path(session_dir) / "tip_detections.json"
                det_path.write_text(json.dumps(tip_detections, indent=2))
                logger.info("Saved tip detections to %s", det_path)
            else:
                _publish_progress(
                    redis_client,
                    session_id,
                    stage="detect_tips",
                    current=1,
                    total=1,
                    detail="No sample frames found",
                )

            _publish_progress(redis_client, session_id, stage="complete", current=1, total=1)
            redis_client.delete(cancel_key)
            update_session_status(session_id, post_export_status)
            logger.info("Session %s: export complete, status -> %s", session_id, post_export_status)

        except ExportCancelledError as exc:
            redis_client.delete(f"{EXPORT_CANCEL_KEY_PREFIX}{session_id}")
            logger.info("Session %s export cancelled: %s", session_id, exc)
            continue

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
