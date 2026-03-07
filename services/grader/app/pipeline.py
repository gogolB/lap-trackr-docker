"""Grading pipeline -- orchestrates all stages from SVO2 to metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.metrics import calculate_metrics
from app.model_loader import get_backend
from app.pose_estimator import estimate_poses
from app.svo_loader import load_svo2

logger = logging.getLogger("grader.pipeline")


def _load_calibration(job: dict) -> dict | None:
    """Load calibration JSON from the path specified in the job, if any."""
    calib_path = job.get("calibration_path")
    if not calib_path:
        logger.info("No calibration_path in job, using config defaults")
        return None

    p = Path(calib_path)
    if not p.exists():
        logger.warning("Calibration file not found: %s, using config defaults", calib_path)
        return None

    try:
        calibration = json.loads(p.read_text())
        logger.info("Loaded calibration from %s", calib_path)
        return calibration
    except Exception as exc:
        logger.warning("Failed to parse calibration %s: %s, using config defaults", calib_path, exc)
        return None


def run_pipeline(job: dict) -> dict[str, Any]:
    """Run the full grading pipeline on a session's SVO2 files.

    Parameters
    ----------
    job : dict
        Must contain ``session_id``, ``on_axis_path`` and ``off_axis_path``.
        Optionally ``calibration_path`` for camera calibration.

    Returns
    -------
    dict
        ``{"metrics": {...}, "poses": [...]}``
    """

    on_axis_path: str = job["on_axis_path"]
    off_axis_path: str = job["off_axis_path"]

    # Load calibration if available
    calibration = _load_calibration(job)

    # Stage 1 -- load SVO2 files and extract frames + depth maps.
    logger.info("Stage 1: Loading SVO2 from %s", on_axis_path)
    frames, depth_maps, fps = load_svo2(on_axis_path)
    logger.info(
        "  Loaded %d frames (%d depth maps) at %.1f fps",
        len(frames),
        len(depth_maps),
        fps,
    )

    # Also load off-axis for future use (currently just log frame count).
    logger.info("Stage 1b: Loading off-axis SVO2 from %s", off_axis_path)
    off_frames, off_depth, off_fps = load_svo2(off_axis_path)
    logger.info("  Off-axis: %d frames at %.1f fps", len(off_frames), off_fps)

    # Stage 2 -- instrument detection via active model backend.
    logger.info("Stage 2: Running instrument detection on %d frames", len(frames))
    backend = get_backend()
    detections = backend.detect(frames)
    logger.info("  Detections generated for %d frames", len(detections))

    # Stage 3 -- 3D pose estimation from detections + depth.
    logger.info("Stage 3: Estimating 3D poses")
    poses_3d = estimate_poses(detections, depth_maps, fps, calibration=calibration)
    logger.info("  Estimated %d pose records", len(poses_3d))

    # Stage 4 -- calculate skill metrics.
    logger.info("Stage 4: Calculating metrics")
    metrics = calculate_metrics(poses_3d, fps)
    logger.info("  Metrics: %s", metrics)

    return {"metrics": metrics, "poses": poses_3d}
