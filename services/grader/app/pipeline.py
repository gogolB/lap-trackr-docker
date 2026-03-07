"""Grading pipeline -- orchestrates all stages from SVO2 to metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.metrics import calculate_metrics
from app.model_loader import get_backend
from app.pose_estimator import estimate_poses, estimate_poses_dual
from app.svo_loader import load_svo2

logger = logging.getLogger("grader.pipeline")


def _load_calibration(job: dict, key: str = "calibration_path") -> dict | None:
    """Load calibration JSON from the path specified in the job, if any."""
    calib_path = job.get(key)
    if not calib_path:
        return None

    p = Path(calib_path)
    if not p.exists():
        logger.warning("Calibration file not found: %s", calib_path)
        return None

    try:
        calibration = json.loads(p.read_text())
        logger.info("Loaded calibration from %s", calib_path)
        return calibration
    except Exception as exc:
        logger.warning("Failed to parse calibration %s: %s", calib_path, exc)
        return None


def _load_query_points(job: dict) -> np.ndarray | None:
    """Load query points from tip_init.json in the session directory."""
    tip_init_path = job.get("tip_init_path")

    # Try explicit path first, then derive from on_axis_path
    if not tip_init_path:
        on_axis_path = job.get("on_axis_path")
        if on_axis_path:
            tip_init_path = str(Path(on_axis_path).parent / "tip_init.json")

    if not tip_init_path or not Path(tip_init_path).exists():
        logger.info("No tip_init.json found, using default detection")
        return None

    try:
        tip_data = json.loads(Path(tip_init_path).read_text())
        # tip_data is {filename: [{label, x, y, confidence, color}, ...]}
        # Convert to (N, 3) array of [frame_idx=0, x, y]
        # Use detections from the first available frame
        points: list[list[float]] = []
        for detections in tip_data.values():
            for det in detections:
                points.append([0.0, float(det["x"]), float(det["y"])])
            if points:
                break  # Use first frame's detections

        if points:
            qp = np.array(points, dtype=np.float32)
            logger.info("Loaded %d query points from tip_init.json", len(qp))
            return qp
    except Exception as exc:
        logger.warning("Failed to load tip_init.json: %s", exc)

    return None


def _load_stereo_calibration(job: dict) -> dict | None:
    """Load stereo calibration from session dir."""
    stereo_path = job.get("stereo_calibration_path")
    if not stereo_path:
        on_axis_path = job.get("on_axis_path")
        if on_axis_path:
            stereo_path = str(Path(on_axis_path).parent / "stereo_calibration.json")

    if not stereo_path or not Path(stereo_path).exists():
        return None

    try:
        data = json.loads(Path(stereo_path).read_text())
        logger.info("Loaded stereo calibration from %s", stereo_path)
        return data
    except Exception as exc:
        logger.warning("Failed to load stereo calibration: %s", exc)
        return None


def _load_off_axis_calibration(job: dict) -> dict | None:
    """Load off-axis camera calibration from session dir."""
    on_axis_path = job.get("on_axis_path")
    if not on_axis_path:
        return None
    calib_path = str(Path(on_axis_path).parent / "calibration_off_axis.json")
    if not Path(calib_path).exists():
        return None
    try:
        data = json.loads(Path(calib_path).read_text())
        logger.info("Loaded off-axis calibration from %s", calib_path)
        return data
    except Exception as exc:
        logger.warning("Failed to load off-axis calibration: %s", exc)
        return None


def run_pipeline(job: dict) -> dict[str, Any]:
    """Run the full grading pipeline on a session's SVO2 files.

    Parameters
    ----------
    job : dict
        Must contain ``session_id``, ``on_axis_path`` and ``off_axis_path``.
        Optionally ``calibration_path``, ``stereo_calibration_path``,
        ``tip_init_path`` for enhanced tracking.

    Returns
    -------
    dict
        ``{"metrics": {...}, "poses": [...]}``
    """

    on_axis_path: str = job["on_axis_path"]
    off_axis_path: str = job["off_axis_path"]

    # Load calibrations
    on_calibration = _load_calibration(job, "calibration_path")
    off_calibration = _load_off_axis_calibration(job)
    stereo_calibration = _load_stereo_calibration(job)
    query_points = _load_query_points(job)

    # Stage 1 -- load SVO2 files and extract frames + depth maps.
    logger.info("Stage 1: Loading SVO2 from %s", on_axis_path)
    frames, depth_maps, fps = load_svo2(on_axis_path)
    logger.info(
        "  Loaded %d frames (%d depth maps) at %.1f fps",
        len(frames),
        len(depth_maps),
        fps,
    )

    logger.info("Stage 1b: Loading off-axis SVO2 from %s", off_axis_path)
    off_frames, off_depth, off_fps = load_svo2(off_axis_path)
    logger.info("  Off-axis: %d frames at %.1f fps", len(off_frames), off_fps)

    # Stage 2 -- instrument detection via active model backend.
    logger.info("Stage 2: Running instrument detection on %d frames", len(frames))
    backend = get_backend()

    # Run detection on on-axis camera
    on_detections = backend.detect(frames, query_points=query_points)
    logger.info("  On-axis detections generated for %d frames", len(on_detections))

    # Run detection on off-axis camera (with query points if available)
    off_detections = backend.detect(off_frames, query_points=query_points)
    logger.info("  Off-axis detections generated for %d frames", len(off_detections))

    # Stage 3 -- 3D pose estimation
    logger.info("Stage 3: Estimating 3D poses")
    if stereo_calibration and on_calibration and off_calibration:
        logger.info("  Using dual-camera fusion")
        poses_3d = estimate_poses_dual(
            on_detections,
            off_detections,
            depth_maps,
            off_depth,
            fps,
            on_calibration=on_calibration,
            off_calibration=off_calibration,
            stereo_calibration=stereo_calibration,
        )
    else:
        logger.info("  Using single-camera (on-axis only)")
        poses_3d = estimate_poses(on_detections, depth_maps, fps, calibration=on_calibration)
    logger.info("  Estimated %d pose records", len(poses_3d))

    # Stage 4 -- calculate skill metrics.
    logger.info("Stage 4: Calculating metrics")
    metrics = calculate_metrics(poses_3d, fps)
    logger.info("  Metrics: %s", metrics)

    return {"metrics": metrics, "poses": poses_3d}
