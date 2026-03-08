"""Grading pipeline -- orchestrates all stages from SVO2 to metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from app.camera_transform import adjust_calibration
from app.metrics import calculate_metrics
from app.model_loader import get_backend
from app.pose_estimator import estimate_poses, estimate_poses_dual
from app.svo_loader import load_svo2
from app.tracking_renderer import (
    render_tracking_video,
    write_detection_csv,
    write_pose_csv,
)

ProgressCallback = Optional[Callable[[str, int, int, str], None]]

logger = logging.getLogger("grader.pipeline")


def _results_dir(job: dict) -> Path:
    """Return the per-session results directory for generated artifacts."""
    on_axis_path = job.get("on_axis_path")
    if not on_axis_path:
        raise FileNotFoundError("Cannot derive results directory without on_axis_path")
    results_dir = Path(on_axis_path).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


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


def run_pipeline(job: dict, on_progress: ProgressCallback = None) -> dict[str, Any]:
    """Run the full grading pipeline on a session's SVO2 files.

    Parameters
    ----------
    job : dict
        Must contain ``session_id``, ``on_axis_path`` and ``off_axis_path``.
        Optionally ``calibration_path``, ``stereo_calibration_path``,
        ``tip_init_path`` for enhanced tracking.
    on_progress : callable, optional
        ``(stage, current, total, detail)`` callback for live progress updates.

    Returns
    -------
    dict
        ``{"metrics": {...}, "poses": [...]}``
    """

    def _progress(stage: str, current: int, total: int, detail: str = "") -> None:
        if on_progress:
            on_progress(stage, current, total, detail)

    on_axis_path: str = job["on_axis_path"]
    off_axis_path: str = job["off_axis_path"]
    results_dir = _results_dir(job)
    warnings: list[str] = []

    if not Path(on_axis_path).exists():
        raise FileNotFoundError(f"On-axis SVO2 not found: {on_axis_path}")
    if not Path(off_axis_path).exists():
        raise FileNotFoundError(f"Off-axis SVO2 not found: {off_axis_path}")

    # Load calibrations
    on_calibration = _load_calibration(job, "calibration_path")
    off_calibration = _load_off_axis_calibration(job)
    stereo_calibration = _load_stereo_calibration(job)
    query_points = _load_query_points(job)
    camera_config = job.get("camera_config")

    on_calibration = adjust_calibration(on_calibration, camera_config, "on_axis")
    off_calibration = adjust_calibration(off_calibration, camera_config, "off_axis")

    # Stage 1 -- load SVO2 files and extract frames + depth maps.
    _progress("load_on_axis", 0, 1, "Opening on-axis recording")
    logger.info("Stage 1: Loading SVO2 from %s", on_axis_path)
    frames, depth_maps, fps = load_svo2(
        on_axis_path,
        on_progress=lambda current, total: _progress(
            "load_on_axis",
            current,
            total,
            "Loading on-axis frames",
        ),
        camera_config=camera_config,
    )
    logger.info(
        "  Loaded %d frames (%d depth maps) at %.1f fps",
        len(frames),
        len(depth_maps),
        fps,
    )
    _progress("load_on_axis", 1, 1, f"Loaded {len(frames)} sampled frames")

    _progress("load_off_axis", 0, 1, "Opening off-axis recording")
    logger.info("Stage 1b: Loading off-axis SVO2 from %s", off_axis_path)
    off_frames, off_depth, off_fps = load_svo2(
        off_axis_path,
        on_progress=lambda current, total: _progress(
            "load_off_axis",
            current,
            total,
            "Loading off-axis frames",
        ),
        camera_config=camera_config,
    )
    logger.info("  Off-axis: %d frames at %.1f fps", len(off_frames), off_fps)
    _progress("load_off_axis", 1, 1, f"Loaded {len(off_frames)} sampled frames")

    # Stage 2 -- instrument detection via active model backend.
    _progress("detect_on_axis", 0, max(len(frames), 1), "On-axis camera")
    logger.info("Stage 2: Running instrument detection on %d frames", len(frames))
    backend = get_backend()

    # Run detection on on-axis camera
    on_detections = backend.detect(
        frames,
        query_points=query_points,
        on_progress=lambda current, total: _progress(
            "detect_on_axis",
            current,
            total,
            "On-axis camera",
        ),
    )
    logger.info("  On-axis detections generated for %d frames", len(on_detections))
    _progress(
        "detect_on_axis",
        len(on_detections),
        max(len(frames), 1),
        f"Detected instruments in {len(on_detections)} frames",
    )

    _progress("detect_off_axis", 0, max(len(off_frames), 1), "Off-axis camera")
    # Run detection on off-axis camera (with query points if available)
    off_detections = backend.detect(
        off_frames,
        query_points=query_points,
        on_progress=lambda current, total: _progress(
            "detect_off_axis",
            current,
            total,
            "Off-axis camera",
        ),
    )
    logger.info("  Off-axis detections generated for %d frames", len(off_detections))
    _progress(
        "detect_off_axis",
        len(off_detections),
        max(len(off_frames), 1),
        f"Detected instruments in {len(off_detections)} frames",
    )

    tracking_videos: list[str] = []
    tracking_csvs: list[str] = []

    # Persist per-frame 2D detections for playback/reanalysis.
    try:
        tracking_csvs.append(
            write_detection_csv(
                on_detections,
                str(results_dir / "tracking_on_axis.csv"),
                fps,
                "on_axis",
            )
        )
        tracking_csvs.append(
            write_detection_csv(
                off_detections,
                str(results_dir / "tracking_off_axis.csv"),
                off_fps,
                "off_axis",
            )
        )
    except Exception as exc:
        logger.warning("Failed to write tracking CSVs: %s", exc)
        warnings.append(f"Tracking CSV export failed: {exc}")

    # Stage 3 -- render tracking overlay video
    for stage_key, camera_name, cam_frames, cam_detections, cam_fps, filename in (
        ("render_on_axis", "on_axis", frames, on_detections, fps, "tracking_on_axis.mp4"),
        ("render_off_axis", "off_axis", off_frames, off_detections, off_fps, "tracking_off_axis.mp4"),
    ):
        _progress(
            stage_key,
            0,
            max(min(len(cam_frames), len(cam_detections)), 1),
            f"{camera_name.replace('_', '-')} camera",
        )
        try:
            tracking_videos.append(
                render_tracking_video(
                    cam_frames,
                    cam_detections,
                    str(results_dir / filename),
                    cam_fps,
                    on_progress=lambda current, total, _stage=stage_key, _camera=camera_name: _progress(
                        _stage,
                        current,
                        total,
                        f"{_camera.replace('_', '-')} camera",
                    ),
                )
            )
            _progress(
                stage_key,
                min(len(cam_frames), len(cam_detections)),
                max(min(len(cam_frames), len(cam_detections)), 1),
                f"Rendered {camera_name.replace('_', '-')} tracking video",
            )
        except Exception as exc:
            logger.warning("Failed to render %s tracking video: %s", camera_name, exc)
            warnings.append(f"{camera_name} tracking video render failed: {exc}")

    # Stage 4 -- 3D pose estimation
    _progress("estimate_poses", 0, 1, "Fusing 3D tip positions")
    logger.info("Stage 3: Estimating 3D poses")
    if stereo_calibration and on_calibration and off_calibration:
        logger.info("  Using dual-camera fusion")
        try:
            from app.fusion import StereoFusionError
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
        except StereoFusionError as exc:
            logger.warning("Stereo fusion failed (%s), falling back to single-camera", exc)
            warnings.append(f"Stereo fusion failed: {exc}. Used single-camera estimation instead.")
            poses_3d = estimate_poses(on_detections, depth_maps, fps, calibration=on_calibration)
    else:
        logger.info("  Using single-camera (on-axis only)")
        poses_3d = estimate_poses(on_detections, depth_maps, fps, calibration=on_calibration)
    logger.info("  Estimated %d pose records", len(poses_3d))
    _progress("estimate_poses", 1, 1, f"Calculated {len(poses_3d)} fused pose records")

    try:
        tracking_csvs.append(
            write_pose_csv(
                poses_3d,
                str(results_dir / "tracked_positions_world.csv"),
            )
        )
    except Exception as exc:
        logger.warning("Failed to write world pose CSV: %s", exc)
        warnings.append(f"World pose CSV export failed: {exc}")

    # Stage 5 -- calculate skill metrics.
    _progress("calculate_metrics", 0, 1, "Calculating grading metrics")
    logger.info("Stage 4: Calculating metrics")
    metrics = calculate_metrics(poses_3d, fps)
    logger.info("  Metrics: %s", metrics)
    _progress("calculate_metrics", 1, 1, "Metrics complete")

    result: dict[str, Any] = {"metrics": metrics, "poses": poses_3d}
    if tracking_videos:
        result["tracking_videos"] = tracking_videos
    if tracking_csvs:
        result["tracking_csvs"] = tracking_csvs
    if warnings:
        result["warnings"] = warnings
    return result
