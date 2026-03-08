"""3D pose estimation from 2D detections and depth maps.

Projects 2D pixel-coordinate detections into 3D camera-frame coordinates
using the corresponding depth value and camera intrinsics (from calibration
JSON if available, otherwise config placeholders).
"""

from __future__ import annotations

import logging
import math
from typing import Any, List

import numpy as np

from app.config import CAMERA_CX, CAMERA_CY, CAMERA_FX, CAMERA_FY, DEFAULT_FPS
from app.backends.base import Detection

logger = logging.getLogger("grader.pose_estimator")
_POSE_META_KEYS = {"frame_idx", "timestamp"}


def _pixel_to_3d(
    x: float,
    y: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> list[float]:
    """Back-project a 2D pixel + depth into a 3D point [X, Y, Z] in metres."""

    z = float(depth)
    x3d = (x - cx) * z / fx
    y3d = (y - cy) * z / fy
    return [float(x3d), float(y3d), z]


def _transform_point(p: list[float], T: list[list[float]]) -> list[float]:
    """Apply a 4x4 extrinsic transform to a 3D point."""
    T_arr = np.array(T, dtype=np.float64)
    v = np.array([p[0], p[1], p[2], 1.0])
    w = T_arr @ v
    return [float(w[0]), float(w[1]), float(w[2])]


def _lookup_depth(
    depth_map: np.ndarray,
    x: float,
    y: float,
    patch_radius: int = 2,
) -> float:
    """Look up the depth at ``(x, y)``, using a small patch median for robustness.

    If the depth at the exact pixel is NaN or invalid, we take the median of
    a small neighbourhood instead.  Returns ``NaN`` if no valid depth is found.
    """

    h, w = depth_map.shape[:2]
    ix, iy = int(round(x)), int(round(y))

    # Clamp to image bounds.
    ix = max(0, min(ix, w - 1))
    iy = max(0, min(iy, h - 1))

    d = depth_map[iy, ix]
    if math.isfinite(d) and d > 0:
        return float(d)

    # Fallback: median of a patch.
    y0 = max(0, iy - patch_radius)
    y1 = min(h, iy + patch_radius + 1)
    x0 = max(0, ix - patch_radius)
    x1 = min(w, ix + patch_radius + 1)
    patch = depth_map[y0:y1, x0:x1].ravel()
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size > 0:
        return float(np.median(valid))

    return float("nan")


def estimate_poses(
    detections: list[list[Detection]],
    depth_maps: list[np.ndarray],
    fps: float | None = None,
    calibration: dict | None = None,
) -> list[dict[str, Any]]:
    """Convert per-frame 2D detections + depth into 3D pose records.

    Parameters
    ----------
    detections : list[list[Detection]]
        One list of detections per sampled frame.
    depth_maps : list[np.ndarray]
        Corresponding depth maps (same length as *detections*).
    fps : float, optional
        Recording FPS used to compute timestamps.
    calibration : dict, optional
        Calibration JSON dict with ``intrinsics`` and optionally ``extrinsic_matrix``.
        If None, falls back to config placeholder values.

    Returns
    -------
    list[dict]
        Each dict has keys ``frame_idx``, ``timestamp``, and one key per
        detected instrument-tip label. Frames where depth lookup fails for a
        tip will have ``None`` for that label.
    """

    if fps is None or fps <= 0:
        fps = DEFAULT_FPS

    if len(detections) != len(depth_maps):
        raise ValueError(
            f"Mismatch: {len(detections)} detection frames vs "
            f"{len(depth_maps)} depth maps"
        )

    # Extract intrinsics from calibration or use config defaults
    if calibration and "intrinsics" in calibration:
        intr = calibration["intrinsics"]
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
        logger.info(
            "Using calibration intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
            fx, fy, cx, cy,
        )
    else:
        fx, fy, cx, cy = CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
        logger.warning(
            "No calibration intrinsics, using config defaults: "
            "fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
            fx, fy, cx, cy,
        )

    # Extrinsic transform (camera-to-board frame)
    extrinsic = calibration.get("extrinsic_matrix") if calibration else None
    if extrinsic:
        logger.info("Extrinsic transform available, will convert to board frame")
    else:
        logger.info("No extrinsic transform, poses will be in camera frame")

    poses: list[dict[str, Any]] = []
    tip_labels = sorted(
        {
            det.label
            for frame_dets in detections
            for det in frame_dets
            if det.label
        }
    )

    for frame_idx, (frame_dets, depth_map) in enumerate(
        zip(detections, depth_maps)
    ):
        timestamp = frame_idx / fps
        pose: dict[str, Any] = {
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 6),
        }
        for label in tip_labels:
            pose[label] = None

        for det in frame_dets:
            depth = _lookup_depth(depth_map, det.x, det.y)
            if not math.isfinite(depth):
                logger.debug(
                    "Frame %d: no valid depth for %s at (%.1f, %.1f)",
                    frame_idx,
                    det.label,
                    det.x,
                    det.y,
                )
                continue

            point = _pixel_to_3d(det.x, det.y, depth, fx, fy, cx, cy)

            # Apply extrinsic transform if available
            if extrinsic:
                point = _transform_point(point, extrinsic)

            if det.label:
                pose[det.label] = point

        poses.append(pose)

    valid_counts = {
        label: sum(1 for pose in poses if pose.get(label) is not None)
        for label in tip_labels
    }
    logger.info(
        "Estimated 3D poses for %d frames with valid labels %s",
        len(poses),
        valid_counts,
    )
    return poses


def estimate_poses_dual(
    on_detections: list[list[Detection]],
    off_detections: list[list[Detection]],
    on_depth: list[np.ndarray],
    off_depth: list[np.ndarray],
    fps: float,
    on_calibration: dict | None = None,
    off_calibration: dict | None = None,
    stereo_calibration: dict | None = None,
) -> list[dict[str, Any]]:
    """Dual-camera 3D pose estimation with fusion.

    If stereo calibration is available, uses the fusion module for
    weighted 3D combination. Otherwise falls back to single-camera
    estimation on the on-axis camera.
    """
    if (
        stereo_calibration is not None
        and on_calibration is not None
        and off_calibration is not None
    ):
        from app.fusion import fuse_dual_camera

        logger.info("Using dual-camera fusion pipeline")
        return fuse_dual_camera(
            on_detections,
            off_detections,
            on_depth,
            off_depth,
            on_calibration,
            off_calibration,
            stereo_calibration,
            fps,
        )

    # Fallback: single-camera (on-axis only)
    logger.info("No stereo calibration, falling back to single-camera poses")
    return estimate_poses(on_detections, on_depth, fps, calibration=on_calibration)
