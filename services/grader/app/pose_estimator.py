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
        Each dict has keys ``frame_idx``, ``timestamp``, ``left_tip`` and
        ``right_tip`` (each a ``[x, y, z]`` list in metres).  Frames where
        depth lookup fails for a tip will have ``None`` for that tip.
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

    for frame_idx, (frame_dets, depth_map) in enumerate(
        zip(detections, depth_maps)
    ):
        timestamp = frame_idx / fps

        left_tip: list[float] | None = None
        right_tip: list[float] | None = None

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

            if det.label == "left_tip":
                left_tip = point
            elif det.label == "right_tip":
                right_tip = point

        poses.append(
            {
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 6),
                "left_tip": left_tip,
                "right_tip": right_tip,
            }
        )

    valid = sum(
        1 for p in poses if p["left_tip"] is not None and p["right_tip"] is not None
    )
    logger.info(
        "Estimated 3D poses for %d frames (%d with both tips valid)",
        len(poses),
        valid,
    )
    return poses
