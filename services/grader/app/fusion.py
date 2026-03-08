"""Dual-camera 3D fusion -- combines depth back-projection and stereo triangulation."""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

from app.backends.base import Detection
from app.pose_estimator import _lookup_depth, _pixel_to_3d, _transform_point

logger = logging.getLogger("grader.fusion")


class StereoFusionError(Exception):
    """Raised when stereo fusion cannot proceed (e.g. singular extrinsic matrix)."""


def _triangulate_point(
    pt_on: tuple[float, float],
    pt_off: tuple[float, float],
    K_on: np.ndarray,
    K_off: np.ndarray,
    T_on_to_off: np.ndarray,
) -> list[float] | None:
    """Triangulate a 3D point from corresponding 2D observations.

    Uses the on-axis camera as the reference frame (identity projection).
    """
    # Projection matrix for on-axis: K @ [I | 0]
    P_on = K_on @ np.eye(3, 4)
    # Projection matrix for off-axis: K @ [R | t] from T_on_to_off
    P_off = K_off @ T_on_to_off[:3, :]

    pts_on = np.array([[pt_on[0], pt_on[1]]], dtype=np.float64).T  # (2, 1)
    pts_off = np.array([[pt_off[0], pt_off[1]]], dtype=np.float64).T

    points_4d = cv2.triangulatePoints(P_on, P_off, pts_on, pts_off)  # (4, 1)
    if points_4d[3, 0] == 0:
        return None

    point_3d = points_4d[:3, 0] / points_4d[3, 0]
    if not all(math.isfinite(v) for v in point_3d):
        return None

    return [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]


def _build_K(calib: dict) -> np.ndarray:
    """Build 3x3 intrinsic matrix from calibration dict."""
    intr = calib["intrinsics"]
    return np.array([
        [intr["fx"], 0, intr["cx"]],
        [0, intr["fy"], intr["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)


def _get_intrinsics(calib: dict) -> tuple[float, float, float, float]:
    intr = calib["intrinsics"]
    return float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])


def fuse_dual_camera(
    on_detections: list[list[Detection]],
    off_detections: list[list[Detection]],
    on_depth: list[np.ndarray],
    off_depth: list[np.ndarray],
    on_calib: dict,
    off_calib: dict,
    stereo_calib: dict,
    fps: float,
) -> list[dict[str, Any]]:
    """Fuse detections from both cameras into 3D poses.

    Three strategies with weighted averaging:
    1. Monocular depth back-projection from on-axis
    2. Monocular depth back-projection from off-axis (transformed to on-axis frame)
    3. Stereo triangulation using T_on_to_off

    Returns list of pose dicts with frame_idx, timestamp, left_tip, right_tip.
    """
    T_on_to_off = np.array(stereo_calib["T_on_to_off"], dtype=np.float64)
    try:
        T_off_to_on = np.linalg.inv(T_on_to_off)
    except np.linalg.LinAlgError:
        raise StereoFusionError("Singular extrinsic matrix T_on_to_off, cannot fuse cameras")

    K_on = _build_K(on_calib)
    K_off = _build_K(off_calib)
    fx_on, fy_on, cx_on, cy_on = _get_intrinsics(on_calib)
    fx_off, fy_off, cx_off, cy_off = _get_intrinsics(off_calib)

    # Extrinsic transforms (camera-to-board) if available
    T_on_board = np.array(on_calib["extrinsic_matrix"], dtype=np.float64) if on_calib.get("extrinsic_matrix") else None
    T_off_board = np.array(off_calib["extrinsic_matrix"], dtype=np.float64) if off_calib.get("extrinsic_matrix") else None

    n_frames = min(len(on_detections), len(off_detections))
    poses: list[dict[str, Any]] = []

    for frame_idx in range(n_frames):
        timestamp = frame_idx / fps
        on_dets = {d.label: d for d in on_detections[frame_idx]}
        off_dets = {d.label: d for d in off_detections[frame_idx]}

        result: dict[str, Any] = {
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 6),
            "left_tip": None,
            "right_tip": None,
        }

        for label in ("left_tip", "right_tip"):
            points: list[tuple[list[float], float]] = []  # (point, weight)

            # Strategy 1: On-axis monocular depth
            if label in on_dets and frame_idx < len(on_depth):
                det = on_dets[label]
                depth = _lookup_depth(on_depth[frame_idx], det.x, det.y)
                if math.isfinite(depth) and depth > 0:
                    p = _pixel_to_3d(det.x, det.y, depth, fx_on, fy_on, cx_on, cy_on)
                    # Transform to board frame if extrinsic available
                    if T_on_board is not None:
                        p = _transform_point(p, T_on_board.tolist())
                    w = det.confidence * min(1.0, 1.0 / (depth + 0.1))
                    points.append((p, w))

            # Strategy 2: Off-axis monocular depth (transform to on-axis / board frame)
            if label in off_dets and frame_idx < len(off_depth):
                det = off_dets[label]
                depth = _lookup_depth(off_depth[frame_idx], det.x, det.y)
                if math.isfinite(depth) and depth > 0:
                    p_off = _pixel_to_3d(det.x, det.y, depth, fx_off, fy_off, cx_off, cy_off)
                    # Transform from off-axis camera frame to on-axis camera frame
                    p = _transform_point(p_off, T_off_to_on.tolist())
                    # Then to board frame if available
                    if T_on_board is not None:
                        p = _transform_point(p, T_on_board.tolist())
                    w = det.confidence * min(1.0, 1.0 / (depth + 0.1)) * 0.9  # slight discount
                    points.append((p, w))

            # Strategy 3: Stereo triangulation
            if label in on_dets and label in off_dets:
                on_det = on_dets[label]
                off_det = off_dets[label]
                tri_point = _triangulate_point(
                    (on_det.x, on_det.y),
                    (off_det.x, off_det.y),
                    K_on, K_off, T_on_to_off,
                )
                if tri_point is not None:
                    # Triangulated point is in on-axis camera frame
                    if T_on_board is not None:
                        tri_point = _transform_point(tri_point, T_on_board.tolist())
                    w = min(on_det.confidence, off_det.confidence) * 0.8
                    points.append((tri_point, w))

            # Weighted average
            if points:
                total_w = sum(w for _, w in points)
                if total_w > 0:
                    fused = [0.0, 0.0, 0.0]
                    for p, w in points:
                        for i in range(3):
                            fused[i] += p[i] * w / total_w
                    result[label] = [round(v, 6) for v in fused]

        poses.append(result)

    valid = sum(1 for p in poses if p["left_tip"] is not None and p["right_tip"] is not None)
    logger.info("Fused 3D poses for %d frames (%d with both tips)", len(poses), valid)
    return poses
