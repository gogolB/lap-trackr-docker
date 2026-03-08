"""Pass 4 — Stereo triangulation (CPU only)."""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

from app.passes.pass_data import PassData

logger = logging.getLogger("grader.passes.pass4_triangulation")

_REPROJ_ERROR_THRESHOLD = 15.0  # pixels
_VIS_THRESHOLD = 0.3


def _triangulate_dlt_svd(
    pt_on: tuple[float, float],
    pt_off: tuple[float, float],
    P_on: np.ndarray,
    P_off: np.ndarray,
) -> np.ndarray | None:
    """DLT triangulation via SVD for a single point correspondence.

    Returns 3D point as (3,) array or None if degenerate.
    """
    x1, y1 = pt_on
    x2, y2 = pt_off

    A = np.array([
        x1 * P_on[2] - P_on[0],
        y1 * P_on[2] - P_on[1],
        x2 * P_off[2] - P_off[0],
        y2 * P_off[2] - P_off[1],
    ], dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None

    point = X[:3] / X[3]
    if not all(math.isfinite(v) for v in point):
        return None
    return point


def _compute_reprojection_error(
    point_3d: np.ndarray,
    pt_2d: tuple[float, float],
    P: np.ndarray,
) -> float:
    """Compute reprojection error for a 3D point projected through P."""
    X_h = np.append(point_3d, 1.0)
    projected = P @ X_h
    if abs(projected[2]) < 1e-10:
        return float("inf")
    px = projected[0] / projected[2]
    py = projected[1] / projected[2]
    return float(math.hypot(px - pt_2d[0], py - pt_2d[1]))


def _backproject_single_camera(
    x: float,
    y: float,
    depth_maps: list[np.ndarray],
    frame_idx: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    """Back-project a 2D point using depth map (single-camera fallback)."""
    if frame_idx >= len(depth_maps):
        return None

    from app.pose_estimator import _lookup_depth

    depth = _lookup_depth(depth_maps[frame_idx], x, y)
    if not math.isfinite(depth) or depth <= 0:
        return None

    z = float(depth)
    x3d = (x - cx) * z / fx
    y3d = (y - cy) * z / fy
    return np.array([x3d, y3d, z], dtype=np.float64)


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Execute Pass 4: Stereo triangulation."""
    from app.config import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY

    logger.info("Pass 4: Stereo triangulation starting")

    labels = sorted(set(data.on_tracks.keys()) | set(data.off_tracks.keys()))
    if not labels:
        logger.warning("No tracks available, skipping triangulation")
        return

    n_frames = max(
        max((len(t) for t in data.on_tracks.values()), default=0),
        max((len(t) for t in data.off_tracks.values()), default=0),
    )
    if n_frames == 0:
        logger.warning("Zero frames, skipping triangulation")
        return

    # Build projection matrices from calibration
    has_stereo = (
        data.stereo_calib is not None
        and data.on_calib is not None
        and data.off_calib is not None
    )

    P_on = None
    P_off = None
    fx_on = fy_on = cx_on = cy_on = 0.0
    fx_off = fy_off = cx_off = cy_off = 0.0

    if has_stereo:
        from app.fusion import _build_K

        K_on = _build_K(data.on_calib)
        K_off = _build_K(data.off_calib)
        T_on_to_off = np.array(data.stereo_calib["T_on_to_off"], dtype=np.float64)
        P_on = K_on @ np.eye(3, 4, dtype=np.float64)
        P_off = K_off @ T_on_to_off[:3, :]

        intr_on = data.on_calib["intrinsics"]
        fx_on, fy_on = float(intr_on["fx"]), float(intr_on["fy"])
        cx_on, cy_on = float(intr_on["cx"]), float(intr_on["cy"])
        intr_off = data.off_calib["intrinsics"]
        fx_off, fy_off = float(intr_off["fx"]), float(intr_off["fy"])
        cx_off, cy_off = float(intr_off["cx"]), float(intr_off["cy"])
    else:
        fx_on, fy_on, cx_on, cy_on = CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
        fx_off, fy_off, cx_off, cy_off = fx_on, fy_on, cx_on, cy_on

    if on_progress:
        on_progress("pass4_triangulation", 0, n_frames, "Triangulating 3D positions")

    for label in labels:
        on_track = data.on_tracks.get(label)
        off_track = data.off_tracks.get(label)
        on_vis = data.on_visibility.get(label)
        off_vis = data.off_visibility.get(label)

        traj = np.full((n_frames, 3), np.nan, dtype=np.float64)
        errors = np.full(n_frames, np.nan, dtype=np.float64)

        for fidx in range(n_frames):
            on_ok = (
                on_track is not None
                and fidx < len(on_track)
                and on_vis is not None
                and fidx < len(on_vis)
                and on_vis[fidx] >= _VIS_THRESHOLD
                and not np.any(np.isnan(on_track[fidx]))
            )
            off_ok = (
                off_track is not None
                and fidx < len(off_track)
                and off_vis is not None
                and fidx < len(off_vis)
                and off_vis[fidx] >= _VIS_THRESHOLD
                and not np.any(np.isnan(off_track[fidx]))
            )

            if on_ok and off_ok and has_stereo:
                # Stereo DLT triangulation
                pt_on = (float(on_track[fidx, 0]), float(on_track[fidx, 1]))
                pt_off = (float(off_track[fidx, 0]), float(off_track[fidx, 1]))
                point = _triangulate_dlt_svd(pt_on, pt_off, P_on, P_off)
                if point is not None:
                    err_on = _compute_reprojection_error(point, pt_on, P_on)
                    err_off = _compute_reprojection_error(point, pt_off, P_off)
                    mean_err = (err_on + err_off) / 2.0
                    if mean_err <= _REPROJ_ERROR_THRESHOLD:
                        traj[fidx] = point
                        errors[fidx] = mean_err
                        continue

            # Single-camera depth fallback
            if on_ok:
                pt = _backproject_single_camera(
                    float(on_track[fidx, 0]),
                    float(on_track[fidx, 1]),
                    data.on_depth,
                    fidx,
                    fx_on, fy_on, cx_on, cy_on,
                )
                if pt is not None:
                    traj[fidx] = pt
                    errors[fidx] = 0.0  # no reprojection error for monocular
                    continue

            if off_ok:
                pt = _backproject_single_camera(
                    float(off_track[fidx, 0]),
                    float(off_track[fidx, 1]),
                    data.off_depth,
                    fidx,
                    fx_off, fy_off, cx_off, cy_off,
                )
                if pt is not None:
                    traj[fidx] = pt
                    errors[fidx] = 0.0

            if on_progress and (fidx + 1) % 50 == 0:
                on_progress("pass4_triangulation", fidx + 1, n_frames, f"Triangulating {label}")

        data.trajectories_3d[label] = traj
        data.reprojection_errors[label] = errors

        valid_count = np.sum(~np.isnan(traj[:, 0]))
        logger.info(
            "%s: %d/%d frames with valid 3D positions (mean reproj err: %.2f px)",
            label,
            valid_count,
            n_frames,
            float(np.nanmean(errors)) if valid_count > 0 else 0.0,
        )

    if on_progress:
        on_progress("pass4_triangulation", n_frames, n_frames, "Triangulation complete")
    logger.info("Pass 4 complete")
