"""Pass 4 — Depth-map back-projection with cross-camera validation.

Each ZED camera already produces high-quality depth maps from its internal
stereo pair.  Instead of trying DLT triangulation across two cameras that are
~90° apart with different focal lengths (which fails catastrophically when
CoTracker drifts on one view), we:

1. Back-project the 2D track from each camera using its own depth map →
   3D point in that camera's coordinate frame.
2. Transform both points into the on-axis frame using stereo calibration.
3. If both cameras have valid back-projections and they agree within a
   threshold, average them (weighted by visibility confidence).
4. If they disagree, use the higher-confidence single-camera estimate.
5. If only one camera has a valid track + depth, use it directly.

This eliminates the DLT failure mode entirely while using the best depth
data available.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

from app.passes.pass_data import PassData

logger = logging.getLogger("grader.passes.pass4_triangulation")

_VIS_THRESHOLD = 0.3

# Physical depth range for the surgical training scene (metres).
# ZED depth maps show 0.11–1.14 m; allow generous margin.
_DEPTH_MIN = 0.05   # 5 cm
_DEPTH_MAX = 3.0     # 3 m

# Maximum distance (metres) between the two cameras' 3D estimates for them
# to be considered in agreement and averaged.  Beyond this, we take the one
# with higher confidence.  50 mm is generous — real agreement is <10 mm.
_CROSS_CAM_AGREE_THRESHOLD = 0.050  # 50 mm


def _backproject(
    x: float,
    y: float,
    depth_maps: list[np.ndarray],
    frame_idx: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    """Back-project a 2D pixel to 3D using the camera's depth map.

    Returns the 3D point in the camera's own coordinate frame, or None.
    """
    if frame_idx >= len(depth_maps) or len(depth_maps) == 0:
        return None

    from app.pose_estimator import _lookup_depth

    depth = _lookup_depth(depth_maps[frame_idx], x, y)
    if not math.isfinite(depth) or depth <= 0:
        return None

    z = float(depth)
    if z < _DEPTH_MIN or z > _DEPTH_MAX:
        return None

    x3d = (x - cx) * z / fx
    y3d = (y - cy) * z / fy
    return np.array([x3d, y3d, z], dtype=np.float64)


def _transform_point(point: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 rigid transform to a 3D point."""
    p_h = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    result = T @ p_h
    return result[:3]


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Execute Pass 4: Depth-map back-projection with cross-camera validation."""
    from app.config import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY

    logger.info("Pass 4: Depth-map back-projection starting")

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

    # Camera intrinsics
    has_calib = data.on_calib is not None and data.off_calib is not None

    if has_calib:
        intr_on = data.on_calib["intrinsics"]
        fx_on, fy_on = float(intr_on["fx"]), float(intr_on["fy"])
        cx_on, cy_on = float(intr_on["cx"]), float(intr_on["cy"])
        intr_off = data.off_calib["intrinsics"]
        fx_off, fy_off = float(intr_off["fx"]), float(intr_off["fy"])
        cx_off, cy_off = float(intr_off["cx"]), float(intr_off["cy"])
    else:
        fx_on, fy_on, cx_on, cy_on = CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
        fx_off, fy_off, cx_off, cy_off = fx_on, fy_on, cx_on, cy_on

    # Transform from off-axis camera frame to on-axis camera frame
    T_off_to_on = None
    if data.stereo_calib is not None:
        T_on_to_off = np.array(data.stereo_calib["T_on_to_off"], dtype=np.float64)
        T_off_to_on = np.linalg.inv(T_on_to_off)
        logger.info(
            "Cross-camera validation enabled (baseline: %.1f mm)",
            np.linalg.norm(T_on_to_off[:3, 3]) * 1000,
        )

    if on_progress:
        on_progress("pass4_triangulation", 0, n_frames, "Back-projecting 3D positions")

    for label in labels:
        on_track = data.on_tracks.get(label)
        off_track = data.off_tracks.get(label)
        on_vis = data.on_visibility.get(label)
        off_vis = data.off_visibility.get(label)

        traj = np.full((n_frames, 3), np.nan, dtype=np.float64)
        # Track which method produced each point for diagnostics
        n_fused = 0       # both cameras agreed → averaged
        n_on_only = 0     # on-axis back-projection only
        n_off_only = 0    # off-axis back-projection only
        n_disagree = 0    # both valid but disagreed → used higher confidence
        n_no_depth = 0    # track visible but no valid depth

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

            # Back-project from each camera
            pt_on = None
            pt_off_in_on = None  # off-axis point transformed to on-axis frame

            if on_ok:
                pt_on = _backproject(
                    float(on_track[fidx, 0]), float(on_track[fidx, 1]),
                    data.on_depth, fidx,
                    fx_on, fy_on, cx_on, cy_on,
                )

            if off_ok:
                pt_off_local = _backproject(
                    float(off_track[fidx, 0]), float(off_track[fidx, 1]),
                    data.off_depth, fidx,
                    fx_off, fy_off, cx_off, cy_off,
                )
                if pt_off_local is not None and T_off_to_on is not None:
                    pt_off_in_on = _transform_point(pt_off_local, T_off_to_on)

            # Fuse the two estimates
            if pt_on is not None and pt_off_in_on is not None:
                dist = float(np.linalg.norm(pt_on - pt_off_in_on))
                if dist <= _CROSS_CAM_AGREE_THRESHOLD:
                    # Agreement — visibility-weighted average
                    w_on = float(on_vis[fidx])
                    w_off = float(off_vis[fidx])
                    total_w = w_on + w_off
                    traj[fidx] = (pt_on * w_on + pt_off_in_on * w_off) / total_w
                    n_fused += 1
                else:
                    # Disagreement — use higher confidence estimate
                    if on_vis[fidx] >= off_vis[fidx]:
                        traj[fidx] = pt_on
                    else:
                        traj[fidx] = pt_off_in_on
                    n_disagree += 1
            elif pt_on is not None:
                traj[fidx] = pt_on
                n_on_only += 1
            elif pt_off_in_on is not None:
                traj[fidx] = pt_off_in_on
                n_off_only += 1
            elif on_ok or off_ok:
                n_no_depth += 1

            if on_progress and (fidx + 1) % 50 == 0:
                on_progress("pass4_triangulation", fidx + 1, n_frames, f"Back-projecting {label}")

        data.trajectories_3d[label] = traj

        valid_count = int(np.sum(~np.isnan(traj[:, 0])))
        logger.info(
            "%s: %d/%d valid — %d fused, %d on-only, %d off-only, "
            "%d disagreed, %d no-depth",
            label, valid_count, n_frames,
            n_fused, n_on_only, n_off_only, n_disagree, n_no_depth,
        )

        if valid_count > 0:
            valid_pts = traj[~np.isnan(traj[:, 0])]
            logger.info(
                "%s depth stats: min=%.3fm, max=%.3fm, median=%.3fm, std=%.4fm",
                label,
                valid_pts[:, 2].min(), valid_pts[:, 2].max(),
                np.median(valid_pts[:, 2]), np.std(valid_pts[:, 2]),
            )

    if on_progress:
        on_progress("pass4_triangulation", n_frames, n_frames, "Back-projection complete")
    logger.info("Pass 4 complete")
