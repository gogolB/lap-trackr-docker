"""Skill metrics calculated from 3D instrument-tip trajectories.

Each metric captures a different facet of laparoscopic dexterity:

* **workspace_volume** -- convex hull of all tip positions (cm^3).
* **avg_speed** -- mean tip speed across frames (mm/s).
* **max_jerk** -- peak jerk magnitude (mm/s^3), a smoothness indicator.
* **path_length** -- total distance travelled by both tips (mm).
* **economy_of_motion** -- ratio of direct (start-to-end) distance to
  actual path length (dimensionless, 0-1; higher is more efficient).
* **total_time** -- session duration in seconds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from app.config import DEFAULT_FPS

logger = logging.getLogger("grader.metrics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_points(
    poses: list[dict[str, Any]],
    tip_key: str,
) -> np.ndarray:
    """Gather valid 3D points for a given tip into an (N, 3) array."""

    pts = [p[tip_key] for p in poses if p[tip_key] is not None]
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def _path_length(points: np.ndarray) -> float:
    """Total Euclidean path length through an ordered sequence of points."""

    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _compute_speeds(points: np.ndarray, dt: float) -> np.ndarray:
    """Per-segment speed (distance / time)."""

    if len(points) < 2 or dt <= 0:
        return np.array([], dtype=np.float64)
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return distances / dt


def _compute_max_jerk(points: np.ndarray, dt: float) -> float:
    """Maximum jerk magnitude (third derivative of position)."""

    if len(points) < 4 or dt <= 0:
        return 0.0

    # velocity (first differences / dt)
    vel = np.diff(points, axis=0) / dt
    # acceleration
    acc = np.diff(vel, axis=0) / dt
    # jerk
    jerk = np.diff(acc, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk, axis=1)
    return float(np.max(jerk_mag))


def _economy(points: np.ndarray) -> float:
    """Ratio of direct start-to-end distance to total path length.

    Returns 1.0 for a perfectly straight path, lower for more wandering,
    and 0.0 when there is no movement or not enough data.
    """

    if len(points) < 2:
        return 0.0
    direct = float(np.linalg.norm(points[-1] - points[0]))
    total = _path_length(points)
    if total == 0:
        return 0.0
    return min(direct / total, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_metrics(
    poses: list[dict[str, Any]],
    fps: float | None = None,
) -> dict[str, float]:
    """Compute all skill metrics from 3D pose records.

    Parameters
    ----------
    poses : list[dict]
        Output of :func:`app.pose_estimator.estimate_poses`.
    fps : float, optional
        Frames per second of the *sampled* sequence (i.e. after
        ``sample_interval`` has been applied).  Used only for speed /
        jerk calculations.

    Returns
    -------
    dict[str, float]
        Keys: ``workspace_volume``, ``avg_speed``, ``max_jerk``,
        ``path_length``, ``economy_of_motion``, ``total_time``.
    """

    if fps is None or fps <= 0:
        fps = DEFAULT_FPS

    left_pts = _collect_points(poses, "left_tip")
    right_pts = _collect_points(poses, "right_tip")

    # --- workspace volume (cm^3) -------------------------------------------
    # Combine all tip positions for the convex hull.
    all_pts = np.concatenate(
        [p for p in (left_pts, right_pts) if len(p) > 0], axis=0
    ) if (len(left_pts) + len(right_pts)) > 0 else np.empty((0, 3))

    workspace_volume = 0.0
    if len(all_pts) >= 4:
        try:
            hull = ConvexHull(all_pts)
            # Points are in metres; convert volume to cm^3 (1 m^3 = 1e6 cm^3).
            workspace_volume = float(hull.volume) * 1e6
        except Exception as exc:
            logger.warning("ConvexHull failed: %s", exc)

    # --- time step between sampled frames -----------------------------------
    dt = 1.0 / fps  # seconds between consecutive sampled frames

    # --- total time ---------------------------------------------------------
    if len(poses) >= 2:
        total_time = poses[-1]["timestamp"] - poses[0]["timestamp"]
    else:
        total_time = 0.0

    # --- path length (mm) ---------------------------------------------------
    left_path = _path_length(left_pts) * 1000.0   # m -> mm
    right_path = _path_length(right_pts) * 1000.0
    path_length = left_path + right_path

    # --- average speed (mm/s) -----------------------------------------------
    left_speeds = _compute_speeds(left_pts, dt) * 1000.0
    right_speeds = _compute_speeds(right_pts, dt) * 1000.0
    all_speeds = np.concatenate([left_speeds, right_speeds]) if (
        len(left_speeds) + len(right_speeds)
    ) > 0 else np.array([0.0])
    avg_speed = float(np.mean(all_speeds))

    # --- max jerk (mm/s^3) --------------------------------------------------
    left_jerk = _compute_max_jerk(left_pts, dt) * 1000.0
    right_jerk = _compute_max_jerk(right_pts, dt) * 1000.0
    max_jerk = max(left_jerk, right_jerk)

    # --- economy of motion --------------------------------------------------
    left_econ = _economy(left_pts)
    right_econ = _economy(right_pts)
    economy_of_motion = (left_econ + right_econ) / 2.0

    metrics = {
        "workspace_volume": round(workspace_volume, 4),
        "avg_speed": round(avg_speed, 4),
        "max_jerk": round(max_jerk, 4),
        "path_length": round(path_length, 4),
        "economy_of_motion": round(economy_of_motion, 6),
        "total_time": round(total_time, 3),
    }

    logger.info("Calculated metrics: %s", metrics)
    return metrics
