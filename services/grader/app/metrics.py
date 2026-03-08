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
_POSE_META_KEYS = {"frame_idx", "timestamp"}


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


def _workspace_volume(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume) * 1e6
    except Exception as exc:
        logger.warning("ConvexHull failed: %s", exc)
        return 0.0


def _tip_labels(poses: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for pose in poses:
        for key in pose:
            if key not in _POSE_META_KEYS:
                labels.add(key)
    return sorted(labels)


def _summarize_points(
    points: np.ndarray,
    dt: float,
    total_time: float,
) -> tuple[dict[str, float], np.ndarray]:
    path_length = _path_length(points) * 1000.0
    speeds = _compute_speeds(points, dt) * 1000.0
    avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    summary = {
        "workspace_volume": round(_workspace_volume(points), 4),
        "avg_speed": round(avg_speed, 4),
        "max_jerk": round(_compute_max_jerk(points, dt) * 1000.0, 4),
        "path_length": round(path_length, 4),
        "economy_of_motion": round(_economy(points), 6),
        "total_time": round(total_time, 3),
    }
    return summary, speeds


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

    tip_labels = _tip_labels(poses)

    dt = 1.0 / fps  # seconds between consecutive sampled frames

    if len(poses) >= 2:
        total_time = poses[-1]["timestamp"] - poses[0]["timestamp"]
    else:
        total_time = 0.0

    per_instrument: dict[str, dict[str, float]] = {}
    per_instrument_speeds: list[np.ndarray] = []
    all_point_sets: list[np.ndarray] = []
    per_instrument_jerk: list[float] = []
    per_instrument_economy: list[float] = []
    path_length = 0.0

    for label in tip_labels:
        points = _collect_points(poses, label)
        if len(points) > 0:
            all_point_sets.append(points)
        summary, speeds = _summarize_points(points, dt, total_time)
        per_instrument[label] = summary
        per_instrument_speeds.append(speeds)
        if len(points) > 0:
            per_instrument_jerk.append(summary["max_jerk"])
            per_instrument_economy.append(summary["economy_of_motion"])
        path_length += summary["path_length"]

    all_pts = (
        np.concatenate([points for points in all_point_sets if len(points) > 0], axis=0)
        if all_point_sets
        else np.empty((0, 3), dtype=np.float64)
    )
    all_speeds = (
        np.concatenate([speeds for speeds in per_instrument_speeds if len(speeds) > 0])
        if any(len(speeds) > 0 for speeds in per_instrument_speeds)
        else np.array([0.0], dtype=np.float64)
    )
    avg_speed = float(np.mean(all_speeds))
    max_jerk = max(per_instrument_jerk) if per_instrument_jerk else 0.0
    economy_of_motion = (
        float(np.mean(per_instrument_economy))
        if per_instrument_economy
        else 0.0
    )

    metrics = {
        "workspace_volume": round(_workspace_volume(all_pts), 4),
        "avg_speed": round(avg_speed, 4),
        "max_jerk": round(max_jerk, 4),
        "path_length": round(path_length, 4),
        "economy_of_motion": round(economy_of_motion, 6),
        "total_time": round(total_time, 3),
        "per_instrument": per_instrument,
    }

    logger.info("Calculated metrics: %s", metrics)
    return metrics
