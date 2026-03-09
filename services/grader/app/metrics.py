"""Skill metrics calculated from 3D instrument-tip trajectories.

Each metric captures a different facet of laparoscopic dexterity:

* **workspace_volume** -- convex hull of all tip positions (cm^3).
* **avg_speed** -- mean tip speed across frames (mm/s).
* **max_jerk** -- 95th-percentile jerk magnitude (mm/s^3), a smoothness
  indicator.  Uses P95 rather than absolute max for robustness against
  residual triangulation noise.
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

# MAD-based outlier rejection threshold (modified Z-score).
# 3.5 is a standard choice — keeps ~99.7% of normally-distributed data
# while rejecting extreme triangulation failures.
_OUTLIER_MAD_THRESHOLD = 3.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_points_with_times(
    poses: list[dict[str, Any]],
    tip_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather valid 3D points and their timestamps for a given tip.

    Returns (points, timestamps) where points is (N, 3) and timestamps is (N,).
    """
    pts: list[list[float]] = []
    times: list[float] = []
    for p in poses:
        if p.get(tip_key) is not None:
            pts.append(p[tip_key])
            times.append(p["timestamp"])
    if not pts:
        return np.empty((0, 3), dtype=np.float64), np.array([], dtype=np.float64)
    return np.array(pts, dtype=np.float64), np.array(times, dtype=np.float64)


def _filter_outliers_mad(
    points: np.ndarray,
    threshold: float = _OUTLIER_MAD_THRESHOLD,
) -> np.ndarray:
    """Return boolean mask of inlier points using MAD-based detection.

    For each coordinate (x, y, z), points whose modified Z-score exceeds
    *threshold* are marked as outliers.  MAD (Median Absolute Deviation) is
    robust to outliers unlike standard deviation — even if 30% of the data
    are wild triangulation failures, the median stays anchored to the true
    workspace.

    The modified Z-score is::

        0.6745 * |x_i - median(x)| / MAD(x)

    where 0.6745 normalizes MAD to be consistent with standard deviation
    for normally-distributed data.
    """
    n = len(points)
    if n < 10:
        return np.ones(n, dtype=bool)

    mask = np.ones(n, dtype=bool)
    for dim in range(points.shape[1]):
        vals = points[:, dim]
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad < 1e-10:
            continue
        modified_z = 0.6745 * np.abs(vals - median) / mad
        mask &= modified_z <= threshold

    return mask


def _workspace_volume(points: np.ndarray) -> float:
    """Convex hull volume in cm^3 (input in metres)."""
    if len(points) < 4:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume) * 1e6  # m^3 → cm^3
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


def _path_length(points: np.ndarray) -> float:
    """Total Euclidean path length through an ordered sequence of points (metres)."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _compute_speeds(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Per-segment speed (m/s) accounting for actual time gaps."""
    if len(points) < 2:
        return np.array([], dtype=np.float64)
    diffs = np.diff(points, axis=0)
    dt = np.diff(times)
    dt = np.maximum(dt, 1e-10)  # avoid division by zero
    distances = np.linalg.norm(diffs, axis=1)
    return distances / dt


def _compute_jerk_p95(points: np.ndarray, times: np.ndarray) -> float:
    """95th-percentile jerk magnitude (m/s^3).

    Uses P95 instead of absolute max because jerk involves the third
    finite difference divided by dt^3 — at 60 fps even tiny position noise
    produces extreme max-jerk values.  P95 captures real movement
    characteristics while being robust to residual noise.
    """
    if len(points) < 4:
        return 0.0

    dt = np.diff(times)
    dt = np.maximum(dt, 1e-10)

    # velocity: (N-1, 3)
    vel = np.diff(points, axis=0) / dt[:, np.newaxis]
    # time between velocity samples
    dt_v = (dt[:-1] + dt[1:]) / 2.0
    dt_v = np.maximum(dt_v, 1e-10)
    # acceleration: (N-2, 3)
    acc = np.diff(vel, axis=0) / dt_v[:, np.newaxis]
    if len(acc) < 2:
        return 0.0
    # time between acceleration samples
    dt_a = (dt_v[:-1] + dt_v[1:]) / 2.0
    dt_a = np.maximum(dt_a, 1e-10)
    # jerk: (N-3, 3)
    jerk = np.diff(acc, axis=0) / dt_a[:, np.newaxis]
    jerk_mag = np.linalg.norm(jerk, axis=1)
    if len(jerk_mag) == 0:
        return 0.0
    return float(np.percentile(jerk_mag, 95))


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


def _summarize_points(
    points: np.ndarray,
    times: np.ndarray,
    total_time: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute per-instrument summary metrics."""
    path_length = _path_length(points) * 1000.0  # m → mm
    speeds = _compute_speeds(points, times) * 1000.0  # m/s → mm/s
    avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    summary = {
        "workspace_volume": round(_workspace_volume(points), 4),
        "avg_speed": round(avg_speed, 4),
        "max_jerk": round(_compute_jerk_p95(points, times) * 1000.0, 4),  # m/s³ → mm/s³
        "path_length": round(path_length, 4),
        "economy_of_motion": round(_economy(points), 6),
        "total_time": round(total_time, 3),
    }
    return summary, speeds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_metrics(
    poses: list[dict[str, Any]],
    fps: float | None = None,
) -> dict[str, float]:
    """Compute all skill metrics from 3D pose records.

    Before computing any metrics, outlier 3D positions are rejected using
    MAD-based filtering (per-tip, per-coordinate).  This removes wild
    triangulation failures (e.g. 20 m depth spikes) that would otherwise
    dominate workspace volume, speed, and jerk.

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
        raw_points, raw_times = _collect_points_with_times(poses, label)

        # --- Outlier rejection ---
        if len(raw_points) >= 10:
            inlier_mask = _filter_outliers_mad(raw_points)
            points = raw_points[inlier_mask]
            times = raw_times[inlier_mask]
            n_removed = int(np.sum(~inlier_mask))
            if n_removed > 0:
                logger.info(
                    "Filtered %d / %d outlier points for %s "
                    "(%.1f%% removed)",
                    n_removed, len(raw_points), label,
                    100.0 * n_removed / len(raw_points),
                )
        else:
            points = raw_points
            times = raw_times

        if len(points) > 0:
            all_point_sets.append(points)

        summary, speeds = _summarize_points(points, times, total_time)
        per_instrument[label] = summary
        per_instrument_speeds.append(speeds)
        if len(points) > 0:
            per_instrument_jerk.append(summary["max_jerk"])
            per_instrument_economy.append(summary["economy_of_motion"])
        path_length += summary["path_length"]

    all_pts = (
        np.concatenate([pts for pts in all_point_sets if len(pts) > 0], axis=0)
        if all_point_sets
        else np.empty((0, 3), dtype=np.float64)
    )
    all_speeds = (
        np.concatenate([sp for sp in per_instrument_speeds if len(sp) > 0])
        if any(len(sp) > 0 for sp in per_instrument_speeds)
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
