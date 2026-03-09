"""Pass 5 — RTS (Rauch-Tung-Striebel) trajectory smoothing (CPU only).

Improvements over the initial version:

* **Realistic measurement noise** — R_BASE reflects actual ZED depth + triangulation
  uncertainty (~5–10 mm), not 1 μm.
* **Innovation gating** — measurements whose Mahalanobis distance exceeds a chi-squared
  threshold (3 DOF, 99 %) are rejected rather than absorbed.  This prevents outliers
  (CoTracker drift → DLT → wild 3D point) from corrupting the smoothed trajectory.
* **Velocity-aware process noise** — Q couples position and velocity via the standard
  piecewise-constant-acceleration model, giving physically meaningful uncertainty growth.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from app.passes.pass_data import PassData

logger = logging.getLogger("grader.passes.pass5_smoothing")

# State: [x, y, z, vx, vy, vz]
_STATE_DIM = 6
_MEAS_DIM = 3

# Process noise — standard deviation of *acceleration* assumed between frames.
# Surgical instruments: peak acceleration ~2 m/s², typical ~0.5 m/s².
_ACCEL_STD = 0.8  # m/s²

# Base measurement noise — standard deviation of triangulated 3D position.
# ZED stereo at 0.3–1 m depth: ~5–10 mm noise.  DLT triangulation adds more.
_R_BASE = 5.0e-5   # m²  (≈ 7 mm std dev)

# Innovation gating: chi-squared critical value for 3 DOF at 99 % confidence.
# Observations whose Mahalanobis distance exceeds this are rejected.
_CHI2_GATE = 11.345

# Gap penalty: multiply process noise by this factor per unobserved frame
_GAP_Q_MULTIPLIER = 5.0


def _build_transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition matrix F."""
    F = np.eye(_STATE_DIM, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def _build_process_noise(dt: float, gap_frames: int = 0) -> np.ndarray:
    """Piecewise-constant-acceleration process noise matrix Q.

    Assumes acceleration is white noise with std = _ACCEL_STD.
    This couples position and velocity uncertainty correctly:

        Q = G @ G^T @ σ_a²

    where G = [½dt², dt]^T per axis.
    """
    scale = 1.0 + gap_frames * _GAP_Q_MULTIPLIER
    q = (_ACCEL_STD ** 2) * scale

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt

    Q = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)
    for i in range(3):
        vi = i + 3
        Q[i, i] = dt4 / 4.0 * q     # pos-pos
        Q[i, vi] = dt3 / 2.0 * q     # pos-vel
        Q[vi, i] = dt3 / 2.0 * q     # vel-pos
        Q[vi, vi] = dt2 * q          # vel-vel
    return Q


def _build_measurement_matrix() -> np.ndarray:
    """Measurement matrix H: observe position only."""
    H = np.zeros((_MEAS_DIM, _STATE_DIM), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    return H


def _build_measurement_noise(visibility: float) -> np.ndarray:
    """Measurement noise R scaled by inverse visibility confidence."""
    # Lower visibility → higher noise
    conf = max(visibility, 0.1)
    r = _R_BASE / conf
    return np.eye(_MEAS_DIM, dtype=np.float64) * r


def _rts_smooth(
    trajectory: np.ndarray,
    visibility: np.ndarray | None,
    dt: float,
) -> np.ndarray:
    """Apply RTS smoother with innovation gating to a single instrument trajectory.

    Parameters
    ----------
    trajectory : (T, 3) array with NaN for missing frames
    visibility : (T,) confidence values, or None (defaults to 1.0 everywhere valid)
    dt : time between frames

    Returns
    -------
    smoothed : (T, 3) array
    """
    T = len(trajectory)
    if T < 2:
        return trajectory.copy()

    if visibility is None:
        visibility = np.where(np.isnan(trajectory[:, 0]), 0.0, 1.0)

    F = _build_transition_matrix(dt)
    H = _build_measurement_matrix()

    # Initialize state from first valid observation
    first_valid = None
    for i in range(T):
        if not np.any(np.isnan(trajectory[i])):
            first_valid = i
            break

    if first_valid is None:
        return trajectory.copy()

    # Forward Kalman pass
    x_pred = np.zeros((T, _STATE_DIM), dtype=np.float64)
    P_pred = np.zeros((T, _STATE_DIM, _STATE_DIM), dtype=np.float64)
    x_filt = np.zeros((T, _STATE_DIM), dtype=np.float64)
    P_filt = np.zeros((T, _STATE_DIM, _STATE_DIM), dtype=np.float64)

    # Initialize at first valid frame
    x_init = np.zeros(_STATE_DIM, dtype=np.float64)
    x_init[:3] = trajectory[first_valid]
    P_init = np.eye(_STATE_DIM, dtype=np.float64) * 1e-3

    x_filt[first_valid] = x_init
    P_filt[first_valid] = P_init

    # Fill frames before first_valid with the initial state
    for i in range(first_valid):
        x_filt[i] = x_init
        P_filt[i] = P_init * 10.0

    gap_count = 0
    n_gated = 0
    for t in range(first_valid + 1, T):
        # Predict
        observed = not np.any(np.isnan(trajectory[t]))
        if not observed:
            gap_count += 1
        else:
            gap_count = 0

        Q = _build_process_noise(dt, gap_count)
        x_pred[t] = F @ x_filt[t - 1]
        P_pred[t] = F @ P_filt[t - 1] @ F.T + Q

        if observed and visibility[t] > 0.1:
            z = trajectory[t]
            R = _build_measurement_noise(float(visibility[t]))
            y = z - H @ x_pred[t]  # innovation
            S = H @ P_pred[t] @ H.T + R  # innovation covariance

            # Innovation gating: Mahalanobis distance check
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            mahal_sq = float(y @ S_inv @ y)

            if mahal_sq > _CHI2_GATE:
                # Reject this measurement — treat as unobserved
                x_filt[t] = x_pred[t]
                P_filt[t] = P_pred[t]
                n_gated += 1
            else:
                # Standard Kalman update
                K = P_pred[t] @ H.T @ S_inv
                x_filt[t] = x_pred[t] + K @ y
                P_filt[t] = (np.eye(_STATE_DIM) - K @ H) @ P_pred[t]
        else:
            # No observation: prediction only
            x_filt[t] = x_pred[t]
            P_filt[t] = P_pred[t]

    # Backward RTS smoothing pass
    x_smooth = np.zeros((T, _STATE_DIM), dtype=np.float64)
    P_smooth = np.zeros((T, _STATE_DIM, _STATE_DIM), dtype=np.float64)
    x_smooth[T - 1] = x_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    for t in range(T - 2, -1, -1):
        if t < first_valid:
            x_smooth[t] = x_smooth[first_valid]
            P_smooth[t] = P_smooth[first_valid]
            continue

        P_pred_t1 = P_pred[t + 1]
        # Regularize to avoid singular matrix
        P_pred_inv = np.linalg.inv(P_pred_t1 + np.eye(_STATE_DIM) * 1e-10)
        C = P_filt[t] @ F.T @ P_pred_inv
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred_t1) @ C.T

    return x_smooth[:, :3], n_gated


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Execute Pass 5: RTS trajectory smoothing."""
    logger.info("Pass 5: RTS trajectory smoothing starting")

    if not data.trajectories_3d:
        logger.warning("No 3D trajectories to smooth")
        return

    dt = 1.0 / max(data.fps, 1.0)
    labels = sorted(data.trajectories_3d.keys())
    total = len(labels)

    if on_progress:
        on_progress("pass5_smoothing", 0, total, "Smoothing trajectories")

    for idx, label in enumerate(labels):
        traj = data.trajectories_3d[label]
        # Use CoTracker visibility as confidence proxy
        vis = None
        if label in data.on_visibility:
            vis = data.on_visibility[label]

        smoothed, n_gated = _rts_smooth(traj, vis, dt)
        data.smoothed_3d[label] = smoothed

        valid_before = int(np.sum(~np.isnan(traj[:, 0])))
        valid_after = int(np.sum(~np.isnan(smoothed[:, 0])))
        logger.info(
            "%s: smoothed %d→%d valid frames (%d measurements rejected by gating)",
            label, valid_before, valid_after, n_gated,
        )

        if on_progress:
            on_progress("pass5_smoothing", idx + 1, total, f"Smoothed {label}")

    if on_progress:
        on_progress("pass5_smoothing", total, total, "Smoothing complete")
    logger.info("Pass 5 complete")
