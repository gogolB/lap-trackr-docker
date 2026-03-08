"""Pass 5 — RTS (Rauch-Tung-Striebel) trajectory smoothing (CPU only)."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from app.passes.pass_data import PassData

logger = logging.getLogger("grader.passes.pass5_smoothing")

# State: [x, y, z, vx, vy, vz]
_STATE_DIM = 6
_MEAS_DIM = 3

# Process noise scaling (mm^2 level for instrument dynamics)
_Q_POS_VARIANCE = 0.5e-6     # position process noise (m^2)
_Q_VEL_VARIANCE = 1.0e-4     # velocity process noise (m^2/s^2)

# Base measurement noise
_R_BASE = 1.0e-6              # base measurement noise (m^2)

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
    """Process noise matrix Q scaled by dt and gap penalty."""
    scale = 1.0 + gap_frames * _GAP_Q_MULTIPLIER
    Q = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)
    # Position block
    Q[0, 0] = _Q_POS_VARIANCE * dt * scale
    Q[1, 1] = _Q_POS_VARIANCE * dt * scale
    Q[2, 2] = _Q_POS_VARIANCE * dt * scale
    # Velocity block
    Q[3, 3] = _Q_VEL_VARIANCE * dt * scale
    Q[4, 4] = _Q_VEL_VARIANCE * dt * scale
    Q[5, 5] = _Q_VEL_VARIANCE * dt * scale
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
    """Apply RTS smoother to a single instrument trajectory.

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
            # Update
            z = trajectory[t]
            R = _build_measurement_noise(float(visibility[t]))
            y = z - H @ x_pred[t]
            S = H @ P_pred[t] @ H.T + R
            K = P_pred[t] @ H.T @ np.linalg.inv(S)
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

    return x_smooth[:, :3]


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
        # Use reprojection error as inverse proxy for visibility
        vis = None
        if label in data.on_visibility:
            vis = data.on_visibility[label]

        smoothed = _rts_smooth(traj, vis, dt)
        data.smoothed_3d[label] = smoothed

        valid_before = int(np.sum(~np.isnan(traj[:, 0])))
        valid_after = int(np.sum(~np.isnan(smoothed[:, 0])))
        logger.info(
            "%s: smoothed %d→%d valid frames",
            label, valid_before, valid_after,
        )

        if on_progress:
            on_progress("pass5_smoothing", idx + 1, total, f"Smoothed {label}")

    if on_progress:
        on_progress("pass5_smoothing", total, total, "Smoothing complete")
    logger.info("Pass 5 complete")
