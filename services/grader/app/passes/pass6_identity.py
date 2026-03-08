"""Pass 6 — Identity verification via color consistency check (CPU only)."""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from app.passes.pass_data import PassData

logger = logging.getLogger("grader.passes.pass6_identity")

# Corrected HSV ranges for identity verification (OpenCV 0-180 H scale)
_EXPECTED_HSV = {
    "green_tip": {
        "lower": np.array([80, 30, 50]),
        "upper": np.array([95, 255, 255]),
    },
    "pink_tip": {
        "lower": np.array([125, 25, 50]),
        "upper": np.array([145, 255, 255]),
    },
}

_SAMPLE_FRAMES = 20
_PATCH_RADIUS = 20
_MIN_COLOR_RATIO = 0.15
_SWAP_THRESHOLD = 0.6  # If >60% of sampled frames show wrong color, swap


def _check_color_consistency(
    frames: list[np.ndarray],
    tracks: dict[str, np.ndarray],
    visibility: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """For each instrument track, check what fraction of sampled frames
    have the expected HSV color at the tracked position.

    Returns {label: {"correct": ratio, "swapped": ratio}}.
    """
    results: dict[str, dict[str, float]] = {}
    n_frames = len(frames)

    for label in tracks:
        track = tracks[label]
        vis = visibility.get(label, np.ones(len(track)))

        # Select visible frames to sample
        visible_indices = [i for i in range(min(len(track), n_frames)) if vis[i] > 0.3 and not np.any(np.isnan(track[i]))]
        if not visible_indices:
            results[label] = {"correct": 0.0, "swapped": 0.0}
            continue

        n_sample = min(_SAMPLE_FRAMES, len(visible_indices))
        sample_indices = [visible_indices[i] for i in np.linspace(0, len(visible_indices) - 1, n_sample, dtype=int)]

        correct_count = 0
        swapped_count = 0
        # Determine the "other" label for swap detection
        other_label = "pink_tip" if label == "green_tip" else "green_tip"

        for fidx in sample_indices:
            x, y = float(track[fidx, 0]), float(track[fidx, 1])
            h, w = frames[fidx].shape[:2]
            ix, iy = int(round(x)), int(round(y))

            x0 = max(0, ix - _PATCH_RADIUS)
            x1 = min(w, ix + _PATCH_RADIUS + 1)
            y0 = max(0, iy - _PATCH_RADIUS)
            y1 = min(h, iy + _PATCH_RADIUS + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            patch = frames[fidx][y0:y1, x0:x1]
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            patch_area = max(1, patch.shape[0] * patch.shape[1])

            # Check expected color
            expected_range = _EXPECTED_HSV.get(label)
            if expected_range:
                mask = cv2.inRange(hsv_patch, expected_range["lower"], expected_range["upper"])
                ratio = float(cv2.countNonZero(mask)) / patch_area
                if ratio >= _MIN_COLOR_RATIO:
                    correct_count += 1

            # Check swapped color
            other_range = _EXPECTED_HSV.get(other_label)
            if other_range:
                mask = cv2.inRange(hsv_patch, other_range["lower"], other_range["upper"])
                ratio = float(cv2.countNonZero(mask)) / patch_area
                if ratio >= _MIN_COLOR_RATIO:
                    swapped_count += 1

        total = len(sample_indices)
        results[label] = {
            "correct": correct_count / max(total, 1),
            "swapped": swapped_count / max(total, 1),
        }

    return results


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Execute Pass 6: Identity verification."""
    logger.info("Pass 6: Identity verification starting")

    if on_progress:
        on_progress("pass6_identity", 0, 2, "Checking color consistency")

    # Check on-axis tracks
    on_results = {}
    if data.on_tracks and data.on_frames:
        on_results = _check_color_consistency(
            data.on_frames, data.on_tracks, data.on_visibility,
        )
        logger.info("On-axis color consistency: %s", on_results)

    if on_progress:
        on_progress("pass6_identity", 1, 2, "Checking off-axis consistency")

    # Check off-axis tracks
    off_results = {}
    if data.off_tracks and data.off_frames:
        off_results = _check_color_consistency(
            data.off_frames, data.off_tracks, data.off_visibility,
        )
        logger.info("Off-axis color consistency: %s", off_results)

    # Determine if swap is needed
    # A swap is detected if BOTH instruments show higher swapped ratio than correct
    swap_detected = False
    labels = ["green_tip", "pink_tip"]
    for results_dict, view_name in [(on_results, "on-axis"), (off_results, "off-axis")]:
        swap_votes = 0
        for label in labels:
            r = results_dict.get(label, {})
            if r.get("swapped", 0) > r.get("correct", 0) and r.get("swapped", 0) >= _SWAP_THRESHOLD:
                swap_votes += 1
        if swap_votes == len(labels):
            logger.warning("Identity swap detected in %s view!", view_name)
            swap_detected = True
            break

    if swap_detected:
        logger.info("Applying identity swap: green_tip <-> pink_tip")
        data.swap_map = {"green_tip": "pink_tip", "pink_tip": "green_tip"}

        # Swap tracks
        for tracks_dict in [data.on_tracks, data.off_tracks]:
            if "green_tip" in tracks_dict and "pink_tip" in tracks_dict:
                tracks_dict["green_tip"], tracks_dict["pink_tip"] = (
                    tracks_dict["pink_tip"],
                    tracks_dict["green_tip"],
                )
        for vis_dict in [data.on_visibility, data.off_visibility]:
            if "green_tip" in vis_dict and "pink_tip" in vis_dict:
                vis_dict["green_tip"], vis_dict["pink_tip"] = (
                    vis_dict["pink_tip"],
                    vis_dict["green_tip"],
                )

        # Swap 3D trajectories
        for traj_dict in [data.trajectories_3d, data.smoothed_3d]:
            if "green_tip" in traj_dict and "pink_tip" in traj_dict:
                traj_dict["green_tip"], traj_dict["pink_tip"] = (
                    traj_dict["pink_tip"],
                    traj_dict["green_tip"],
                )
        if "green_tip" in data.reprojection_errors and "pink_tip" in data.reprojection_errors:
            data.reprojection_errors["green_tip"], data.reprojection_errors["pink_tip"] = (
                data.reprojection_errors["pink_tip"],
                data.reprojection_errors["green_tip"],
            )
    else:
        data.swap_map = None

    data.identity_verified = True

    if on_progress:
        on_progress("pass6_identity", 2, 2, "Identity verification complete")
    logger.info("Pass 6 complete — swap_detected=%s", swap_detected)
