"""Pass 3 — Adaptive color gap filling (CPU only, no GPU model)."""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from app.passes.pass_data import PassData
from app.passes.pass1_sam2 import _decode_rle

logger = logging.getLogger("grader.passes.pass3_color")

# Corrected default HSV ranges (OpenCV 0-180 H scale)
_DEFAULT_HSV = {
    "green_tip": {
        "lower": np.array([80, 30, 50]),
        "upper": np.array([95, 255, 255]),
    },
    "pink_tip": {
        "lower": np.array([125, 25, 50]),
        "upper": np.array([145, 255, 255]),
    },
}

_GAP_VISIBILITY_THRESHOLD = 0.4
_FILL_CONFIDENCE = 0.6
_MIN_CONTOUR_AREA = 80


def _build_adaptive_hsv_model(
    frames: list[np.ndarray],
    masks: dict[str, list[np.ndarray | None]],
    frame_shape: tuple[int, int],
) -> dict[str, dict[str, np.ndarray]]:
    """Build adaptive HSV ranges from SAM2 mask pixel statistics.

    For frames where a mask exists, sample HSV pixels within the mask
    and compute robust mean +/- 2 sigma for H, S, V.
    Falls back to defaults if insufficient data.
    """
    h, w = frame_shape
    models: dict[str, dict[str, np.ndarray]] = {}

    for label, mask_list in masks.items():
        h_values: list[float] = []
        s_values: list[float] = []
        v_values: list[float] = []

        # Sample up to 30 frames evenly
        valid_indices = [i for i, m in enumerate(mask_list) if m is not None]
        if not valid_indices:
            models[label] = _DEFAULT_HSV.get(label, {
                "lower": np.array([0, 30, 50]),
                "upper": np.array([180, 255, 255]),
            })
            continue

        n_sample = min(30, len(valid_indices))
        sample_indices = [valid_indices[i] for i in np.linspace(0, len(valid_indices) - 1, n_sample, dtype=int)]

        for fidx in sample_indices:
            if fidx >= len(frames):
                continue
            rle = mask_list[fidx]
            if rle is None:
                continue
            mask = _decode_rle(rle, (h, w))
            hsv = cv2.cvtColor(frames[fidx], cv2.COLOR_BGR2HSV)
            pixels = hsv[mask > 0]
            if len(pixels) == 0:
                continue
            # Sub-sample for speed
            if len(pixels) > 500:
                idx = np.random.choice(len(pixels), 500, replace=False)
                pixels = pixels[idx]
            h_values.extend(pixels[:, 0].astype(float))
            s_values.extend(pixels[:, 1].astype(float))
            v_values.extend(pixels[:, 2].astype(float))

        if len(h_values) < 20:
            # Insufficient data, use defaults
            models[label] = _DEFAULT_HSV.get(label, {
                "lower": np.array([0, 30, 50]),
                "upper": np.array([180, 255, 255]),
            })
            logger.info("Insufficient mask pixels for %s, using default HSV", label)
            continue

        h_arr = np.array(h_values)
        s_arr = np.array(s_values)
        v_arr = np.array(v_values)

        # Robust mean +/- 2 sigma
        lower = np.array([
            max(0, np.mean(h_arr) - 2 * np.std(h_arr)),
            max(0, np.mean(s_arr) - 2 * np.std(s_arr)),
            max(0, np.mean(v_arr) - 2 * np.std(v_arr)),
        ], dtype=np.float32)
        upper = np.array([
            min(180, np.mean(h_arr) + 2 * np.std(h_arr)),
            min(255, np.mean(s_arr) + 2 * np.std(s_arr)),
            min(255, np.mean(v_arr) + 2 * np.std(v_arr)),
        ], dtype=np.float32)

        models[label] = {
            "lower": lower.astype(np.uint8),
            "upper": upper.astype(np.uint8),
        }
        logger.info(
            "Adaptive HSV for %s: lower=%s upper=%s (from %d pixels)",
            label, lower.astype(int).tolist(), upper.astype(int).tolist(), len(h_values),
        )

    return models


def _fill_gaps_for_view(
    frames: list[np.ndarray],
    tracks: dict[str, np.ndarray],
    visibility: dict[str, np.ndarray],
    hsv_models: dict[str, dict[str, np.ndarray]],
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Fill visibility gaps using adaptive HSV color detection.

    Modifies tracks and visibility in-place. Returns number of frames filled.
    """
    filled = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    total_work = sum(
        np.sum(visibility.get(label, np.array([])) < _GAP_VISIBILITY_THRESHOLD)
        for label in tracks
    )
    progress_count = 0

    for label in list(tracks.keys()):
        track = tracks[label]
        vis = visibility[label]
        hsv_range = hsv_models.get(label, _DEFAULT_HSV.get(label))
        if hsv_range is None:
            continue

        lower = hsv_range["lower"]
        upper = hsv_range["upper"]

        for fidx in range(len(frames)):
            if vis[fidx] >= _GAP_VISIBILITY_THRESHOLD:
                continue

            progress_count += 1
            if on_progress and progress_count % 20 == 0:
                on_progress(progress_count, max(total_work, 1))

            hsv = cv2.cvtColor(frames[fidx], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Take largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < _MIN_CONTOUR_AREA:
                continue

            moments = cv2.moments(largest)
            if moments["m00"] == 0:
                continue

            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])

            track[fidx] = [cx, cy]
            vis[fidx] = _FILL_CONFIDENCE
            filled += 1

    if on_progress:
        on_progress(max(total_work, 1), max(total_work, 1))

    return filled


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Execute Pass 3: Adaptive color gap filling (CPU only)."""
    logger.info("Pass 3: Adaptive color gap filling starting")

    if on_progress:
        on_progress("pass3_color", 0, 4, "Building adaptive HSV models")

    # Build adaptive HSV models from SAM2 masks
    on_hsv = {}
    off_hsv = {}

    if data.on_masks and data.on_frames:
        on_shape = data.on_frames[0].shape[:2]
        on_hsv = _build_adaptive_hsv_model(data.on_frames, data.on_masks, on_shape)

    if data.off_masks and data.off_frames:
        off_shape = data.off_frames[0].shape[:2]
        off_hsv = _build_adaptive_hsv_model(data.off_frames, data.off_masks, off_shape)

    # Fill gaps in on-axis tracks
    if data.on_tracks and data.on_frames:
        if on_progress:
            on_progress("pass3_color", 1, 4, "Filling on-axis gaps")
        on_filled = _fill_gaps_for_view(
            data.on_frames,
            data.on_tracks,
            data.on_visibility,
            on_hsv,
            on_progress=lambda c, t: on_progress("pass3_color", c, t, "Color fill on-axis") if on_progress else None,
        )
        logger.info("On-axis: filled %d gap frames via color", on_filled)

    # Fill gaps in off-axis tracks
    if data.off_tracks and data.off_frames:
        if on_progress:
            on_progress("pass3_color", 3, 4, "Filling off-axis gaps")
        off_filled = _fill_gaps_for_view(
            data.off_frames,
            data.off_tracks,
            data.off_visibility,
            off_hsv,
            on_progress=lambda c, t: on_progress("pass3_color", c, t, "Color fill off-axis") if on_progress else None,
        )
        logger.info("Off-axis: filled %d gap frames via color", off_filled)

    if on_progress:
        on_progress("pass3_color", 4, 4, "Color gap filling complete")
    logger.info("Pass 3 complete")
