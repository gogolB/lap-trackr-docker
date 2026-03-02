"""Placeholder instrument-tip detector.

This module will eventually wrap an ONNX / TensorRT model that localises
laparoscopic instrument tips in each video frame.  For now it returns
deterministic pseudo-random detections so that the rest of the pipeline
can be developed and tested end-to-end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger("grader.detector")


@dataclass(frozen=True)
class Detection:
    """A single 2D instrument-tip detection in pixel coordinates."""

    x: float
    y: float
    confidence: float
    label: str


def detect_instruments(
    frames: list[np.ndarray],
) -> list[list[Detection]]:
    """Detect instrument tips in a batch of video frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        BGR images, each shaped ``(H, W, 3)``.

    Returns
    -------
    list[list[Detection]]
        One inner list per frame, each containing detections for that frame.
        The placeholder always returns exactly two detections per frame
        (``"left_tip"`` and ``"right_tip"``).
    """

    if not frames:
        return []

    height, width = frames[0].shape[:2]
    all_detections: list[list[Detection]] = []

    for frame_idx in range(len(frames)):
        # Seed from frame index so results are reproducible.
        rng = np.random.RandomState(seed=frame_idx)

        # Simulate smooth instrument motion by keeping tips in a plausible
        # region of the image (central 60 %).
        margin_x = width * 0.2
        margin_y = height * 0.2

        left_x = rng.uniform(margin_x, width / 2)
        left_y = rng.uniform(margin_y, height - margin_y)

        right_x = rng.uniform(width / 2, width - margin_x)
        right_y = rng.uniform(margin_y, height - margin_y)

        detections: list[Detection] = [
            Detection(
                x=float(left_x),
                y=float(left_y),
                confidence=float(rng.uniform(0.85, 0.99)),
                label="left_tip",
            ),
            Detection(
                x=float(right_x),
                y=float(right_y),
                confidence=float(rng.uniform(0.85, 0.99)),
                label="right_tip",
            ),
        ]
        all_detections.append(detections)

    logger.info(
        "Detected instruments in %d frames (placeholder mode)", len(frames)
    )
    return all_detections
