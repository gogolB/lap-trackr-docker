"""Placeholder backend — wraps the original random detector as a fallback."""

from __future__ import annotations

import logging

import numpy as np

from app.backends.base import Detection, ModelBackend

logger = logging.getLogger("grader.backends.placeholder")


class PlaceholderBackend(ModelBackend):
    """Deterministic pseudo-random detections for pipeline testing."""

    def load(self, path: str) -> None:
        logger.info("Placeholder backend loaded (no model file needed)")

    def detect(self, frames: list[np.ndarray], query_points: np.ndarray | None = None) -> list[list[Detection]]:
        if not frames:
            return []

        height, width = frames[0].shape[:2]
        all_detections: list[list[Detection]] = []

        for frame_idx in range(len(frames)):
            rng = np.random.RandomState(seed=frame_idx)
            margin_x = width * 0.2
            margin_y = height * 0.2

            left_x = rng.uniform(margin_x, width / 2)
            left_y = rng.uniform(margin_y, height - margin_y)
            right_x = rng.uniform(width / 2, width - margin_x)
            right_y = rng.uniform(margin_y, height - margin_y)

            detections = [
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

        logger.info("Placeholder detections for %d frames", len(frames))
        return all_detections

    def unload(self) -> None:
        pass
