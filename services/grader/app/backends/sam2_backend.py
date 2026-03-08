"""SAM 2 segmentation backend.

Uses SAM 2 video segmentation to produce instrument masks, then computes
tip centroids from the mask contours.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from app.backends.base import Detection, ModelBackend

logger = logging.getLogger("grader.backends.sam2")


class SAM2Backend(ModelBackend):
    """Instrument tip detection via SAM 2 mask segmentation."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, path: str) -> None:
        try:
            import torch

            logger.info("Loading SAM 2 model from %s", path)
            self._model = torch.load(path, map_location="cpu")
            if hasattr(self._model, "eval"):
                self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            logger.info("SAM 2 model loaded")
        except Exception as exc:
            logger.error("Failed to load SAM 2: %s", exc)
            raise

    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]:
        """Segment instruments and derive tip positions from mask extrema.

        Falls back to empty detections if inference fails.
        """
        if not frames or self._model is None:
            return [[] for _ in frames]

        try:
            import torch

            all_detections: list[list[Detection]] = []
            total = len(frames)
            for idx, frame in enumerate(frames, start=1):
                # Convert to tensor: (1, 3, H, W)
                tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
                if torch.cuda.is_available():
                    tensor = tensor.cuda()

                with torch.no_grad():
                    masks = self._model(tensor)  # Expected: (1, N, H, W)

                if masks is not None and masks.numel() > 0:
                    mask = masks[0, 0].cpu().numpy() > 0.5
                    detections = self._tips_from_mask(mask)
                else:
                    detections = []

                all_detections.append(detections)
                if on_progress and (idx == total or idx % 10 == 0):
                    on_progress(idx, total)

            logger.info("SAM 2 detections for %d frames", len(frames))
            return all_detections

        except Exception as exc:
            logger.warning("SAM 2 inference failed, returning empty: %s", exc)
            return [[] for _ in frames]

    @staticmethod
    def _tips_from_mask(mask: np.ndarray) -> list[Detection]:
        """Extract instrument tip positions from a binary mask.

        Uses the topmost/bottommost or leftmost/rightmost extrema of
        connected components as proxies for instrument tips.
        """
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []

        # Leftmost and rightmost points of the mask as tip proxies
        left_idx = np.argmin(xs)
        right_idx = np.argmax(xs)

        return [
            Detection(
                x=float(xs[left_idx]),
                y=float(ys[left_idx]),
                confidence=0.85,
                label="left_tip",
            ),
            Detection(
                x=float(xs[right_idx]),
                y=float(ys[right_idx]),
                confidence=0.85,
                label="right_tip",
            ),
        ]

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("SAM 2 model unloaded")
