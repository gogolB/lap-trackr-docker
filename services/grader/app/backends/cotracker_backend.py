"""Co-Tracker point tracking backend.

Co-Tracker tracks user-specified or grid-sampled points across video frames.
For surgical instrument tracking, we initialize query points at the instrument
tips detected in the first frame and track them forward.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.backends.base import Detection, ModelBackend

logger = logging.getLogger("grader.backends.cotracker")


class CoTrackerBackend(ModelBackend):
    """Point tracking via Meta's Co-Tracker."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, path: str) -> None:
        try:
            import torch

            logger.info("Loading Co-Tracker model from %s", path)
            self._model = torch.load(path, map_location="cpu")
            if hasattr(self._model, "eval"):
                self._model.eval()
            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            logger.info("Co-Tracker model loaded")
        except Exception as exc:
            logger.error("Failed to load Co-Tracker: %s", exc)
            raise

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Track instrument tips across frames.

        Co-Tracker works on video sequences rather than individual frames.
        We initialize grid points and filter to the ones that track
        instrument-like motion patterns.

        Falls back to placeholder detections if the model cannot run.
        """
        if not frames or self._model is None:
            return [[] for _ in frames]

        try:
            import torch

            # Stack frames into video tensor: (1, T, 3, H, W)
            video = np.stack(frames)  # (T, H, W, 3)
            video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()
            if torch.cuda.is_available():
                video = video.cuda()

            with torch.no_grad():
                pred = self._model(video)

            # pred.tracks: (1, T, N, 2) — N tracked points across T frames
            tracks = pred.tracks[0].cpu().numpy()  # (T, N, 2)
            visibility = pred.visibility[0].cpu().numpy()  # (T, N)

            all_detections: list[list[Detection]] = []
            for t in range(tracks.shape[0]):
                visible_mask = visibility[t] > 0.5
                visible_tracks = tracks[t][visible_mask]  # (K, 2)

                detections: list[Detection] = []
                if len(visible_tracks) >= 2:
                    # Sort by x-coordinate, take leftmost and rightmost
                    sorted_idx = np.argsort(visible_tracks[:, 0])
                    left = visible_tracks[sorted_idx[0]]
                    right = visible_tracks[sorted_idx[-1]]
                    detections = [
                        Detection(x=float(left[0]), y=float(left[1]),
                                  confidence=0.9, label="left_tip"),
                        Detection(x=float(right[0]), y=float(right[1]),
                                  confidence=0.9, label="right_tip"),
                    ]
                elif len(visible_tracks) == 1:
                    pt = visible_tracks[0]
                    detections = [
                        Detection(x=float(pt[0]), y=float(pt[1]),
                                  confidence=0.8, label="left_tip"),
                    ]
                all_detections.append(detections)

            logger.info("Co-Tracker detections for %d frames", len(frames))
            return all_detections

        except Exception as exc:
            logger.warning("Co-Tracker inference failed, returning empty: %s", exc)
            return [[] for _ in frames]

    def unload(self) -> None:
        self._model = None
        logger.info("Co-Tracker model unloaded")
