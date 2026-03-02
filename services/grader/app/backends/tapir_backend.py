"""TAPIR point tracking backend.

DeepMind's TAPIR (Tracking Any Point with per-frame Initialization and
temporal Refinement) tracks query points across video sequences.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.backends.base import Detection, ModelBackend

logger = logging.getLogger("grader.backends.tapir")


class TAPIRBackend(ModelBackend):
    """Point tracking via DeepMind's TAPIR."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, path: str) -> None:
        try:
            import torch

            logger.info("Loading TAPIR model from %s", path)
            checkpoint = torch.load(path, map_location="cpu")
            self._model = checkpoint
            logger.info("TAPIR model loaded")
        except Exception as exc:
            logger.error("Failed to load TAPIR: %s", exc)
            raise

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Track instrument tips across frames using TAPIR.

        Initializes query points at the center of the first frame and tracks
        them forward. Falls back to empty detections on failure.
        """
        if not frames or self._model is None:
            return [[] for _ in frames]

        try:
            import torch

            height, width = frames[0].shape[:2]

            # Stack frames into video tensor
            video = np.stack(frames).astype(np.float32) / 255.0
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)
            if torch.cuda.is_available():
                video_tensor = video_tensor.cuda()

            # Initialize two query points near typical instrument positions
            query_points = torch.tensor([
                [0, height * 0.5, width * 0.35],  # frame 0, left instrument
                [0, height * 0.5, width * 0.65],  # frame 0, right instrument
            ]).unsqueeze(0).float()
            if torch.cuda.is_available():
                query_points = query_points.cuda()

            with torch.no_grad():
                output = self._model(video_tensor, query_points)

            # output expected: tracks (1, T, N, 2), occlusion (1, T, N)
            tracks = output["tracks"][0].cpu().numpy()  # (T, N, 2)
            occlusion = output.get("occlusion", None)

            all_detections: list[list[Detection]] = []
            for t in range(tracks.shape[0]):
                detections: list[Detection] = []
                for p in range(tracks.shape[1]):
                    visible = True
                    if occlusion is not None:
                        visible = occlusion[0, t, p].item() < 0.5

                    if visible:
                        label = "left_tip" if p == 0 else "right_tip"
                        detections.append(Detection(
                            x=float(tracks[t, p, 0]),
                            y=float(tracks[t, p, 1]),
                            confidence=0.9,
                            label=label,
                        ))
                all_detections.append(detections)

            logger.info("TAPIR detections for %d frames", len(frames))
            return all_detections

        except Exception as exc:
            logger.warning("TAPIR inference failed, returning empty: %s", exc)
            return [[] for _ in frames]

    def unload(self) -> None:
        self._model = None
        logger.info("TAPIR model unloaded")
