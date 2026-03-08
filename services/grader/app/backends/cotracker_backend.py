"""Co-Tracker v2 point tracking backend.

Tracks user-specified query points (from tip_init.json) across video frames.
Each query point is (frame_idx, x, y). The model returns per-frame 2D
positions and visibility scores.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from app.backends.base import Detection, ModelBackend

logger = logging.getLogger("grader.backends.cotracker")


class CoTrackerBackend(ModelBackend):
    """Point tracking via Meta's Co-Tracker v2."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, path: str) -> None:
        try:
            import torch

            logger.info("Loading Co-Tracker v2 model from %s", path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # CoTracker v2 checkpoint may contain the model directly or
            # a state_dict wrapper. Handle both cases.
            if hasattr(checkpoint, "eval"):
                self._model = checkpoint
            else:
                # Try loading via cotracker API
                from cotracker.predictor import CoTrackerPredictor
                self._model = CoTrackerPredictor(checkpoint=path)

            if hasattr(self._model, "eval"):
                self._model.eval()
            if torch.cuda.is_available() and hasattr(self._model, "cuda"):
                self._model = self._model.cuda()
            logger.info("Co-Tracker v2 model loaded")
        except Exception as exc:
            logger.error("Failed to load Co-Tracker v2: %s", exc)
            raise

    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]:
        """Track instrument tips across frames using query points.

        Parameters
        ----------
        frames : list[np.ndarray]
            BGR video frames.
        query_points : np.ndarray, optional
            (N, 3) array of [frame_idx, x, y]. Each row defines a point
            to track from the given frame. Labels alternate: even indices
            are ``left_tip``, odd indices are ``right_tip``.

        Returns
        -------
        list[list[Detection]]
            Per-frame detections with visibility confidence.
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

            # Build queries tensor
            queries = None
            if query_points is not None and len(query_points) > 0:
                # query_points: (N, 3) with [frame_idx, x, y]
                queries = torch.from_numpy(query_points).float().unsqueeze(0)  # (1, N, 3)
                if torch.cuda.is_available():
                    queries = queries.cuda()

            with torch.no_grad():
                if queries is not None:
                    pred = self._model(video, queries=queries)
                else:
                    pred = self._model(video)

            # pred.tracks: (1, T, N, 2) -- N tracked points across T frames
            tracks = pred.tracks[0].cpu().numpy()  # (T, N, 2)
            visibility = pred.visibility[0].cpu().numpy()  # (T, N)
            n_points = tracks.shape[1]

            # Determine labels for each tracked point
            labels: list[str] = []
            if query_points is not None and len(query_points) > 0:
                for i in range(n_points):
                    labels.append("left_tip" if i % 2 == 0 else "right_tip")
            else:
                labels = [f"point_{i}" for i in range(n_points)]

            all_detections: list[list[Detection]] = []
            for t in range(tracks.shape[0]):
                detections: list[Detection] = []
                for p in range(n_points):
                    vis = float(visibility[t, p])
                    if vis > 0.5:
                        detections.append(
                            Detection(
                                x=float(tracks[t, p, 0]),
                                y=float(tracks[t, p, 1]),
                                confidence=vis,
                                label=labels[p],
                            )
                        )
                all_detections.append(detections)
                current = t + 1
                if on_progress and (current == tracks.shape[0] or current % 10 == 0):
                    on_progress(current, tracks.shape[0])

            logger.info(
                "Co-Tracker v2: tracked %d points across %d frames",
                n_points,
                len(frames),
            )
            return all_detections

        except Exception as exc:
            logger.warning("Co-Tracker v2 inference failed, returning empty: %s", exc)
            return [[] for _ in frames]

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
        logger.info("Co-Tracker v2 model unloaded")
