"""CoTracker point tracking backend.

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
_TRACK_LABELS = ("green_tip", "pink_tip")


class CoTrackerBackend(ModelBackend):
    """Point tracking via Meta's CoTracker offline predictor."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, path: str) -> None:
        try:
            import torch

            logger.info("Loading CoTracker model from %s", path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Some checkpoints deserialize directly to a model, while others
            # require instantiating the predictor class from the CoTracker package.
            if hasattr(checkpoint, "eval"):
                self._model = checkpoint
            else:
                from cotracker.predictor import CoTrackerPredictor
                self._model = CoTrackerPredictor(checkpoint=path)

            if hasattr(self._model, "eval"):
                self._model.eval()
            if torch.cuda.is_available() and hasattr(self._model, "cuda"):
                self._model = self._model.cuda()
            logger.info("CoTracker model loaded")
        except Exception as exc:
            logger.error("Failed to load CoTracker model: %s", exc)
            raise

    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        query_labels: list[str] | tuple[str, ...] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]:
        """Track instrument tips across frames using query points.

        Parameters
        ----------
        frames : list[np.ndarray]
            BGR video frames.
        query_points : np.ndarray, optional
            (N, 3) array of [frame_idx, x, y]. Each row defines a point
            to track from the given frame. The first two query points map
            to ``green_tip`` and ``pink_tip`` respectively.

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

            tracks_tensor, visibility_tensor = _extract_prediction_tensors(pred)
            tracks = tracks_tensor[0].cpu().numpy()  # (T, N, 2)
            visibility = visibility_tensor[0].cpu().numpy()  # (T, N)
            n_points = tracks.shape[1]

            # Determine labels for each tracked point
            labels: list[str] = []
            if query_labels is not None and len(query_labels) >= n_points:
                labels = [str(query_labels[i]) for i in range(n_points)]
            elif query_points is not None and len(query_points) > 0:
                for i in range(n_points):
                    labels.append(
                        _TRACK_LABELS[i]
                        if i < len(_TRACK_LABELS)
                        else f"instrument_{i + 1}_tip"
                    )
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
                                source="cotracker",
                            )
                        )
                all_detections.append(detections)
                current = t + 1
                if on_progress and (current == tracks.shape[0] or current % 10 == 0):
                    on_progress(current, tracks.shape[0])

            logger.info(
                "CoTracker: tracked %d points across %d frames",
                n_points,
                len(frames),
            )
            return all_detections

        except Exception as exc:
            logger.warning("CoTracker inference failed, returning empty: %s", exc)
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
        logger.info("CoTracker model unloaded")


def _extract_prediction_tensors(pred: Any) -> tuple[Any, Any]:
    """Handle both legacy object-style and tuple-style CoTracker outputs."""

    if isinstance(pred, tuple) and len(pred) >= 2:
        return pred[0], pred[1]

    if isinstance(pred, dict):
        tracks = pred.get("tracks")
        visibility = pred.get("visibility")
        if tracks is not None and visibility is not None:
            return tracks, visibility

    tracks = getattr(pred, "tracks", None)
    visibility = getattr(pred, "visibility", None)
    if tracks is None or visibility is None:
        raise ValueError("Unsupported CoTracker output format")
    return tracks, visibility
