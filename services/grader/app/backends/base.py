"""Abstract base class for detection model backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single 2D instrument-tip detection in pixel coordinates."""

    x: float
    y: float
    confidence: float
    label: str
    source: str = "unknown"


class ModelBackend(abc.ABC):
    """Interface that every detection backend must implement."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from *path*."""

    @abc.abstractmethod
    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        query_labels: list[str] | tuple[str, ...] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]:
        """Run detection on a batch of BGR frames.

        Parameters
        ----------
        frames : list[np.ndarray]
            BGR images.
        query_points : np.ndarray, optional
            (N, 3) array of [frame_idx, x, y] initialization points.
        on_progress : callable, optional
            ``(current, total)`` callback for batch progress updates.

        Returns one inner list per frame, each containing detections.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources."""
