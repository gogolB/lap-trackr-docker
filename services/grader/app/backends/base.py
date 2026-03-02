"""Abstract base class for detection model backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single 2D instrument-tip detection in pixel coordinates."""

    x: float
    y: float
    confidence: float
    label: str


class ModelBackend(abc.ABC):
    """Interface that every detection backend must implement."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from *path*."""

    @abc.abstractmethod
    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch of BGR frames.

        Returns one inner list per frame, each containing detections.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources."""
