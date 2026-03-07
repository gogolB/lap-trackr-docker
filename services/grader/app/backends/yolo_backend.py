"""YOLO backend using the ultralytics package."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.backends.base import Detection, ModelBackend

logger = logging.getLogger("grader.backends.yolo")


class YOLOBackend(ModelBackend):
    """Instrument detection via Ultralytics YOLO."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, path: str) -> None:
        from ultralytics import YOLO

        logger.info("Loading YOLO model from %s", path)
        self._model = YOLO(path)
        logger.info("YOLO model loaded")

    def detect(self, frames: list[np.ndarray], query_points: np.ndarray | None = None) -> list[list[Detection]]:
        if not frames or self._model is None:
            return [[] for _ in frames]

        all_detections: list[list[Detection]] = []

        for frame in frames:
            results = self._model(frame, verbose=False)
            detections: list[Detection] = []

            for result in results:
                for box in result.boxes:
                    x_center = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    y_center = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names.get(cls_id, f"class_{cls_id}")

                    detections.append(Detection(
                        x=x_center,
                        y=y_center,
                        confidence=conf,
                        label=label,
                    ))

            # If a custom surgical model returns labelled tips, use them.
            # Otherwise pick the top-2 detections as left/right tip proxies.
            tip_detections = [d for d in detections if "tip" in d.label]
            if not tip_detections and len(detections) >= 2:
                detections.sort(key=lambda d: d.confidence, reverse=True)
                tip_detections = [
                    Detection(x=detections[0].x, y=detections[0].y,
                              confidence=detections[0].confidence, label="left_tip"),
                    Detection(x=detections[1].x, y=detections[1].y,
                              confidence=detections[1].confidence, label="right_tip"),
                ]
            elif not tip_detections and detections:
                tip_detections = [
                    Detection(x=detections[0].x, y=detections[0].y,
                              confidence=detections[0].confidence, label="left_tip"),
                ]

            all_detections.append(tip_detections)

        logger.info("YOLO detections for %d frames", len(frames))
        return all_detections

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
        logger.info("YOLO model unloaded")
