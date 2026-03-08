"""YOLO pose backend with color-backed instrument identity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import hypot
from typing import Any, Callable

import numpy as np

from app.backends.base import Detection, ModelBackend
from app.color_detector import classify_tip_color

logger = logging.getLogger("grader.backends.yolo")

_TRACK_LABELS = ("green_tip", "pink_tip")
_TARGET_CLASS_IDS = {1}
_TARGET_CLASS_NAMES = {"lap_tool"}
_MAX_INSTRUMENTS = 2
_TIP_INDEX = 0
_CONF_THRESH = 0.45
_MAX_TRACK_MISSES = 45
_MAX_ASSIGNMENT_DISTANCE_RATIO = 0.20


@dataclass
class _TrackState:
    x: float
    y: float
    missed: int = 0


@dataclass(frozen=True)
class _RawTipDetection:
    x: float
    y: float
    confidence: float
    label: str | None
    color_score: float


class YOLOBackend(ModelBackend):
    """Instrument detection via Ultralytics YOLO pose."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, path: str) -> None:
        from ultralytics import YOLO

        logger.info("Loading YOLO model from %s", path)
        self._model = YOLO(path)
        logger.info("YOLO model loaded")

    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        query_labels: list[str] | tuple[str, ...] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]:
        if not frames or self._model is None:
            return [[] for _ in frames]

        track_states: dict[str, _TrackState] = {}
        all_detections: list[list[Detection]] = []
        total = len(frames)

        for idx, frame in enumerate(frames, start=1):
            results = self._model(frame, verbose=False)
            raw_detections = self._extract_raw_detections(results, frame)
            frame_detections = self._assign_labels(raw_detections, track_states, frame.shape[:2])
            self._update_track_states(track_states, frame_detections)
            all_detections.append(frame_detections)

            if on_progress and (idx == total or idx % 10 == 0):
                on_progress(idx, total)

        logger.info("YOLO pose detections for %d frames", len(frames))
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

    def _extract_raw_detections(
        self,
        results: list[Any],
        frame: np.ndarray,
    ) -> list[_RawTipDetection]:
        raw: list[_RawTipDetection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None or keypoints is None or len(boxes) == 0:
                continue

            names = getattr(result, "names", {}) or {}
            box_conf = boxes.conf.cpu().numpy()
            box_cls = boxes.cls.cpu().numpy().astype(int)
            kp_xy = keypoints.xy.cpu().numpy()
            kp_conf = (
                keypoints.conf.cpu().numpy()
                if getattr(keypoints, "conf", None) is not None
                else None
            )

            for det_idx in range(min(len(box_conf), len(kp_xy))):
                cls_id = int(box_cls[det_idx])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                if cls_id not in _TARGET_CLASS_IDS and cls_name not in _TARGET_CLASS_NAMES:
                    continue

                tip_point = kp_xy[det_idx][_TIP_INDEX]
                x = float(tip_point[0])
                y = float(tip_point[1])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue

                tip_conf = (
                    float(kp_conf[det_idx][_TIP_INDEX])
                    if kp_conf is not None
                    else float(box_conf[det_idx])
                )
                confidence = min(float(box_conf[det_idx]), tip_conf)
                if confidence < _CONF_THRESH:
                    continue

                label, color_score = classify_tip_color(frame, x, y)
                raw.append(
                    _RawTipDetection(
                        x=x,
                        y=y,
                        confidence=confidence,
                        label=label,
                        color_score=color_score,
                    )
                )

        raw.sort(key=lambda det: det.confidence, reverse=True)
        return raw[:_MAX_INSTRUMENTS]

    def _assign_labels(
        self,
        raw_detections: list[_RawTipDetection],
        track_states: dict[str, _TrackState],
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        assigned: dict[str, _RawTipDetection] = {}
        used_indices: set[int] = set()

        for label in _TRACK_LABELS:
            label_candidates = [
                (idx, det)
                for idx, det in enumerate(raw_detections)
                if det.label == label
            ]
            if not label_candidates:
                continue
            best_idx, best_det = max(
                label_candidates,
                key=lambda item: (item[1].color_score, item[1].confidence),
            )
            assigned[label] = best_det
            used_indices.add(best_idx)

        remaining = [
            det for idx, det in enumerate(raw_detections) if idx not in used_indices
        ]
        missing_labels = [label for label in _TRACK_LABELS if label not in assigned]

        if (
            len(remaining) == 1
            and len(missing_labels) == 1
            and remaining[0].label in (None, missing_labels[0])
        ):
            assigned[missing_labels[0]] = remaining[0]
            remaining = []
            missing_labels = []

        if remaining and missing_labels:
            self._assign_by_distance(
                remaining,
                missing_labels,
                track_states,
                frame_shape,
                assigned,
            )

        return [
            Detection(
                x=assigned[label].x,
                y=assigned[label].y,
                confidence=assigned[label].confidence,
                label=label,
                source="yolo",
            )
            for label in _TRACK_LABELS
            if label in assigned
        ]

    def _assign_by_distance(
        self,
        remaining: list[_RawTipDetection],
        missing_labels: list[str],
        track_states: dict[str, _TrackState],
        frame_shape: tuple[int, int],
        assigned: dict[str, _RawTipDetection],
    ) -> None:
        if not remaining or not missing_labels:
            return

        if len(remaining) == 1:
            best_label: str | None = None
            best_cost: float | None = None
            for label in missing_labels:
                cost = self._distance_cost(remaining[0], label, track_states, frame_shape)
                if cost is None:
                    continue
                if best_cost is None or cost < best_cost:
                    best_label = label
                    best_cost = cost
            if best_label is not None:
                assigned[best_label] = remaining[0]
            return

        if len(remaining) < 2 or len(missing_labels) < 2:
            return

        label_a, label_b = missing_labels[:2]
        det_a, det_b = remaining[:2]
        cost_direct = (
            self._distance_cost(det_a, label_a, track_states, frame_shape),
            self._distance_cost(det_b, label_b, track_states, frame_shape),
        )
        cost_swap = (
            self._distance_cost(det_a, label_b, track_states, frame_shape),
            self._distance_cost(det_b, label_a, track_states, frame_shape),
        )
        direct_total = self._pair_cost_total(cost_direct)
        swap_total = self._pair_cost_total(cost_swap)

        if direct_total is None and swap_total is None:
            return
        if swap_total is None or (direct_total is not None and direct_total <= swap_total):
            if cost_direct[0] is not None:
                assigned[label_a] = det_a
            if cost_direct[1] is not None:
                assigned[label_b] = det_b
            return

        if cost_swap[0] is not None:
            assigned[label_b] = det_a
        if cost_swap[1] is not None:
            assigned[label_a] = det_b

    def _distance_cost(
        self,
        det: _RawTipDetection,
        label: str,
        track_states: dict[str, _TrackState],
        frame_shape: tuple[int, int],
    ) -> float | None:
        state = track_states.get(label)
        if state is None or state.missed > _MAX_TRACK_MISSES:
            return None

        height, width = frame_shape
        gate = hypot(width, height) * _MAX_ASSIGNMENT_DISTANCE_RATIO * max(1.0, state.missed / 5 + 1.0)
        distance = hypot(det.x - state.x, det.y - state.y)
        if distance > gate:
            return None
        return distance

    @staticmethod
    def _pair_cost_total(costs: tuple[float | None, float | None]) -> float | None:
        if any(cost is None for cost in costs):
            return None
        return float(sum(costs))

    def _update_track_states(
        self,
        track_states: dict[str, _TrackState],
        detections: list[Detection],
    ) -> None:
        detected_by_label = {det.label: det for det in detections}
        for label in _TRACK_LABELS:
            det = detected_by_label.get(label)
            if det is not None:
                track_states[label] = _TrackState(x=det.x, y=det.y, missed=0)
                continue

            state = track_states.get(label)
            if state is not None:
                state.missed += 1
