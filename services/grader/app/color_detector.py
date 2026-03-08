"""HSV color thresholding for green/pink tape instrument tips."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

logger = logging.getLogger("grader.color_detector")

# HSV ranges for instrument tape colors (OpenCV 0-180 H scale)
# Corrected from 360° scale: Green H∈[168.4°–180.7°] → ~84–90,
# Pink H≈264.5° → ~132
_COLOR_RANGES = {
    "green": {
        "lower": np.array([80, 30, 50]),
        "upper": np.array([95, 255, 255]),
        "label": "green_tip",
    },
    "pink": {
        "lower": np.array([125, 25, 50]),
        "upper": np.array([145, 255, 255]),
        "label": "pink_tip",
    },
}

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MIN_CONTOUR_AREA = 120.0
MIN_BOUNDING_DIM = 6
MIN_EXTENT = 0.18
MIN_SOLIDITY = 0.65
MIN_DETECTION_CONFIDENCE = 0.45
PAIR_DISTANCE_MIN_RATIO = 0.01
PAIR_DISTANCE_MAX_RATIO = 0.55
TIP_PATCH_RADIUS = 18
MIN_COLOR_RATIO = 0.015
MIN_COLOR_MARGIN = 1.25


@dataclass(frozen=True)
class TipCandidate:
    label: str
    color: str
    x: float
    y: float
    confidence: float
    area: float

    def as_detection(self) -> dict:
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "color": self.color,
        }


def _build_mask(hsv_image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(hsv_image, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return mask


def _score_candidate(
    area: float,
    extent: float,
    solidity: float,
    mean_saturation: float,
    mean_value: float,
    img_area: int,
) -> float:
    area_reference = max(img_area * 0.00025, 300.0)
    area_score = min(1.0, area / area_reference)
    extent_score = min(1.0, extent / 0.55)
    solidity_score = min(1.0, solidity / 0.9)
    return float(
        0.45 * area_score
        + 0.20 * extent_score
        + 0.20 * solidity_score
        + 0.10 * mean_saturation
        + 0.05 * mean_value
    )


def _detect_color_candidate(
    hsv_image: np.ndarray,
    color: str,
    cfg: dict,
) -> TipCandidate | None:
    mask = _build_mask(hsv_image, cfg["lower"], cfg["upper"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = int(hsv_image.shape[0] * hsv_image.shape[1])
    min_area = max(MIN_CONTOUR_AREA, img_area * 0.00004)
    best: TipCandidate | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < MIN_BOUNDING_DIM:
            continue

        rect_area = float(w * h)
        if rect_area <= 0:
            continue
        extent = area / rect_area
        if extent < MIN_EXTENT:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            continue

        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        mean_saturation = float(cv2.mean(hsv_image[:, :, 1], mask=contour_mask)[0] / 255.0)
        mean_value = float(cv2.mean(hsv_image[:, :, 2], mask=contour_mask)[0] / 255.0)
        confidence = _score_candidate(
            area=area,
            extent=extent,
            solidity=solidity,
            mean_saturation=mean_saturation,
            mean_value=mean_value,
            img_area=img_area,
        )
        if confidence < MIN_DETECTION_CONFIDENCE:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        candidate = TipCandidate(
            label=cfg["label"],
            color=color,
            x=float(moments["m10"] / moments["m00"]),
            y=float(moments["m01"] / moments["m00"]),
            confidence=confidence,
            area=area,
        )
        if best is None or (candidate.confidence, candidate.area) > (best.confidence, best.area):
            best = candidate

    return best


def analyze_tip_frame(bgr_image: np.ndarray) -> dict:
    """Analyze a frame for plausible instrument tip detections."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    candidates: list[TipCandidate] = []

    for color, cfg in _COLOR_RANGES.items():
        candidate = _detect_color_candidate(hsv, color, cfg)
        if candidate is not None:
            candidates.append(candidate)

    detections = [candidate.as_detection() for candidate in sorted(candidates, key=lambda item: item.label)]
    score = float(sum(candidate.confidence for candidate in candidates))
    if len(candidates) == 2:
        distance = hypot(candidates[0].x - candidates[1].x, candidates[0].y - candidates[1].y)
        diagonal = hypot(bgr_image.shape[1], bgr_image.shape[0]) or 1.0
        distance_ratio = distance / diagonal
        if PAIR_DISTANCE_MIN_RATIO <= distance_ratio <= PAIR_DISTANCE_MAX_RATIO:
            score += 0.5
        else:
            score = max(0.0, score - 0.35)

    return {
        "detections": detections,
        "score": round(score, 3),
        "colors_found": len(candidates),
    }


def detect_tips(bgr_image: np.ndarray) -> list[dict]:
    """Detect green and pink tape tips in a BGR image."""
    analysis = analyze_tip_frame(bgr_image)
    detections = analysis["detections"]
    logger.info(
        "Detected %d tips with frame score %.3f: %s",
        len(detections),
        analysis["score"],
        [d["label"] for d in detections],
    )
    return detections


def classify_tip_color(
    bgr_image: np.ndarray,
    x: float,
    y: float,
    patch_radius: int = TIP_PATCH_RADIUS,
) -> tuple[str | None, float]:
    """Classify the tape color around a predicted tip point."""
    h, w = bgr_image.shape[:2]
    ix = int(round(x))
    iy = int(round(y))
    x0 = max(0, ix - patch_radius)
    x1 = min(w, ix + patch_radius + 1)
    y0 = max(0, iy - patch_radius)
    y1 = min(h, iy + patch_radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None, 0.0

    patch = bgr_image[y0:y1, x0:x1]
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    patch_area = max(1, patch.shape[0] * patch.shape[1])
    scores: dict[str, float] = {}
    for color, cfg in _COLOR_RANGES.items():
        mask = _build_mask(hsv_patch, cfg["lower"], cfg["upper"])
        scores[color] = float(cv2.countNonZero(mask) / patch_area)

    best_color, best_score = max(scores.items(), key=lambda item: item[1])
    second_score = max(
        (score for color, score in scores.items() if color != best_color),
        default=0.0,
    )
    if best_score < MIN_COLOR_RATIO:
        return None, best_score
    if second_score > 0 and best_score < second_score * MIN_COLOR_MARGIN:
        return None, best_score
    return _COLOR_RANGES[best_color]["label"], best_score
