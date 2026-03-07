"""HSV color thresholding for green/pink tape instrument tips."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("grader.color_detector")

# HSV ranges for instrument tape colors
_COLOR_RANGES = {
    "green": {
        "lower": np.array([35, 50, 50]),
        "upper": np.array([85, 255, 255]),
        "label": "left_tip",
    },
    "pink": {
        "lower": np.array([140, 50, 50]),
        "upper": np.array([175, 255, 255]),
        "label": "right_tip",
    },
}

MIN_CONTOUR_AREA = 50


def detect_tips(bgr_image: np.ndarray) -> list[dict]:
    """Detect green and pink tape tips in a BGR image.

    Returns a list of dicts with keys: label, x, y, confidence, color.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    results: list[dict] = []

    for color, cfg in _COLOR_RANGES.items():
        mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Filter by area, take largest
        valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        if not valid:
            continue

        largest = max(valid, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Confidence based on area relative to image size
        img_area = bgr_image.shape[0] * bgr_image.shape[1]
        confidence = min(1.0, area / (img_area * 0.01))

        results.append({
            "label": cfg["label"],
            "x": float(cx),
            "y": float(cy),
            "confidence": float(confidence),
            "color": color,
        })

    logger.info("Detected %d tips: %s", len(results), [r["label"] for r in results])
    return results
