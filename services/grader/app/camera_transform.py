"""Camera-config-driven image and calibration transforms."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("grader.camera_transform")

_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def get_camera_transform(
    camera_config: dict[str, Any] | None,
    camera_name: str,
) -> dict[str, Any]:
    """Return the transform settings for one camera."""
    if camera_name not in {"on_axis", "off_axis"}:
        raise ValueError(f"Unsupported camera name: {camera_name}")

    if not camera_config:
        return {
            "swap_eyes": False,
            "rotation": 0,
            "flip_h": False,
            "flip_v": False,
        }

    rotation = int(camera_config.get(f"{camera_name}_rotation", 0) or 0) % 360
    if rotation not in {0, 90, 180, 270}:
        logger.warning("Invalid %s rotation=%s, defaulting to 0", camera_name, rotation)
        rotation = 0

    return {
        "swap_eyes": bool(camera_config.get(f"{camera_name}_swap_eyes", False)),
        "rotation": rotation,
        "flip_h": bool(camera_config.get(f"{camera_name}_flip_h", False)),
        "flip_v": bool(camera_config.get(f"{camera_name}_flip_v", False)),
    }


def apply_transforms(image: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    """Apply rotation, horizontal flip, then vertical flip."""
    rotation = int(transform.get("rotation", 0) or 0)
    cv2_code = _ROTATION_MAP.get(rotation)
    if cv2_code is not None:
        image = cv2.rotate(image, cv2_code)
    if transform.get("flip_h", False):
        image = cv2.flip(image, 1)
    if transform.get("flip_v", False):
        image = cv2.flip(image, 0)
    return image


def transformed_dimensions(
    width: int,
    height: int,
    transform: dict[str, Any],
) -> tuple[int, int]:
    """Return output width/height after rotation."""
    rotation = int(transform.get("rotation", 0) or 0)
    if rotation in {90, 270}:
        return height, width
    return width, height


def adjust_intrinsics(
    intrinsics: dict[str, Any],
    transform: dict[str, Any],
) -> dict[str, Any]:
    """Adjust intrinsics to match the transformed image geometry."""
    adjusted = copy.deepcopy(intrinsics)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    width = int(intrinsics["image_width"])
    height = int(intrinsics["image_height"])
    rotation = int(transform.get("rotation", 0) or 0)

    if rotation == 90:
        fx, fy = fy, fx
        cx, cy = height - 1 - cy, cx
        width, height = height, width
    elif rotation == 180:
        cx = width - 1 - cx
        cy = height - 1 - cy
    elif rotation == 270:
        fx, fy = fy, fx
        cx, cy = cy, width - 1 - cx
        width, height = height, width

    if transform.get("flip_h", False):
        cx = width - 1 - cx
    if transform.get("flip_v", False):
        cy = height - 1 - cy

    adjusted.update(
        {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "image_width": int(width),
            "image_height": int(height),
        }
    )
    return adjusted


def adjust_calibration(
    calibration: dict[str, Any] | None,
    camera_config: dict[str, Any] | None,
    camera_name: str,
) -> dict[str, Any] | None:
    """Return a copy of calibration with transform-adjusted intrinsics."""
    if calibration is None:
        return None

    adjusted = copy.deepcopy(calibration)
    intrinsics = adjusted.get("intrinsics")
    if not isinstance(intrinsics, dict):
        return adjusted

    transform = get_camera_transform(camera_config, camera_name)
    adjusted["intrinsics"] = adjust_intrinsics(intrinsics, transform)
    return adjusted


def load_camera_config_from_session_dir(session_dir: Path) -> dict[str, Any] | None:
    """Load camera config captured in session_metadata.json, if available."""
    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        data = json.loads(metadata_path.read_text())
    except Exception as exc:
        logger.warning("Failed to read %s: %s", metadata_path, exc)
        return None

    camera_config = data.get("camera_config")
    if isinstance(camera_config, dict):
        return camera_config
    return None
