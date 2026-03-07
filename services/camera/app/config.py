from __future__ import annotations

import os
from typing import Optional


class Config:
    """Configuration sourced from environment variables."""

    ZED_SERIAL_ON_AXIS: Optional[str] = os.environ.get("ZED_SERIAL_ON_AXIS")
    ZED_SERIAL_OFF_AXIS: Optional[str] = os.environ.get("ZED_SERIAL_OFF_AXIS")
    DATA_DIR: str = os.environ.get("DATA_DIR", "/data")
    CAMERA_HOST: str = os.environ.get("CAMERA_HOST", "0.0.0.0")
    CAMERA_PORT: int = int(os.environ.get("CAMERA_PORT", "8001"))

    # ChArUco board calibration
    CHARUCO_ROWS: int = int(os.environ.get("CHARUCO_ROWS", "14"))
    CHARUCO_COLS: int = int(os.environ.get("CHARUCO_COLS", "9"))
    CHARUCO_SQUARE_SIZE_MM: float = float(os.environ.get("CHARUCO_SQUARE_SIZE_MM", "20.0"))
    CHARUCO_MARKER_SIZE_MM: float = float(os.environ.get("CHARUCO_MARKER_SIZE_MM", "15.0"))
    CHARUCO_DICT: str = os.environ.get("CHARUCO_DICT", "DICT_5X5_50")
    CALIBRATION_DIR: str = os.environ.get("CALIBRATION_DIR", "/data/calibration")


config = Config()
