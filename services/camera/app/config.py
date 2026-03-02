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


config = Config()
