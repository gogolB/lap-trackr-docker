"""Mock camera manager for local development without ZED cameras.

Generates test-pattern frames so the full stack (API, frontend, grader)
can run on any machine (macOS, Linux x86) without the ZED SDK.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from app.config import config


class MockCameraManager:
    """Drop-in replacement for CameraManager that produces synthetic frames."""

    def __init__(self) -> None:
        self.cameras: dict[str, bool] = {}
        self.recording: bool = False
        self.serials: dict[str, str | None] = {
            "on_axis": config.ZED_SERIAL_ON_AXIS or "MOCK-00001",
            "off_axis": config.ZED_SERIAL_OFF_AXIS or "MOCK-00002",
        }
        self._intrinsics: dict[str, dict] = {}
        self._camera_info: dict[str, dict] = {}
        self._frame_counter = 0
        self._swap_eyes: dict[str, bool] = {"on_axis": False, "off_axis": False}
        self._flip: dict[str, bool] = {"on_axis": False, "off_axis": False}
        self._camera_fps: int = getattr(config, "CAMERA_TARGET_FPS_DEFAULT", 60)
        self._whitebalance_auto: dict[str, bool] = {"on_axis": True, "off_axis": True}
        self._whitebalance_temperature: dict[str, int] = {
            "on_axis": getattr(config, "WHITEBALANCE_TEMPERATURE_DEFAULT", 4600),
            "off_axis": getattr(config, "WHITEBALANCE_TEMPERATURE_DEFAULT", 4600),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_cameras(self) -> None:
        for name, serial in self.serials.items():
            self.cameras[name] = True
            self._intrinsics[name] = {
                "fx": 1065.0,
                "fy": 1065.0,
                "cx": 960.0,
                "cy": 600.0,
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "image_width": 1920,
                "image_height": 1200,
            }
            self._camera_info[name] = {
                "serial": serial,
                "resolution": [1920, 1200],
                "fps": self._camera_fps,
            }
            print(f"[mock_camera] Opened {name} (serial {serial}) -- MOCK MODE")

    def close(self) -> None:
        self.cameras.clear()
        self._intrinsics.clear()
        self._camera_info.clear()
        self.recording = False
        print("[mock_camera] Closed all cameras")

    # ------------------------------------------------------------------
    # Recording (stubs -- creates empty placeholder files)
    # ------------------------------------------------------------------

    def start_recording(self, session_dir: str) -> dict[str, str]:
        if self.recording:
            raise RuntimeError("Already recording")
        if not self.cameras:
            raise RuntimeError("No cameras are open")

        os.makedirs(session_dir, exist_ok=True)
        paths: dict[str, str] = {}
        for name in self.cameras:
            svo_path = os.path.join(session_dir, f"{name}.svo2")
            # Create an empty placeholder file
            with open(svo_path, "wb") as f:
                f.write(b"MOCK_SVO2")
            paths[f"{name}_path"] = svo_path

        self.recording = True
        return paths

    def stop_recording(self) -> None:
        self.recording = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def apply_config(self, config: dict) -> None:
        """Apply camera configuration changes (mock)."""
        self._swap_eyes = {
            "on_axis": config.get("on_axis_swap_eyes", False),
            "off_axis": config.get("off_axis_swap_eyes", False),
        }
        self._camera_fps = int(config.get("camera_fps", self._camera_fps))
        self._whitebalance_auto = {
            "on_axis": bool(config.get("on_axis_whitebalance_auto", True)),
            "off_axis": bool(config.get("off_axis_whitebalance_auto", True)),
        }
        self._whitebalance_temperature = {
            "on_axis": int(config.get("on_axis_whitebalance_temperature", self._whitebalance_temperature["on_axis"])),
            "off_axis": int(config.get("off_axis_whitebalance_temperature", self._whitebalance_temperature["off_axis"])),
        }
        self._flip = {
            "on_axis": config.get("on_axis_flip", False),
            "off_axis": config.get("off_axis_flip", False),
        }

        new_on = config.get("on_axis_serial", "")
        new_off = config.get("off_axis_serial", "")
        if new_on:
            self.serials["on_axis"] = new_on
        if new_off:
            self.serials["off_axis"] = new_off

        print(
            f"[mock_camera] Config applied: swap_eyes={self._swap_eyes}, "
            f"flip={self._flip}, serials={self.serials}, fps={self._camera_fps}, "
            f"wb_auto={self._whitebalance_auto}, wb_temp={self._whitebalance_temperature}"
        )

    # ------------------------------------------------------------------
    # Frame retrieval (generates a test pattern)
    # ------------------------------------------------------------------

    def get_frame(self, camera_name: str, eye: str = "left") -> bytes | None:
        if camera_name not in self.cameras:
            return None

        actual_eye = eye
        if self._swap_eyes.get(camera_name, False):
            actual_eye = "right" if eye == "left" else "left"

        self._frame_counter += 1
        frame = self._generate_test_frame(camera_name, actual_eye)

        if self._flip.get(camera_name, False):
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        return jpeg.tobytes()

    def _generate_test_frame(
        self, camera_name: str, eye: str, width: int = 1920, height: int = 1200
    ) -> np.ndarray:
        """Generate a test-pattern frame with camera/eye label."""
        # Dark gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Vertical gradient
        for y in range(height):
            val = int(30 + 20 * (y / height))
            frame[y, :] = (val, val + 5, val + 10)

        # Grid lines
        for x in range(0, width, 120):
            cv2.line(frame, (x, 0), (x, height), (50, 60, 70), 1)
        for y in range(0, height, 120):
            cv2.line(frame, (0, y), (width, y), (50, 60, 70), 1)

        # Center crosshair
        cx, cy = width // 2, height // 2
        cv2.line(frame, (cx - 40, cy), (cx + 40, cy), (0, 200, 200), 2)
        cv2.line(frame, (cx, cy - 40), (cx, cy + 40), (0, 200, 200), 2)

        # Labels
        label = f"{camera_name} / {eye}"
        cv2.putText(
            frame, label, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 200), 3,
        )
        cv2.putText(
            frame, "MOCK MODE", (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 200), 2,
        )

        # Animated element (moving dot)
        t = time.time()
        dot_x = int(cx + 300 * np.sin(t * 0.5))
        dot_y = int(cy + 200 * np.cos(t * 0.7))
        cv2.circle(frame, (dot_x, dot_y), 15, (0, 255, 100), -1)

        return frame

    # ------------------------------------------------------------------
    # Intrinsics / camera info
    # ------------------------------------------------------------------

    def get_intrinsics(self, camera_name: str) -> dict | None:
        return self._intrinsics.get(camera_name)

    def get_camera_info(self) -> dict:
        return {
            "cameras": self._camera_info,
            "zed_sdk_version": "mock",
        }

    def get_stream_interval_seconds(self, camera_name: str) -> float:
        fps = max(1, int(self._camera_fps))
        return 1.0 / fps

    def capture_calibration_frame(self, camera_name: str) -> tuple[bytes | None, np.ndarray | None]:
        if camera_name not in self.cameras:
            return None, None

        bgr = self._generate_test_frame(camera_name, "left")
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None, None
        return jpeg.tobytes(), bgr

    # ------------------------------------------------------------------
    # Discovery / status
    # ------------------------------------------------------------------

    def list_cameras(self) -> list[dict]:
        return [
            {"serial_number": s, "camera_model": "MOCK", "camera_state": "AVAILABLE"}
            for s in self.serials.values()
        ]

    def status(self) -> dict:
        cameras_info: dict[str, dict] = {}
        for name, serial in self.serials.items():
            cameras_info[name] = {
                "serial": serial,
                "opened": name in self.cameras,
                "fps": self._camera_fps,
            }
        return {
            "recording": self.recording,
            "cameras": cameras_info,
        }
