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
                "fps": 30,
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
    # Frame retrieval (generates a test pattern)
    # ------------------------------------------------------------------

    def get_frame(self, camera_name: str, eye: str = "left") -> bytes | None:
        if camera_name not in self.cameras:
            return None

        self._frame_counter += 1
        frame = self._generate_test_frame(camera_name, eye)
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
            }
        return {
            "recording": self.recording,
            "cameras": cameras_info,
        }
