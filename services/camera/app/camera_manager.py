from __future__ import annotations

import os
import threading
import time

import cv2
import pyzed.sl as sl

from app.config import config


class CameraManager:
    """Manages two ZED X cameras identified by serial number.

    Cameras are labelled "on_axis" and "off_axis".  The class handles
    opening/closing, SVO2 recording with a background grab thread, and
    single-frame retrieval for MJPEG streaming.
    """

    def __init__(self) -> None:
        self.cameras: dict[str, sl.Camera] = {}
        self.recording: bool = False
        self.serials: dict[str, str | None] = {
            "on_axis": config.ZED_SERIAL_ON_AXIS,
            "off_axis": config.ZED_SERIAL_OFF_AXIS,
        }

        # Protects mutations to self.cameras, self._locks, self._latest_images
        self._camera_lock = threading.RLock()

        # Per-camera lock protects grab/retrieve races between the
        # background recording thread and the MJPEG streaming endpoint.
        self._locks: dict[str, threading.Lock] = {}

        # Latest captured sl.Mat per camera, written by the grab thread
        # and read by get_frame() during recording.
        self._latest_images: dict[str, sl.Mat] = {}

        # Pre-allocated sl.Mat for streaming/calibration retrieval
        self._streaming_mats: dict[str, sl.Mat] = {}

        # ZED SDK intrinsics extracted at open time
        self._intrinsics: dict[str, dict] = {}
        self._intrinsics_by_eye: dict[str, dict[str, dict]] = {}

        # Camera info (resolution, fps) extracted at open time
        self._camera_info: dict[str, dict] = {}

        # Camera config flags
        self._swap_eyes: dict[str, bool] = {"on_axis": False, "off_axis": False}
        self._rotation: dict[str, int] = {"on_axis": 0, "off_axis": 0}
        self._flip_h: dict[str, bool] = {"on_axis": False, "off_axis": False}
        self._flip_v: dict[str, bool] = {"on_axis": False, "off_axis": False}

        # Grab-thread control
        self._grab_thread: threading.Thread | None = None
        self._grab_stop_event = threading.Event()

        # Background retry loop for cameras that were not ready at startup.
        self._open_retry_thread: threading.Thread | None = None
        self._open_retry_stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_cameras(self) -> None:
        """Open configured cameras and keep retrying any missing devices."""
        self._attempt_open_missing_cameras()
        if self._missing_serials():
            self._start_open_retry_thread()

    def close(self) -> None:
        """Stop any active recording and close every camera."""
        self._stop_open_retry_thread()
        if self.recording:
            self.stop_recording()
        # Ensure grab thread is fully stopped before clearing state,
        # so the thread cannot access cleared dicts.
        self._stop_grab_thread()
        with self._camera_lock:
            for name, cam in self.cameras.items():
                print(f"[camera_manager] Closing {name}")
                cam.close()
            self.cameras.clear()
            self._locks.clear()
            self._latest_images.clear()
            self._streaming_mats.clear()
            self._intrinsics.clear()
            self._intrinsics_by_eye.clear()
            self._camera_info.clear()
        if self._grab_thread is not None and self._grab_thread.is_alive():
            print("[camera_manager] WARNING: grab thread still alive after close")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def apply_config(self, config: dict) -> None:
        """Apply camera configuration changes.

        Updates eye-swap and flip flags immediately. Serial changes
        require camera re-open (only done if serials actually changed).
        """
        self._swap_eyes = {
            "on_axis": config.get("on_axis_swap_eyes", False),
            "off_axis": config.get("off_axis_swap_eyes", False),
        }
        self._rotation = {
            "on_axis": config.get("on_axis_rotation", 0),
            "off_axis": config.get("off_axis_rotation", 0),
        }
        self._flip_h = {
            "on_axis": config.get("on_axis_flip_h", False),
            "off_axis": config.get("off_axis_flip_h", False),
        }
        self._flip_v = {
            "on_axis": config.get("on_axis_flip_v", False),
            "off_axis": config.get("off_axis_flip_v", False),
        }

        new_on = config.get("on_axis_serial", "")
        new_off = config.get("off_axis_serial", "")

        serials_changed = (
            new_on and new_off
            and (new_on != self.serials.get("on_axis") or new_off != self.serials.get("off_axis"))
        )

        if serials_changed and not self.recording:
            print(f"[camera_manager] Serial change detected, re-opening cameras")
            self.close()
            self.serials["on_axis"] = new_on
            self.serials["off_axis"] = new_off
            self.open_cameras()

        print(
            f"[camera_manager] Config applied: swap_eyes={self._swap_eyes}, "
            f"rotation={self._rotation}, flip_h={self._flip_h}, flip_v={self._flip_v}"
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self, session_dir: str) -> dict[str, str]:
        """Start SVO2 recording on every open camera.

        Parameters
        ----------
        session_dir:
            Absolute path to the directory where .svo2 files are written.

        Returns
        -------
        dict mapping ``"{name}_path"`` to the SVO2 file path for each camera.
        """
        if self.recording:
            raise RuntimeError("Already recording")

        if not self.cameras:
            raise RuntimeError("No cameras are open")

        os.makedirs(session_dir, exist_ok=True)

        paths: dict[str, str] = {}
        started: list[str] = []
        for name, cam in self.cameras.items():
            svo_path = os.path.join(session_dir, f"{name}.svo2")
            rec_params = sl.RecordingParameters()
            rec_params.video_filename = svo_path
            rec_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264
            err = cam.enable_recording(rec_params)
            if err != sl.ERROR_CODE.SUCCESS:
                # Roll back cameras that already started.
                for prev_name in started:
                    self.cameras[prev_name].disable_recording()
                raise RuntimeError(
                    f"Failed to start recording on {name}: {err}"
                )
            started.append(name)
            paths[f"{name}_path"] = svo_path

        self.recording = True
        self._start_grab_thread()
        return paths

    def stop_recording(self) -> None:
        """Stop recording on all cameras and join the grab thread."""
        self._stop_grab_thread()
        self._disable_all_recording()
        self.recording = False

    # ------------------------------------------------------------------
    # Background grab thread (feeds the SVO2 recorder)
    # ------------------------------------------------------------------

    def _start_grab_thread(self) -> None:
        self._grab_stop_event.clear()
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True, name="zed-grab"
        )
        self._grab_thread.start()

    def _stop_grab_thread(self) -> None:
        if self._grab_thread is not None and self._grab_thread.is_alive():
            self._grab_stop_event.set()
            self._grab_thread.join(timeout=5.0)
            self._grab_thread = None

    def _start_open_retry_thread(self) -> None:
        if self._open_retry_thread is not None and self._open_retry_thread.is_alive():
            return
        self._open_retry_stop_event.clear()
        self._open_retry_thread = threading.Thread(
            target=self._open_retry_loop,
            daemon=True,
            name="zed-open-retry",
        )
        self._open_retry_thread.start()

    def _stop_open_retry_thread(self) -> None:
        if self._open_retry_thread is not None and self._open_retry_thread.is_alive():
            self._open_retry_stop_event.set()
            self._open_retry_thread.join(timeout=5.0)
        self._open_retry_thread = None

    def _open_retry_loop(self) -> None:
        while not self._open_retry_stop_event.is_set():
            missing = self._missing_serials()
            if not missing:
                return
            print(
                "[camera_manager] Retrying missing cameras: "
                + ", ".join(f"{name}={serial}" for name, serial in missing.items())
            )
            self._attempt_open_missing_cameras()
            if not self._missing_serials():
                return
            self._open_retry_stop_event.wait(config.ZED_OPEN_RETRY_INTERVAL_S)

    def _grab_loop(self) -> None:
        """Continuously call ``grab()`` on every open camera.

        While recording, each successful grab feeds the SVO2 encoder.
        The latest left-eye image is also stored so that ``get_frame()``
        can return it without a separate grab call.
        """
        runtime = sl.RuntimeParameters()
        error_counts: dict[str, int] = {}
        while not self._grab_stop_event.is_set():
            with self._camera_lock:
                camera_snapshot = list(self.cameras.items())
            for name, cam in camera_snapshot:
                lock = self._locks.get(name)
                if lock is None:
                    continue
                image = self._latest_images.get(name)
                if image is None:
                    continue
                with lock:
                    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                        cam.retrieve_image(image, sl.VIEW.LEFT)
                        error_counts[name] = 0
                    else:
                        count = error_counts.get(name, 0) + 1
                        error_counts[name] = count
                        if count == 1 or count % 100 == 0:
                            print(f"[camera_manager] grab error on {name} (count={count})")
                        if count >= 500:
                            print(f"[camera_manager] CRITICAL: {name} exceeded 500 consecutive grab errors, stopping grab loop")
                            self._grab_stop_event.set()
                            return
            # Yield the CPU briefly so we don't spin at 100 %.
            # The ZED grab() itself blocks until the next frame is ready,
            # so this sleep is mainly a safety net.
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Frame retrieval (MJPEG streaming)
    # ------------------------------------------------------------------

    def get_frame(self, camera_name: str, eye: str = "left") -> bytes | None:
        """Return a JPEG-encoded frame from *camera_name*, or ``None``.

        *eye* must be ``"left"`` or ``"right"``.

        * During recording the grab thread owns ``grab()``, so we
          retrieve the requested view from the last grabbed frame.
        * When idle we call ``grab()`` ourselves (under lock).
        """
        cam = self.cameras.get(camera_name)
        if cam is None:
            return None

        lock = self._locks.get(camera_name)
        if lock is None:
            return None

        # Apply eye swap: if swap_eyes is set, flip the requested eye
        actual_eye = eye
        if self._swap_eyes.get(camera_name, False):
            actual_eye = "right" if eye == "left" else "left"

        view = sl.VIEW.RIGHT if actual_eye == "right" else sl.VIEW.LEFT
        image = self._streaming_mats.get(camera_name) or sl.Mat()

        if self.recording:
            with lock:
                cam.retrieve_image(image, view)
                data = image.get_data()
        else:
            with lock:
                runtime = sl.RuntimeParameters()
                if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    return None
                cam.retrieve_image(image, view)
                data = image.get_data()

        if data is None:
            return None

        data = self._apply_transforms(camera_name, data)

        ok, jpeg = cv2.imencode(
            ".jpg", data, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        if not ok:
            return None
        return jpeg.tobytes()

    # ------------------------------------------------------------------
    # Intrinsics / camera info
    # ------------------------------------------------------------------

    def get_intrinsics(self, camera_name: str) -> dict | None:
        """Return the ZED SDK intrinsics for a camera, or None."""
        per_eye = self._intrinsics_by_eye.get(camera_name)
        if per_eye:
            active_eye = "right" if self._swap_eyes.get(camera_name, False) else "left"
            intrinsics = per_eye.get(active_eye) or per_eye.get("left")
            if intrinsics is not None:
                return intrinsics
        return self._intrinsics.get(camera_name)

    def get_camera_info(self) -> dict:
        """Return per-camera info (serial, resolution, fps) and SDK version."""
        try:
            sdk_version = str(sl.Camera().get_sdk_version())
        except Exception:
            sdk_version = "unknown"
        return {
            "cameras": self._camera_info,
            "zed_sdk_version": sdk_version,
        }

    def capture_calibration_frame(self, camera_name: str) -> tuple[bytes | None, "np.ndarray | None"]:
        """Grab a single frame and return (jpeg_bytes, bgr_numpy_array).

        Used by the calibrator to get a frame for ChArUco detection.
        Applies eye-swap and flip config so downstream processing sees the
        correctly oriented image.
        Returns (None, None) if the camera is unavailable.
        """
        cam = self.cameras.get(camera_name)
        if cam is None:
            return None, None

        lock = self._locks.get(camera_name)
        if lock is None:
            return None, None

        # Apply eye swap
        view = sl.VIEW.LEFT
        if self._swap_eyes.get(camera_name, False):
            view = sl.VIEW.RIGHT

        image = self._streaming_mats.get(camera_name) or sl.Mat()

        if self.recording:
            with lock:
                cam.retrieve_image(image, view)
                data = image.get_data()
        else:
            with lock:
                runtime = sl.RuntimeParameters()
                if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    return None, None
                cam.retrieve_image(image, view)
                data = image.get_data()

        if data is None:
            return None, None

        data = self._apply_transforms(camera_name, data)

        # ZED SDK returns BGRA, convert to BGR for OpenCV
        bgr = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None, None
        return jpeg.tobytes(), bgr

    # ------------------------------------------------------------------
    # Discovery / status
    # ------------------------------------------------------------------

    def list_cameras(self) -> list[dict]:
        """Return information about all ZED cameras visible on the system."""
        devices = sl.Camera.get_device_list()
        result: list[dict] = []
        for dev in devices:
            result.append(
                {
                    "serial_number": dev.serial_number,
                    "camera_model": str(dev.camera_model),
                    "camera_state": str(dev.camera_state),
                }
            )
        return result

    def status(self) -> dict:
        """Return a status dict suitable for the ``/status`` endpoint."""
        cameras_info: dict[str, dict] = {}
        for name, serial in self.serials.items():
            opened = name in self.cameras
            cameras_info[name] = {
                "serial": serial,
                "opened": opened,
            }
        return {
            "recording": self.recording,
            "cameras": cameras_info,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _ROTATION_MAP = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def _apply_transforms(self, camera_name: str, image: "np.ndarray") -> "np.ndarray":
        """Apply configured rotation and flips to an image.

        Order: rotation first, then horizontal flip, then vertical flip.
        """
        degrees = self._rotation.get(camera_name, 0)
        cv2_code = self._ROTATION_MAP.get(degrees)
        if cv2_code is not None:
            image = cv2.rotate(image, cv2_code)
        if self._flip_h.get(camera_name, False):
            image = cv2.flip(image, 1)  # horizontal (left-right)
        if self._flip_v.get(camera_name, False):
            image = cv2.flip(image, 0)  # vertical (top-bottom)
        return image

    @staticmethod
    def _build_intrinsics(cam_params: "sl.CameraParameters", resolution: "sl.Resolution") -> dict:
        """Convert ZED SDK camera parameters into the calibration JSON shape."""
        return {
            "fx": float(cam_params.fx),
            "fy": float(cam_params.fy),
            "cx": float(cam_params.cx),
            "cy": float(cam_params.cy),
            "distortion": [float(d) for d in cam_params.disto],
            "image_width": int(resolution.width),
            "image_height": int(resolution.height),
        }

    def _disable_all_recording(self) -> None:
        for cam in self.cameras.values():
            cam.disable_recording()

    def _missing_serials(self) -> dict[str, str]:
        missing: dict[str, str] = {}
        for name, serial in self.serials.items():
            if not serial:
                continue
            if name not in self.cameras:
                missing[name] = serial
        return missing

    def _attempt_open_missing_cameras(self) -> None:
        missing = self._missing_serials()
        if not missing:
            return

        available_devices = self.list_cameras()
        if available_devices:
            print(f"[camera_manager] Visible ZED devices: {available_devices}")
        else:
            print("[camera_manager] Visible ZED devices: none")

        for name, serial in missing.items():
            if name in self.cameras:
                continue
            self._open_single_camera(name, serial)

    def _open_single_camera(self, name: str, serial: str) -> None:
        cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.set_from_serial_number(int(serial))
        status = cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(
                f"[camera_manager] Failed to open {name} "
                f"(serial {serial}): {status}"
            )
            try:
                cam.close()
            except Exception:
                pass
            return

        try:
            with self._camera_lock:
                self.cameras[name] = cam
                self._locks[name] = threading.Lock()
                self._latest_images[name] = sl.Mat()
                self._streaming_mats[name] = sl.Mat()

            # Extract factory-calibrated intrinsics from the ZED SDK
            try:
                cam_info = cam.get_camera_information()
                calib = cam_info.camera_configuration.calibration_parameters
                left = calib.left_cam
                right = calib.right_cam
                res = cam_info.camera_configuration.resolution
                left_intrinsics = self._build_intrinsics(left, res)
                right_intrinsics = self._build_intrinsics(right, res)
                self._intrinsics_by_eye[name] = {
                    "left": left_intrinsics,
                    "right": right_intrinsics,
                }
                self._intrinsics[name] = left_intrinsics
                self._camera_info[name] = {
                    "serial": serial,
                    "resolution": [int(res.width), int(res.height)],
                    "fps": 30,
                }
                print(
                    f"[camera_manager] {name} intrinsics: "
                    f"left=({left.fx:.1f}, {left.fy:.1f}, {left.cx:.1f}, {left.cy:.1f}) "
                    f"right=({right.fx:.1f}, {right.fy:.1f}, {right.cx:.1f}, {right.cy:.1f}) "
                    f"res={res.width}x{res.height}"
                )
            except Exception as exc:
                print(f"[camera_manager] Warning: could not extract intrinsics for {name}: {exc}")

            print(f"[camera_manager] Opened {name} (serial {serial})")
        except Exception as exc:
            print(f"[camera_manager] Error setting up {name}, closing: {exc}")
            cam.close()
            with self._camera_lock:
                self.cameras.pop(name, None)
                self._locks.pop(name, None)
                self._latest_images.pop(name, None)
                self._streaming_mats.pop(name, None)
            self._intrinsics.pop(name, None)
            self._intrinsics_by_eye.pop(name, None)
            self._camera_info.pop(name, None)
