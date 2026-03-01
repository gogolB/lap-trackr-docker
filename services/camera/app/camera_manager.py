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

        # Per-camera lock protects grab/retrieve races between the
        # background recording thread and the MJPEG streaming endpoint.
        self._locks: dict[str, threading.Lock] = {}

        # Latest captured sl.Mat per camera, written by the grab thread
        # and read by get_frame() during recording.
        self._latest_images: dict[str, sl.Mat] = {}

        # Grab-thread control
        self._grab_thread: threading.Thread | None = None
        self._grab_stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_cameras(self) -> None:
        """Open every configured camera.  Called once at startup."""
        for name, serial in self.serials.items():
            if not serial:
                print(f"[camera_manager] {name} serial not configured, skipping")
                continue
            cam = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30
            init_params.depth_mode = sl.DEPTH_MODE.NONE
            init_params.set_from_serial_number(int(serial))
            status = cam.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(
                    f"[camera_manager] Failed to open {name} "
                    f"(serial {serial}): {status}"
                )
                continue
            self.cameras[name] = cam
            self._locks[name] = threading.Lock()
            self._latest_images[name] = sl.Mat()
            print(f"[camera_manager] Opened {name} (serial {serial})")

    def close(self) -> None:
        """Stop any active recording and close every camera."""
        if self.recording:
            self.stop_recording()
        for name, cam in self.cameras.items():
            print(f"[camera_manager] Closing {name}")
            cam.close()
        self.cameras.clear()
        self._locks.clear()
        self._latest_images.clear()

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

    def _grab_loop(self) -> None:
        """Continuously call ``grab()`` on every open camera.

        While recording, each successful grab feeds the SVO2 encoder.
        The latest left-eye image is also stored so that ``get_frame()``
        can return it without a separate grab call.
        """
        runtime = sl.RuntimeParameters()
        while not self._grab_stop_event.is_set():
            for name, cam in self.cameras.items():
                lock = self._locks.get(name)
                if lock is None:
                    continue
                with lock:
                    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                        cam.retrieve_image(
                            self._latest_images[name], sl.VIEW.LEFT
                        )
            # Yield the CPU briefly so we don't spin at 100 %.
            # The ZED grab() itself blocks until the next frame is ready,
            # so this sleep is mainly a safety net.
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Frame retrieval (MJPEG streaming)
    # ------------------------------------------------------------------

    def get_frame(self, camera_name: str) -> bytes | None:
        """Return a JPEG-encoded frame from *camera_name*, or ``None``.

        * During recording the grab thread owns ``grab()``, so we just
          read the most recently retrieved image.
        * When idle we call ``grab()`` ourselves (under lock).
        """
        cam = self.cameras.get(camera_name)
        if cam is None:
            return None

        lock = self._locks.get(camera_name)
        if lock is None:
            return None

        image = self._latest_images.get(camera_name)
        if image is None:
            image = sl.Mat()
            self._latest_images[camera_name] = image

        if self.recording:
            # The grab thread keeps _latest_images up to date.
            with lock:
                data = image.get_data()
        else:
            # Not recording -- we drive grab() from here.
            with lock:
                runtime = sl.RuntimeParameters()
                if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    return None
                cam.retrieve_image(image, sl.VIEW.LEFT)
                data = image.get_data()

        if data is None:
            return None

        ok, jpeg = cv2.imencode(
            ".jpg", data, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        if not ok:
            return None
        return jpeg.tobytes()

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

    def _disable_all_recording(self) -> None:
        for cam in self.cameras.values():
            cam.disable_recording()
