"""SVO2 → MP4 (left/right eyes) + NPZ (depth frames) exporter.

Converts ZED proprietary SVO2 recordings into portable formats for
offline playback and analysis on machines without the ZED SDK.
"""

from __future__ import annotations

import io
import logging
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger("grader.exporter")
_TARGET_BITRATE = 20_000_000


class ExportCancelledError(RuntimeError):
    """Raised when an export job is cancelled mid-run."""


def export_svo2(
    svo2_path: str,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Export a single SVO2 file to MP4 (left/right eyes) + NPZ (depth).

    Parameters
    ----------
    svo2_path : str
        Absolute path to the ``.svo2`` file.

    Returns
    -------
    dict
        ``{"mp4_path": ..., "npz_path": ..., "frame_count": int}``
    """
    import pyzed.sl as sl

    path = Path(svo2_path)
    session_dir = path.parent
    cam_name = path.stem  # "on_axis" or "off_axis"

    left_mp4_path = session_dir / f"{cam_name}_left.mp4"
    right_mp4_path = session_dir / f"{cam_name}_right.mp4"
    npz_path = session_dir / f"{cam_name}_depth.npz"

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(svo2_path)
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO2 '{svo2_path}': {status}")

    total_frames = zed.get_svo_number_of_frames()
    if total_frames <= 0:
        total_frames = 0

    left_writer = None
    right_writer = None
    depth_zip = None
    success = False
    try:
        info = zed.get_camera_information()
        w = int(info.camera_configuration.resolution.width)
        h = int(info.camera_configuration.resolution.height)
        fps = int(info.camera_configuration.fps)
        if fps <= 0:
            fps = 30

        left_writer = _create_writer(left_mp4_path, fps, w, h)
        right_writer = _create_writer(right_mp4_path, fps, w, h)
        depth_zip = zipfile.ZipFile(
            npz_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )

        left_image = sl.Mat()
        right_image = sl.Mat()
        depth = sl.Mat()

        frame_idx = 0

        runtime = sl.RuntimeParameters()

        while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            if should_cancel and should_cancel():
                raise ExportCancelledError(f"Export cancelled for session file '{svo2_path}'")

            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            left_bgr = left_image.get_data()[:, :, :3].copy()
            right_bgr = right_image.get_data()[:, :, :3].copy()
            left_writer.write(left_bgr)
            right_writer.write(right_bgr)

            depth_arr = depth.get_data().copy().astype(np.float32)
            _write_depth_frame(depth_zip, frame_idx, depth_arr)
            frame_idx += 1

            if frame_idx % 10 == 0:
                if on_progress:
                    on_progress(frame_idx, total_frames)
                logger.info("Progress: %d / %d frames", frame_idx, total_frames)

        if on_progress:
            on_progress(frame_idx, max(total_frames, frame_idx, 1))

        depth_zip.close()
        depth_zip = None

        left_writer.release()
        left_writer = None
        right_writer.release()
        right_writer = None

        logger.info(
            "Exported %s: %d frames → %s + %s",
            svo2_path,
            frame_idx,
            left_mp4_path,
            npz_path,
        )

        # Extract sample frames from the left-eye video for tip initialization.
        sample_paths = _extract_sample_frames(str(left_mp4_path), session_dir, cam_name, frame_idx)

        result = {
            "mp4_path": str(left_mp4_path),
            "left_mp4_path": str(left_mp4_path),
            "right_mp4_path": str(right_mp4_path),
            "mp4_paths": [str(left_mp4_path), str(right_mp4_path)],
            "npz_path": str(npz_path),
            "frame_count": frame_idx,
            "sample_paths": sample_paths,
        }
        success = True
        return result
    finally:
        release_errors: list[str] = []
        if depth_zip is not None:
            try:
                depth_zip.close()
            except Exception as exc:
                release_errors.append(f"depth archive: {exc}")
                logger.warning("Failed to close depth archive", exc_info=True)
        if left_writer is not None:
            try:
                left_writer.release()
            except Exception as exc:
                release_errors.append(f"left writer: {exc}")
                logger.warning("Failed to release left VideoWriter", exc_info=True)
        if right_writer is not None:
            try:
                right_writer.release()
            except Exception as exc:
                release_errors.append(f"right writer: {exc}")
                logger.warning("Failed to release right VideoWriter", exc_info=True)
        try:
            zed.close()
        except Exception:
            logger.warning("Failed to close ZED camera", exc_info=True)
        if not success:
            _safe_unlink(left_mp4_path)
            _safe_unlink(right_mp4_path)
            _safe_unlink(npz_path)
        if release_errors and sys.exc_info()[0] is None:
            raise RuntimeError("Failed to finalize export writers: " + "; ".join(release_errors))


def _extract_sample_frames(
    mp4_path: str, session_dir: Path, cam_name: str, total_frames: int
) -> list[str]:
    """Extract 3 sample frames (first, middle, last) as JPEG from an MP4."""
    if total_frames < 1:
        return []

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        logger.warning("Cannot open %s for sample frame extraction", mp4_path)
        return []

    try:
        # Frame indices: first, middle, last
        indices = [0]
        if total_frames > 1:
            indices.append(total_frames // 2)
        if total_frames > 2:
            indices.append(total_frames - 1)

        sample_paths: list[str] = []
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            if not ret:
                continue
            filename = f"{cam_name}_sample_{i}.jpg"
            out_path = session_dir / filename
            cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            sample_paths.append(str(out_path))

        logger.info("Extracted %d sample frames for %s", len(sample_paths), cam_name)
        return sample_paths
    finally:
        cap.release()


def _write_depth_frame(depth_zip: zipfile.ZipFile, frame_idx: int, depth_arr: np.ndarray) -> None:
    """Append one depth frame to the final NPZ archive without compression."""
    buffer = io.BytesIO()
    np.save(buffer, depth_arr, allow_pickle=False)
    depth_zip.writestr(f"frame_{frame_idx:06d}.npy", buffer.getvalue())


def _create_writer(path: Path, fps: int, width: int, height: int) -> "_FrameWriter":
    """Create a Jetson hardware writer with software fallback."""
    try:
        writer = _GStreamerNvencWriter(path, fps, width, height)
        logger.info("Using Jetson GStreamer NVENC for %s", path)
        return writer
    except Exception as exc:
        logger.warning(
            "Jetson GStreamer NVENC unavailable for %s, falling back to OpenCV software encode: %s",
            path,
            exc,
        )
        _safe_unlink(path)

    return _OpenCvFrameWriter(path, fps, width, height)


class _FrameWriter:
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class _OpenCvFrameWriter(_FrameWriter):
    """Software encoder fallback via OpenCV VideoWriter."""

    def __init__(self, path: Path, fps: int, width: int, height: int) -> None:
        self._path = path
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            logger.info("avc1 codec unavailable for %s, falling back to mp4v", path)
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to create VideoWriter for '{path}': both avc1 and mp4v codecs failed"
            )
        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


class _GStreamerNvencWriter(_FrameWriter):
    """Jetson hardware H.264 encoder driven through gst-launch."""

    def __init__(self, path: Path, fps: int, width: int, height: int) -> None:
        self._path = path
        self._width = width
        self._height = height
        self._frame_bytes = width * height * 3
        framerate = f"{max(int(round(fps)), 1)}/1"
        interval = max(int(round(fps)), 1)
        cmd = [
            "gst-launch-1.0",
            "-q",
            "fdsrc",
            "fd=0",
            f"blocksize={self._frame_bytes}",
            "!",
            "rawvideoparse",
            "format=bgr",
            f"width={width}",
            f"height={height}",
            f"framerate={framerate}",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            f"bitrate={_TARGET_BITRATE}",
            "insert-sps-pps=true",
            f"iframeinterval={interval}",
            f"idrinterval={interval}",
            "!",
            "h264parse",
            "!",
            "qtmux",
            "faststart=true",
            "!",
            "filesink",
            f"location={path}",
            "sync=false",
            "async=false",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdin is None:
            self._terminate()
            raise RuntimeError("Failed to open stdin for gst-launch pipeline")
        if self._process.poll() is not None:
            raise RuntimeError(self._failure_message("gst-launch exited immediately"))

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self._height, self._width):
            raise ValueError(
                f"Frame size mismatch for {self._path}: got {frame.shape[1]}x{frame.shape[0]}, "
                f"expected {self._width}x{self._height}"
            )
        if self._process.poll() is not None:
            raise RuntimeError(self._failure_message("gst-launch pipeline stopped during encode"))

        data = np.ascontiguousarray(frame, dtype=np.uint8)
        try:
            self._process.stdin.write(data.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self._failure_message("Broken pipe while writing frame")) from exc

    def release(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        try:
            returncode = self._process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            self._terminate()
            raise RuntimeError(f"GStreamer pipeline timed out while finalizing {self._path}") from exc
        stderr_output = self._read_stderr()
        if returncode != 0:
            raise RuntimeError(
                f"GStreamer pipeline failed for {self._path} (exit {returncode}): {stderr_output or 'no stderr output'}"
            )
        self._close_stderr()

    def _failure_message(self, message: str) -> str:
        return f"{message} for {self._path}: {self._read_stderr() or 'no stderr output'}"

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", "replace").strip()
        except Exception:
            return ""

    def _close_stderr(self) -> None:
        if self._process.stderr is not None and not self._process.stderr.closed:
            self._process.stderr.close()

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=5)
        self._close_stderr()


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        logger.warning("Failed to remove partial file %s", path, exc_info=True)
