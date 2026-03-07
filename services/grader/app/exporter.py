"""SVO2 → MP4 (left eye) + NPZ (depth frames) exporter.

Converts ZED proprietary SVO2 recordings into portable formats for
offline playback and analysis on machines without the ZED SDK.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("grader.exporter")


def export_svo2(svo2_path: str) -> dict:
    """Export a single SVO2 file to MP4 (left eye) + NPZ (depth).

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

    mp4_path = session_dir / f"{cam_name}_left.mp4"
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

    info = zed.get_camera_information()
    w = int(info.camera_configuration.resolution.width)
    h = int(info.camera_configuration.resolution.height)
    fps = int(info.camera_configuration.fps)
    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        # Fallback codec if avc1 is unavailable
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

    image = sl.Mat()
    depth = sl.Mat()

    depth_arrays: dict[str, np.ndarray] = {}
    frame_idx = 0
    chunk_idx = 0
    tmp_dir = session_dir / f".{cam_name}_depth_tmp"

    runtime = sl.RuntimeParameters()

    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        bgr = image.get_data()[:, :, :3].copy()
        writer.write(bgr)

        depth_arr = depth.get_data().copy().astype(np.float16)
        depth_arrays[f"frame_{frame_idx:06d}"] = depth_arr
        frame_idx += 1

        # Flush to disk every 100 frames to manage memory
        if len(depth_arrays) >= 100:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = tmp_dir / f"chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(str(chunk_path), **depth_arrays)
            depth_arrays.clear()
            chunk_idx += 1

    # Flush remaining
    if depth_arrays:
        if chunk_idx > 0:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = tmp_dir / f"chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(str(chunk_path), **depth_arrays)
            depth_arrays.clear()
            chunk_idx += 1
        else:
            # Small session, write directly
            np.savez_compressed(str(npz_path), **depth_arrays)
            depth_arrays.clear()

    writer.release()
    zed.close()

    # If we wrote chunks, merge into a single NPZ
    if chunk_idx > 0:
        _merge_chunks(tmp_dir, npz_path, chunk_idx)

    logger.info(
        "Exported %s: %d frames → %s + %s",
        svo2_path,
        frame_idx,
        mp4_path,
        npz_path,
    )

    # Extract sample frames (first, middle, last)
    sample_paths = _extract_sample_frames(str(mp4_path), session_dir, cam_name, frame_idx)

    return {
        "mp4_path": str(mp4_path),
        "npz_path": str(npz_path),
        "frame_count": frame_idx,
        "sample_paths": sample_paths,
    }


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

    cap.release()
    logger.info("Extracted %d sample frames for %s", len(sample_paths), cam_name)
    return sample_paths


def _merge_chunks(tmp_dir: Path, npz_path: Path, num_chunks: int) -> None:
    """Merge chunked NPZ files into a single compressed NPZ."""
    import shutil

    all_arrays: dict[str, np.ndarray] = {}
    for i in range(num_chunks):
        chunk_path = tmp_dir / f"chunk_{i:04d}.npz"
        with np.load(str(chunk_path)) as data:
            for key in data:
                all_arrays[key] = data[key]

    np.savez_compressed(str(npz_path), **all_arrays)

    # Clean up temp dir
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
