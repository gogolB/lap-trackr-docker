"""SVO2 file reader using the ZED SDK.

Extracts left-camera RGB frames and depth maps from an SVO2 recording,
sampling every Nth frame to keep processing tractable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from app.camera_transform import apply_transforms, get_camera_transform
from app.config import DEFAULT_FPS, FRAME_SAMPLE_INTERVAL

logger = logging.getLogger("grader.svo_loader")


def load_svo2(
    svo_path: str,
    sample_interval: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    camera_config: dict | None = None,
    extra_frames: set[int] | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, List[int]]:
    """Open an SVO2 file and return sampled frames, depth maps, FPS, and indices.

    Parameters
    ----------
    svo_path : str
        Absolute path to the ``.svo2`` file.
    sample_interval : int, optional
        Keep every *n*-th frame.  Defaults to ``FRAME_SAMPLE_INTERVAL`` from
        config.
    extra_frames : set[int], optional
        Additional original frame indices to include even if they don't fall
        on the regular sample interval (e.g. tip-init labeled frames).

    Returns
    -------
    frames : list[np.ndarray]
        BGR images as ``(H, W, 3)`` uint8 arrays.
    depth_maps : list[np.ndarray]
        Depth images as ``(H, W)`` float32 arrays (metres).
    fps : float
        Frames-per-second of the original recording.
    original_indices : list[int]
        The original video frame index for each returned frame, in order.
    """

    if sample_interval is None:
        sample_interval = FRAME_SAMPLE_INTERVAL
    cam_name = Path(svo_path).stem
    transform = get_camera_transform(camera_config, cam_name)

    _extra = extra_frames or set()

    # Prefer exported files when available so grading uses the same transformed
    # frames/depth that the user sees in tip init and playback artifacts.
    export_result = _try_load_from_exports(
        svo_path,
        sample_interval,
        on_progress=on_progress,
        camera_config=camera_config,
        extra_frames=_extra,
    )
    if export_result is not None:
        return export_result

    try:
        import pyzed.sl as sl
    except ImportError:
        sl = None
        logger.info("ZED SDK (pyzed) not available, trying exported files")

    if sl is None:
        logger.warning(
            "No ZED SDK and no exported files -- returning synthetic data "
            "for development/testing."
        )
        synth_frames, synth_depths, synth_fps = _generate_synthetic_data(sample_interval)
        synth_indices = list(range(0, len(synth_frames) * sample_interval, sample_interval))
        return synth_frames, synth_depths, synth_fps, synth_indices

    # --- ZED SDK path -------------------------------------------------------
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.1

    camera = sl.Camera()
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to open SVO2 file '{svo_path}': {status}"
        )

    fps: float = camera.get_camera_information().camera_configuration.fps
    if fps <= 0:
        fps = DEFAULT_FPS
    total_frames = camera.get_svo_number_of_frames()
    if total_frames <= 0:
        total_frames = 0

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    frames: list[np.ndarray] = []
    depth_maps: list[np.ndarray] = []
    original_indices: list[int] = []
    frame_idx = 0
    view = sl.VIEW.RIGHT if transform["swap_eyes"] else sl.VIEW.LEFT
    use_right_depth = bool(transform["swap_eyes"]) and hasattr(sl.MEASURE, "DEPTH_RIGHT")
    depth_measure = sl.MEASURE.DEPTH_RIGHT if use_right_depth else sl.MEASURE.DEPTH
    if transform["swap_eyes"] and not use_right_depth:
        logger.warning(
            "swap_eyes enabled for %s but DEPTH_RIGHT is unavailable; using left-eye depth",
            cam_name,
        )

    while True:
        err = camera.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            break

        if frame_idx % sample_interval == 0 or frame_idx in _extra:
            camera.retrieve_image(image_mat, view)
            camera.retrieve_measure(depth_mat, depth_measure)

            # .get_data() returns a numpy view; copy to own the memory.
            frame = np.array(image_mat.get_data()[:, :, :3], dtype=np.uint8)
            depth = np.array(depth_mat.get_data(), dtype=np.float32)
            frames.append(apply_transforms(frame, transform))
            depth_maps.append(apply_transforms(depth, transform))
            original_indices.append(frame_idx)

        current = frame_idx + 1
        if on_progress and (current % 10 == 0 or current == total_frames):
            on_progress(current, max(total_frames, current, 1))
        frame_idx += 1

    if on_progress:
        on_progress(frame_idx, max(total_frames, frame_idx, 1))

    camera.close()
    logger.info(
        "Read %d / %d frames from %s (sample_interval=%d, extra=%d)",
        len(frames),
        frame_idx,
        svo_path,
        sample_interval,
        len(_extra),
    )
    return frames, depth_maps, fps, original_indices


# ---------------------------------------------------------------------------
# Fallback: load from exported MP4 + NPZ files
# ---------------------------------------------------------------------------

def _try_load_from_exports(
    svo_path: str,
    sample_interval: int,
    on_progress: Callable[[int, int], None] | None = None,
    camera_config: dict | None = None,
    extra_frames: set[int] | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, List[int]] | None:
    """Try to load frames and depth from exported MP4 + NPZ files.

    Returns None if the export files don't exist.
    """
    from pathlib import Path
    import cv2

    session_dir = Path(svo_path).parent
    cam_name = Path(svo_path).stem  # "on_axis" or "off_axis"
    transform = get_camera_transform(camera_config, cam_name)
    export_meta_path = session_dir / f"{cam_name}_export.json"
    export_meta: dict | None = None
    if export_meta_path.exists():
        try:
            export_meta = json.loads(export_meta_path.read_text())
        except Exception:
            logger.warning("Failed to parse export metadata %s", export_meta_path, exc_info=True)

    exports_transformed = bool(export_meta and export_meta.get("transforms_applied"))
    if exports_transformed:
        mp4_path = session_dir / f"{cam_name}_left.mp4"
    else:
        logical_eye = "right" if transform["swap_eyes"] else "left"
        mp4_path = session_dir / f"{cam_name}_{logical_eye}.mp4"
        if transform["swap_eyes"]:
            logger.warning(
                "Loading pre-transform exports for %s with swap_eyes enabled; "
                "video can use the right-eye MP4 but depth remains left-eye until re-exported",
                cam_name,
            )
    npz_path = session_dir / f"{cam_name}_depth.npz"

    if not mp4_path.exists() or not npz_path.exists():
        return None

    logger.info("Loading from exports: %s + %s", mp4_path, npz_path)

    # Load video frames
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        logger.warning("Failed to open MP4 %s", mp4_path)
        return None

    depth_data = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0

        # Load depth arrays
        depth_data = np.load(str(npz_path))
        depth_keys = sorted(depth_data.files)

        _extra = extra_frames or set()

        frames: list[np.ndarray] = []
        depth_maps: list[np.ndarray] = []
        original_indices: list[int] = []
        frame_idx = 0

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0 or frame_idx in _extra:
                frame = bgr.copy()
                key = f"frame_{frame_idx:06d}"
                if key in depth_data:
                    depth = depth_data[key].astype(np.float32)
                elif frame_idx < len(depth_keys):
                    depth = depth_data[depth_keys[frame_idx]].astype(np.float32)
                else:
                    # No depth for this frame, use zeros
                    h, w = bgr.shape[:2]
                    depth = np.zeros((h, w), dtype=np.float32)

                if not exports_transformed:
                    frame = apply_transforms(frame, transform)
                    depth = apply_transforms(depth, transform)

                frames.append(frame)
                depth_maps.append(depth)
                original_indices.append(frame_idx)

            current = frame_idx + 1
            if on_progress and (current % 10 == 0 or current == total_frames):
                on_progress(current, max(total_frames, current, 1))
            frame_idx += 1

        if on_progress:
            on_progress(frame_idx, max(total_frames, frame_idx, 1))

        n_extra_loaded = sum(1 for i in original_indices if i in _extra and i % sample_interval != 0)
        logger.info(
            "Loaded %d / %d frames from exports (sample_interval=%d, +%d labeled frames)",
            len(frames),
            frame_idx,
            sample_interval,
            n_extra_loaded,
        )
        return frames, depth_maps, fps, original_indices
    finally:
        cap.release()
        if depth_data is not None:
            depth_data.close()


def load_frames_list(
    svo_path: str,
    sample_interval: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    camera_config: dict | None = None,
    extra_frames: set[int] | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, List[int]]:
    """Convenience wrapper that returns frames and depth for in-memory processing.

    Identical to :func:`load_svo2` but named explicitly for the v2 pipeline
    where callers need raw frame lists rather than file paths.
    """
    return load_svo2(
        svo_path,
        sample_interval=sample_interval,
        on_progress=on_progress,
        camera_config=camera_config,
        extra_frames=extra_frames,
    )


# ---------------------------------------------------------------------------
# Fallback for environments without the ZED SDK
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    sample_interval: int,
    total_frames: int = 900,
    width: int = 1280,
    height: int = 720,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Return synthetic frames and depth maps for development/testing.

    Produces a deterministic sequence so downstream stages behave
    consistently during testing.
    """

    rng = np.random.RandomState(42)
    n_sampled = total_frames // sample_interval

    frames: list[np.ndarray] = []
    depth_maps: list[np.ndarray] = []

    for _ in range(n_sampled):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # Depth in metres: most of the scene between 0.2 and 1.0 m
        depth = rng.uniform(0.2, 1.0, (height, width)).astype(np.float32)
        frames.append(frame)
        depth_maps.append(depth)

    logger.info(
        "Generated %d synthetic frames (%dx%d) with depth",
        n_sampled,
        width,
        height,
    )
    return frames, depth_maps, DEFAULT_FPS
