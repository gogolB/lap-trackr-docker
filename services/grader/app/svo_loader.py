"""SVO2 file reader using the ZED SDK.

Extracts left-camera RGB frames and depth maps from an SVO2 recording,
sampling every Nth frame to keep processing tractable.
"""

from __future__ import annotations

import logging
from typing import Tuple, List

import numpy as np

from app.config import DEFAULT_FPS, FRAME_SAMPLE_INTERVAL

logger = logging.getLogger("grader.svo_loader")


def load_svo2(
    svo_path: str,
    sample_interval: int | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Open an SVO2 file and return sampled frames, depth maps, and FPS.

    Parameters
    ----------
    svo_path : str
        Absolute path to the ``.svo2`` file.
    sample_interval : int, optional
        Keep every *n*-th frame.  Defaults to ``FRAME_SAMPLE_INTERVAL`` from
        config.

    Returns
    -------
    frames : list[np.ndarray]
        BGR images as ``(H, W, 3)`` uint8 arrays.
    depth_maps : list[np.ndarray]
        Depth images as ``(H, W)`` float32 arrays (metres).
    fps : float
        Frames-per-second of the original recording.
    """

    if sample_interval is None:
        sample_interval = FRAME_SAMPLE_INTERVAL

    try:
        import pyzed.sl as sl
    except ImportError:
        logger.warning(
            "ZED SDK (pyzed) not available -- returning synthetic data "
            "for development/testing."
        )
        return _generate_synthetic_data(sample_interval)

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

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    frames: list[np.ndarray] = []
    depth_maps: list[np.ndarray] = []
    frame_idx = 0

    while True:
        err = camera.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            break

        if frame_idx % sample_interval == 0:
            camera.retrieve_image(image_mat, sl.VIEW.LEFT)
            camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

            # .get_data() returns a numpy view; copy to own the memory.
            frames.append(np.array(image_mat.get_data()[:, :, :3], dtype=np.uint8))
            depth_maps.append(np.array(depth_mat.get_data(), dtype=np.float32))

        frame_idx += 1

    camera.close()
    logger.info(
        "Read %d / %d frames from %s (sample_interval=%d)",
        len(frames),
        frame_idx,
        svo_path,
        sample_interval,
    )
    return frames, depth_maps, fps


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
