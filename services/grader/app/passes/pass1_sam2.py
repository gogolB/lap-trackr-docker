"""Pass 1 — SAM2 video segmentation for per-instrument binary masks."""

from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np

from app.passes.pass_data import PassData
from app.passes.tip_loader import load_tip_points, validate_tip_color

logger = logging.getLogger("grader.passes.pass1_sam2")

_LABEL_OBJ_IDS = {"green_tip": 1, "pink_tip": 2}


def _encode_rle(mask: np.ndarray) -> np.ndarray:
    """Encode a binary mask as a simple run-length encoded array for memory efficiency."""
    flat = mask.ravel().astype(np.uint8)
    # Store as packed bytes — much smaller than full mask
    return np.packbits(flat)


def _decode_rle(rle: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Decode RLE back to binary mask."""
    flat = np.unpackbits(rle)[: shape[0] * shape[1]]
    return flat.reshape(shape).astype(np.uint8)


def _frames_to_jpeg_dir(frames: list[np.ndarray]) -> str:
    """Write BGR frames as numbered JPEGs in a temp directory for SAM2."""
    import cv2
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    for idx, frame in enumerate(frames):
        path = f"{tmp_dir}/{idx:06d}.jpg"
        cv2.imwrite(path, frame)
    return tmp_dir


def _segment_view(
    predictor,
    frames: list[np.ndarray],
    tip_points: dict[str, tuple[float, float, int]],
    base_offset: int = 0,
    total_steps: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[np.ndarray | None]]:
    """Run SAM2 segmentation on a single camera view.

    Returns {label: [rle_mask_or_None per frame]}.
    """
    import shutil
    import torch

    if not frames:
        return {}

    h, w = frames[0].shape[:2]
    n_frames = len(frames)

    # SAM2 init_state expects a path to a JPEG directory or MP4 file
    frame_dir = _frames_to_jpeg_dir(frames)
    try:
        return _segment_view_inner(
            predictor, frame_dir, n_frames, tip_points,
            base_offset=base_offset, total_steps=total_steps,
            on_progress=on_progress,
        )
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def _segment_view_inner(
    predictor,
    frame_dir: str,
    n_frames: int,
    tip_points: dict[str, tuple[float, float, int]],
    base_offset: int = 0,
    total_steps: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[np.ndarray | None]]:
    """Inner segmentation loop after frames are written to disk.

    Progress is reported as (base_offset + step, total_steps) so the caller
    can combine on-axis and off-axis progress into a single consistent range.

    Uses streaming merge: forward masks are RLE-encoded immediately (~8x
    smaller), then merged one-at-a-time during backward propagation.  This
    avoids materializing two full-resolution mask dicts (~16 GB at 4k frames).
    """
    import torch

    # Build reverse mapping: obj_id → label
    obj_to_label = {}
    for label, obj_id in _LABEL_OBJ_IDS.items():
        if label in tip_points:
            obj_to_label[obj_id] = label

    masks_by_label: dict[str, list[np.ndarray | None]] = {
        label: [None] * n_frames for label in tip_points
    }

    with torch.inference_mode():
        if on_progress:
            on_progress(base_offset, total_steps)

        inference_state = predictor.init_state(
            video_path=frame_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        # Add points for each instrument
        for label, (x, y, frame_idx) in tip_points.items():
            obj_id = _LABEL_OBJ_IDS.get(label)
            if obj_id is None:
                continue
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # 1 = positive click
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        # Forward propagation — store as RLE immediately to save memory
        forward_rle: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state=inference_state,
        ):
            frame_rle = {}
            for i, oid in enumerate(obj_ids):
                mask = (masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                frame_rle[oid] = _encode_rle(mask)
            forward_rle[frame_idx] = frame_rle
            if on_progress:
                on_progress(base_offset + frame_idx + 1, total_steps)

        # Backward propagation — merge with forward immediately, encode to RLE
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state=inference_state,
            reverse=True,
        ):
            fwd_frame = forward_rle.pop(frame_idx, {})
            for i, oid in enumerate(obj_ids):
                label = obj_to_label.get(oid)
                if label is None:
                    continue
                bwd = (masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                fwd_rle = fwd_frame.get(oid)
                if fwd_rle is not None:
                    fwd = _decode_rle(fwd_rle, bwd.shape)
                    merged = np.maximum(fwd, bwd)
                else:
                    merged = bwd
                masks_by_label[label][frame_idx] = _encode_rle(merged)
            if on_progress:
                on_progress(base_offset + n_frames + frame_idx + 1, total_steps)

        # Forward-only frames (not visited by backward propagation)
        for fidx, frame_rle in forward_rle.items():
            for oid, rle in frame_rle.items():
                label = obj_to_label.get(oid)
                if label is not None and masks_by_label[label][fidx] is None:
                    masks_by_label[label][fidx] = rle

        predictor.reset_state(inference_state)

    return masks_by_label


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
    cameras: set[str] | None = None,
) -> bool:
    """Execute Pass 1: SAM2 video segmentation.

    Returns True if tip_init.json was missing and auto-detected fallback was used.
    """
    import torch

    process_on = cameras is None or "on_axis" in cameras
    process_off = cameras is None or "off_axis" in cameras
    logger.info("Pass 1: SAM2 segmentation starting (cameras=%s)",
                "all" if cameras is None else cameras)

    sample_interval = int(os.environ.get("FRAME_SAMPLE_INTERVAL", "5"))

    # Load tip points separately for each camera view with resolved frame indices
    on_tip_points, on_fallback = (
        load_tip_points(
            data.session_dir, "on_axis", sample_interval, n_frames=len(data.on_frames),
            frame_indices=data.on_frame_indices or None,
        ) if process_on and data.on_frames else ({}, False)
    )
    off_tip_points, off_fallback = (
        load_tip_points(
            data.session_dir, "off_axis", sample_interval, n_frames=len(data.off_frames),
            frame_indices=data.off_frame_indices or None,
        ) if process_off and data.off_frames else ({}, False)
    )
    used_fallback = on_fallback or off_fallback

    if not on_tip_points and not off_tip_points:
        logger.warning("No tip init points found for either camera, skipping SAM2 pass")
        return used_fallback

    # Validate tip colors on the resolved frames
    if data.on_frames and on_tip_points:
        for label, (x, y, idx) in on_tip_points.items():
            validate_tip_color(data.on_frames[idx], x, y, label)
    if data.off_frames and off_tip_points:
        for label, (x, y, idx) in off_tip_points.items():
            validate_tip_color(data.off_frames[idx], x, y, label)

    # Load SAM2 model
    from app.config import SAM2_MODEL_PATH, SAM2_CONFIG_PATH

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        logger.error("sam2 package not installed, skipping Pass 1")
        return used_fallback

    if on_progress:
        on_progress("pass1_sam2", 0, 1, "Loading SAM2 model")

    device = os.environ.get("_GRADER_DEVICE", "")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("SAM2 using device: %s", device)

    predictor = build_sam2_video_predictor(
        config_file=SAM2_CONFIG_PATH,
        ckpt_path=SAM2_MODEL_PATH,
        device=device,
    )

    # Total steps = (forward + backward) for each active camera
    on_steps = len(data.on_frames) * 2 if process_on else 0
    off_steps = len(data.off_frames) * 2 if process_off else 0
    total_steps = on_steps + off_steps

    try:
        # Segment on-axis view
        if on_tip_points:
            logger.info("Segmenting on-axis view (%d frames)", len(data.on_frames))
            data.on_masks = _segment_view(
                predictor,
                data.on_frames,
                on_tip_points,
                base_offset=0,
                total_steps=total_steps,
                on_progress=lambda c, t: on_progress("pass1_sam2", c, t, "SAM2 on-axis") if on_progress else None,
            )
            logger.info("On-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.on_masks.items()})
        else:
            logger.warning("No on-axis tip points, skipping on-axis segmentation")

        # Segment off-axis view
        if off_tip_points:
            logger.info("Segmenting off-axis view (%d frames)", len(data.off_frames))
            data.off_masks = _segment_view(
                predictor,
                data.off_frames,
                off_tip_points,
                base_offset=on_steps,
                total_steps=total_steps,
                on_progress=lambda c, t: on_progress("pass1_sam2", c, t, "SAM2 off-axis") if on_progress else None,
            )
            logger.info("Off-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.off_masks.items()})
        else:
            logger.warning("No off-axis tip points, skipping off-axis segmentation")

    finally:
        # Unload SAM2 to free GPU memory
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM2 model unloaded, GPU memory freed")

    if on_progress:
        on_progress("pass1_sam2", total_steps, total_steps, "SAM2 segmentation complete")
    logger.info("Pass 1 complete")
    return used_fallback
