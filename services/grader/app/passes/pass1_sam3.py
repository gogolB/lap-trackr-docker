"""Pass 1 (SAM3) — SAM3 video segmentation for per-instrument binary masks.

Uses Meta's Segment Anything Model 3 via its SAM2-compatible tracker API
(init_state / add_new_points / propagate_in_video).

Also supports text-based prompting via the handle_request API when no
tip_init.json is available.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Callable

import cv2
import numpy as np

from app.passes.pass_data import PassData
from app.passes.tip_loader import load_tip_points, validate_tip_color

logger = logging.getLogger("grader.passes.pass1_sam3")

_LABEL_OBJ_IDS = {"green_tip": 1, "pink_tip": 2}

# Text prompts used when no tip_init.json is available
_TEXT_PROMPTS = {
    "green_tip": "green surgical instrument tip",
    "pink_tip": "pink surgical instrument tip",
}


def _encode_rle(mask: np.ndarray) -> np.ndarray:
    """Encode a binary mask as packed bits for memory efficiency."""
    flat = mask.ravel().astype(np.uint8)
    return np.packbits(flat)


def _frames_to_jpeg_dir(frames: list[np.ndarray]) -> str:
    """Write BGR frames as numbered JPEGs in a temp directory for SAM3."""
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
    for idx, frame in enumerate(frames):
        path = f"{tmp_dir}/{idx:06d}.jpg"
        cv2.imwrite(path, frame)
    return tmp_dir


def _segment_view_points(
    predictor,
    inference_state,
    frames: list[np.ndarray],
    tip_points: dict[str, tuple[float, float, int]],
    base_offset: int = 0,
    total_steps: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[np.ndarray | None]]:
    """Run SAM3 point-based segmentation using the SAM2-compatible API.

    Returns {label: [rle_mask_or_None per frame]}.
    """
    import torch

    if not frames:
        return {}

    h, w = frames[0].shape[:2]
    n_frames = len(frames)

    if on_progress:
        on_progress(base_offset, total_steps)

    # Add point prompts for each instrument (relative coords)
    points_added = 0
    for label, (x, y, frame_idx) in tip_points.items():
        obj_id = _LABEL_OBJ_IDS.get(label)
        if obj_id is None:
            continue

        rel_points = torch.tensor([[x / w, y / h]], dtype=torch.float32)
        point_labels = torch.tensor([1], dtype=torch.int32)  # 1 = positive click

        _, out_obj_ids, out_low_res, out_video_res = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=rel_points,
            labels=point_labels,
        )
        points_added += 1
        logger.info(
            "Added point prompt for %s (obj_id=%d): (%.1f, %.1f) -> rel (%.4f, %.4f) at frame %d "
            "(returned %d objects)",
            label, obj_id, x, y, x / w, y / h, frame_idx, len(out_obj_ids),
        )

    if points_added == 0:
        logger.warning("No point prompts were added — tip_points may be empty or have unknown labels")
        return {label: [None] * n_frames for label in tip_points}

    from app.passes.pass1_sam2 import _decode_rle

    # Build reverse mapping: obj_id → label
    obj_to_label = {}
    for label, obj_id in _LABEL_OBJ_IDS.items():
        if label in tip_points:
            obj_to_label[obj_id] = label

    masks_by_label: dict[str, list[np.ndarray | None]] = {
        label: [None] * n_frames for label in tip_points
    }

    # Forward propagation — store as RLE immediately to save memory
    forward_rle: dict[int, dict[int, np.ndarray]] = {}
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, start_frame_idx=0, max_frame_num_to_track=n_frames,
        reverse=False, propagate_preflight=True,
    ):
        frame_rle = {}
        for i, oid in enumerate(obj_ids):
            mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
            frame_rle[oid] = _encode_rle(mask)
        forward_rle[frame_idx] = frame_rle
        if on_progress:
            on_progress(base_offset + frame_idx + 1, total_steps)

    # Backward propagation — merge with forward immediately, encode to RLE
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, start_frame_idx=0, max_frame_num_to_track=n_frames,
        reverse=True,
    ):
        fwd_frame = forward_rle.pop(frame_idx, {})
        for i, oid in enumerate(obj_ids):
            label = obj_to_label.get(oid)
            if label is None:
                continue
            bwd = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
            fwd_rle_data = fwd_frame.get(oid)
            if fwd_rle_data is not None:
                fwd = _decode_rle(fwd_rle_data, bwd.shape)
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

    return masks_by_label


def _segment_view_text(
    predictor,
    frames: list[np.ndarray],
    base_offset: int = 0,
    total_steps: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[np.ndarray | None]]:
    """Run SAM3 text-prompt segmentation using the handle_request API.

    Used when no tip_init.json is available.
    Returns {label: [rle_mask_or_None per frame]}.
    """
    if not frames:
        return {}

    n_frames = len(frames)
    frame_dir = _frames_to_jpeg_dir(frames)

    try:
        if on_progress:
            on_progress(base_offset, total_steps)

        # Start session
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": frame_dir,
        })
        session_id = response["session_id"]

        # Add text prompts
        for label, text in _TEXT_PROMPTS.items():
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text,
            })
            logger.info("Added text prompt for %s: '%s'", label, text)

        # Propagate both directions
        masks_by_label: dict[str, list[np.ndarray | None]] = {
            label: [None] * n_frames for label in _TEXT_PROMPTS
        }

        step = 0
        for result in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
            "start_frame_index": 0,
        }):
            frame_idx = result["frame_index"]
            frame_outputs = result.get("outputs", {})

            for obj_id, obj_data in frame_outputs.items():
                mask = obj_data.get("mask")
                if mask is None:
                    continue
                if not isinstance(mask, np.ndarray):
                    mask = np.array(mask, dtype=np.uint8)

                # Map object IDs back to labels by order of prompting
                label_list = list(_TEXT_PROMPTS.keys())
                label_idx = obj_id if isinstance(obj_id, int) else 0
                if label_idx < len(label_list):
                    label = label_list[label_idx]
                    masks_by_label[label][frame_idx] = _encode_rle(mask)

            step += 1
            if on_progress:
                on_progress(base_offset + step, total_steps)

        predictor.handle_request({
            "type": "close_session",
            "session_id": session_id,
        })

        return masks_by_label
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
    cameras: set[str] | None = None,
) -> bool:
    """Execute Pass 1: SAM3 video segmentation.

    Uses the SAM2-compatible tracker API for point prompts, or the
    handle_request API for text prompts when no tip_init.json exists.

    Returns True if tip_init.json was missing and fallback was used.
    """
    import torch

    process_on = cameras is None or "on_axis" in cameras
    process_off = cameras is None or "off_axis" in cameras
    logger.info("Pass 1: SAM3 segmentation starting (cameras=%s)",
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

    use_text_prompts = not on_tip_points and not off_tip_points

    if use_text_prompts:
        logger.info("No tip init points found, using SAM3 text prompts")
        used_fallback = True

    # Validate tip colors on the resolved frames
    if not use_text_prompts:
        if data.on_frames and on_tip_points:
            for label, (x, y, idx) in on_tip_points.items():
                validate_tip_color(data.on_frames[idx], x, y, label)
        if data.off_frames and off_tip_points:
            for label, (x, y, idx) in off_tip_points.items():
                validate_tip_color(data.off_frames[idx], x, y, label)

    try:
        from sam3.model_builder import build_sam3_video_model, build_sam3_video_predictor
    except ImportError:
        logger.error("sam3 package not installed, skipping Pass 1")
        logger.error("Install with: pip install git+https://github.com/facebookresearch/sam3.git")
        return used_fallback

    if on_progress:
        on_progress("pass1_sam3", 0, 1, "Loading SAM3 model")

    device = os.environ.get("_GRADER_DEVICE", "")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("SAM3 using device: %s", device)

    # Total steps = (forward + backward) for each active camera
    on_steps = len(data.on_frames) * 2 if process_on else 0
    off_steps = len(data.off_frames) * 2 if process_off else 0
    total_steps = on_steps + off_steps

    try:
        if use_text_prompts:
            # Text prompt path — uses handle_request API via Sam3VideoPredictorMultiGPU
            gpus_to_use = [torch.cuda.current_device()] if device == "cuda" else []
            predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

            logger.info("Segmenting on-axis view with text prompts (%d frames)", len(data.on_frames))
            data.on_masks = _segment_view_text(
                predictor,
                data.on_frames,
                base_offset=0,
                total_steps=total_steps,
                on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 text on-axis") if on_progress else None,
            )

            logger.info("Segmenting off-axis view with text prompts (%d frames)", len(data.off_frames))
            data.off_masks = _segment_view_text(
                predictor,
                data.off_frames,
                base_offset=on_steps,
                total_steps=total_steps,
                on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 text off-axis") if on_progress else None,
            )

            del predictor
        else:
            # Point prompt path — uses SAM2-compatible tracker API
            logger.info("Building SAM3 tracker (SAM2-compatible mode)")
            sam3_model = build_sam3_video_model()
            predictor = sam3_model.tracker
            predictor.backbone = sam3_model.detector.backbone

            # On-axis
            if on_tip_points:
                logger.info("Segmenting on-axis view (%d frames)", len(data.on_frames))
                on_frame_dir = _frames_to_jpeg_dir(data.on_frames)
                try:
                    on_state = predictor.init_state(
                        video_path=on_frame_dir,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                    data.on_masks = _segment_view_points(
                        predictor, on_state,
                        data.on_frames, on_tip_points,
                        base_offset=0, total_steps=total_steps,
                        on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 on-axis") if on_progress else None,
                    )
                    if hasattr(predictor, "reset_state"):
                        predictor.reset_state(on_state)
                finally:
                    shutil.rmtree(on_frame_dir, ignore_errors=True)
                logger.info("On-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.on_masks.items()})
            else:
                logger.warning("No on-axis tip points, skipping on-axis segmentation")

            # Off-axis
            if off_tip_points:
                logger.info("Segmenting off-axis view (%d frames)", len(data.off_frames))
                off_frame_dir = _frames_to_jpeg_dir(data.off_frames)
                try:
                    off_state = predictor.init_state(
                        video_path=off_frame_dir,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                    data.off_masks = _segment_view_points(
                        predictor, off_state,
                        data.off_frames, off_tip_points,
                        base_offset=on_steps, total_steps=total_steps,
                        on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 off-axis") if on_progress else None,
                    )
                    if hasattr(predictor, "reset_state"):
                        predictor.reset_state(off_state)
                finally:
                    shutil.rmtree(off_frame_dir, ignore_errors=True)
                logger.info("Off-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.off_masks.items()})
            else:
                logger.warning("No off-axis tip points, skipping off-axis segmentation")

            del predictor
            del sam3_model

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM3 model unloaded, GPU memory freed")

    if on_progress:
        on_progress("pass1_sam3", total_steps, total_steps, "SAM3 segmentation complete")
    logger.info("Pass 1 complete")
    return used_fallback
