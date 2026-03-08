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

logger = logging.getLogger("grader.passes.pass1_sam3")

_LABEL_OBJ_IDS = {"green_tip": 1, "pink_tip": 2}

# Text prompts used when no tip_init.json is available
_TEXT_PROMPTS = {
    "green_tip": "green surgical instrument tip",
    "pink_tip": "pink surgical instrument tip",
}


def _load_tip_points(session_dir) -> tuple[dict[str, tuple[float, float, int]], bool]:
    """Load tip init points, preferring tip_init.json with fallback to tip_detections.json.

    Returns ({label: (x, y, frame_idx)}, used_fallback) for each instrument.
    """
    import json
    from pathlib import Path

    tip_init_path = session_dir / "tip_init.json"
    tip_detections_path = session_dir / "tip_detections.json"
    used_fallback = False

    if tip_init_path.exists():
        tip_data = json.loads(tip_init_path.read_text())
    elif tip_detections_path.exists():
        logger.warning("tip_init.json not found, falling back to tip_detections.json")
        tip_data = json.loads(tip_detections_path.read_text())
        used_fallback = True
    else:
        return {}, True

    best: dict[str, tuple[float, float, int]] = {}

    for filename, detections in tip_data.items():
        for det in detections:
            label = det.get("label")
            if label not in _LABEL_OBJ_IDS:
                color = det.get("color")
                if color == "green":
                    label = "green_tip"
                elif color == "pink":
                    label = "pink_tip"
                else:
                    continue
            x, y = float(det["x"]), float(det["y"])
            confidence = float(det.get("confidence", 0.5))
            prev = best.get(label)
            if prev is None or confidence > prev[2]:
                best[label] = (x, y, 0)

    return best, used_fallback


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
    for label, (x, y, frame_idx) in tip_points.items():
        obj_id = _LABEL_OBJ_IDS.get(label)
        if obj_id is None:
            continue

        rel_points = torch.tensor([[x / w, y / h]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int32)  # 1 = positive click

        predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=rel_points,
            labels=labels,
            clear_old_points=False,
        )
        logger.info(
            "Added point prompt for %s: (%.1f, %.1f) -> rel (%.4f, %.4f) at frame %d",
            label, x, y, x / w, y / h, frame_idx,
        )

    # Forward propagation
    forward_results: dict[int, dict[int, np.ndarray]] = {}
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, start_frame_idx=0, reverse=False,
    ):
        frame_masks = {}
        for i, oid in enumerate(obj_ids):
            mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
            frame_masks[oid] = mask
        forward_results[frame_idx] = frame_masks
        if on_progress:
            on_progress(base_offset + frame_idx + 1, total_steps)

    # Backward propagation
    backward_results: dict[int, dict[int, np.ndarray]] = {}
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, start_frame_idx=0, reverse=True,
    ):
        frame_masks = {}
        for i, oid in enumerate(obj_ids):
            mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
            frame_masks[oid] = mask
        backward_results[frame_idx] = frame_masks
        if on_progress:
            on_progress(base_offset + n_frames + frame_idx + 1, total_steps)

    # Merge forward and backward results (union)
    masks_by_label: dict[str, list[np.ndarray | None]] = {
        label: [None] * n_frames for label in tip_points
    }

    for label, obj_id in _LABEL_OBJ_IDS.items():
        if label not in tip_points:
            continue
        for fidx in range(n_frames):
            fwd = forward_results.get(fidx, {}).get(obj_id)
            bwd = backward_results.get(fidx, {}).get(obj_id)
            if fwd is not None and bwd is not None:
                merged = np.maximum(fwd, bwd)
            elif fwd is not None:
                merged = fwd
            elif bwd is not None:
                merged = bwd
            else:
                continue
            masks_by_label[label][fidx] = _encode_rle(merged)

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
) -> bool:
    """Execute Pass 1: SAM3 video segmentation.

    Uses the SAM2-compatible tracker API for point prompts, or the
    handle_request API for text prompts when no tip_init.json exists.

    Returns True if tip_init.json was missing and fallback was used.
    """
    import torch

    logger.info("Pass 1: SAM3 segmentation starting")

    tip_points, used_fallback = _load_tip_points(data.session_dir)
    use_text_prompts = not tip_points

    if use_text_prompts:
        logger.info("No tip init points found, using SAM3 text prompts")
        used_fallback = True

    if not use_text_prompts:
        logger.info("Tip init points: %s", {k: (v[0], v[1]) for k, v in tip_points.items()})

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

    # Total steps = (forward + backward) for on-axis + (forward + backward) for off-axis
    on_steps = len(data.on_frames) * 2
    off_steps = len(data.off_frames) * 2
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
            # build_sam3_video_model returns a model with .tracker (Sam3TrackerPredictor)
            # which has init_state / add_new_points / propagate_in_video
            logger.info("Building SAM3 tracker (SAM2-compatible mode)")
            sam3_model = build_sam3_video_model()
            predictor = sam3_model.tracker
            predictor.backbone = sam3_model.detector.backbone

            # On-axis
            logger.info("Segmenting on-axis view (%d frames)", len(data.on_frames))
            on_frame_dir = _frames_to_jpeg_dir(data.on_frames)
            try:
                on_state = predictor.init_state(video_path=on_frame_dir)
                data.on_masks = _segment_view_points(
                    predictor, on_state,
                    data.on_frames, tip_points,
                    base_offset=0, total_steps=total_steps,
                    on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 on-axis") if on_progress else None,
                )
                predictor.reset_state(on_state)
            finally:
                shutil.rmtree(on_frame_dir, ignore_errors=True)
            logger.info("On-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.on_masks.items()})

            # Off-axis
            logger.info("Segmenting off-axis view (%d frames)", len(data.off_frames))
            off_frame_dir = _frames_to_jpeg_dir(data.off_frames)
            try:
                off_state = predictor.init_state(video_path=off_frame_dir)
                data.off_masks = _segment_view_points(
                    predictor, off_state,
                    data.off_frames, tip_points,
                    base_offset=on_steps, total_steps=total_steps,
                    on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 off-axis") if on_progress else None,
                )
                predictor.reset_state(off_state)
            finally:
                shutil.rmtree(off_frame_dir, ignore_errors=True)
            logger.info("Off-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.off_masks.items()})

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
