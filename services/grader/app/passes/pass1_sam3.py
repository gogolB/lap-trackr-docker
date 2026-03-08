"""Pass 1 (SAM3) — SAM3 video segmentation for per-instrument binary masks.

Uses Meta's Segment Anything Model 3 with the request-response video API.
Supports both point prompts (from tip_init.json) and text prompts.
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


def _segment_view(
    predictor,
    frames: list[np.ndarray],
    tip_points: dict[str, tuple[float, float, int]],
    use_text_prompts: bool = False,
    base_offset: int = 0,
    total_steps: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[np.ndarray | None]]:
    """Run SAM3 segmentation on a single camera view.

    Returns {label: [rle_mask_or_None per frame]}.
    """
    import torch

    if not frames:
        return {}

    h, w = frames[0].shape[:2]
    n_frames = len(frames)

    frame_dir = _frames_to_jpeg_dir(frames)
    try:
        return _segment_view_inner(
            predictor, frame_dir, n_frames, h, w,
            tip_points, use_text_prompts=use_text_prompts,
            base_offset=base_offset, total_steps=total_steps,
            on_progress=on_progress,
        )
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def _segment_view_inner(
    predictor,
    frame_dir: str,
    n_frames: int,
    h: int,
    w: int,
    tip_points: dict[str, tuple[float, float, int]],
    use_text_prompts: bool = False,
    base_offset: int = 0,
    total_steps: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[np.ndarray | None]]:
    """Inner segmentation loop using SAM3 request-response API."""
    import torch

    if on_progress:
        on_progress(base_offset, total_steps)

    # Start a video session
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": frame_dir,
    })
    session_id = response["session_id"]

    try:
        # Add prompts for each instrument
        for label, (x, y, frame_idx) in tip_points.items():
            if use_text_prompts:
                text = _TEXT_PROMPTS.get(label, label)
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_idx,
                    "text": text,
                })
                logger.info("Added text prompt for %s: '%s' at frame %d", label, text, frame_idx)
            else:
                # Normalize coordinates to [0, 1] for SAM3
                norm_x = x / w
                norm_y = y / h
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_idx,
                    "points": [[norm_x, norm_y]],
                    "labels": [1],
                })
                logger.info("Added point prompt for %s: (%.1f, %.1f) -> norm (%.4f, %.4f) at frame %d",
                            label, x, y, norm_x, norm_y, frame_idx)

        # Forward propagation
        forward_results: dict[int, dict[int, np.ndarray]] = {}
        for frame_result in predictor.propagate_in_video(session_id):
            frame_idx = frame_result["frame_idx"]
            obj_ids = frame_result.get("obj_ids", [])
            masks_tensor = frame_result.get("video_res_masks")

            if masks_tensor is not None:
                frame_masks = {}
                for i, oid in enumerate(obj_ids):
                    mask = (masks_tensor[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                    frame_masks[oid] = mask
                forward_results[frame_idx] = frame_masks

            if on_progress:
                on_progress(base_offset + frame_idx + 1, total_steps)

        # Backward propagation
        backward_results: dict[int, dict[int, np.ndarray]] = {}
        for frame_result in predictor.propagate_in_video(
            session_id, direction="backward"
        ):
            frame_idx = frame_result["frame_idx"]
            obj_ids = frame_result.get("obj_ids", [])
            masks_tensor = frame_result.get("video_res_masks")

            if masks_tensor is not None:
                frame_masks = {}
                for i, oid in enumerate(obj_ids):
                    mask = (masks_tensor[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                    frame_masks[oid] = mask
                backward_results[frame_idx] = frame_masks

            if on_progress:
                on_progress(base_offset + n_frames + frame_idx + 1, total_steps)

    finally:
        # Close session to free resources
        try:
            predictor.close_session(session_id)
        except Exception:
            pass

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


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> bool:
    """Execute Pass 1: SAM3 video segmentation.

    Returns True if tip_init.json was missing and auto-detected fallback was used.
    """
    import torch

    logger.info("Pass 1: SAM3 segmentation starting")

    tip_points, used_fallback = _load_tip_points(data.session_dir)
    use_text_prompts = not tip_points

    if use_text_prompts:
        logger.info("No tip init points found, using SAM3 text prompts")
        tip_points = {label: (0, 0, 0) for label in _LABEL_OBJ_IDS}
        used_fallback = True

    logger.info("Tip init points: %s", {k: (v[0], v[1]) for k, v in tip_points.items()})

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError:
        logger.error("sam3 package not installed, skipping Pass 1")
        logger.error("Install with: pip install sam3")
        return used_fallback

    if on_progress:
        on_progress("pass1_sam3", 0, 1, "Loading SAM3 model")

    device = os.environ.get("_GRADER_DEVICE", "")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("SAM3 using device: %s", device)

    # SAM3 auto-downloads checkpoint from HuggingFace if no path given
    checkpoint_path = os.environ.get("SAM3_MODEL_PATH", None)

    predictor = build_sam3_video_predictor(
        checkpoint=checkpoint_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    # Total steps = (forward + backward) for on-axis + (forward + backward) for off-axis
    on_steps = len(data.on_frames) * 2
    off_steps = len(data.off_frames) * 2
    total_steps = on_steps + off_steps

    try:
        # Segment on-axis view
        logger.info("Segmenting on-axis view (%d frames)", len(data.on_frames))
        data.on_masks = _segment_view(
            predictor,
            data.on_frames,
            tip_points,
            use_text_prompts=use_text_prompts,
            base_offset=0,
            total_steps=total_steps,
            on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 on-axis") if on_progress else None,
        )
        logger.info("On-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.on_masks.items()})

        # Segment off-axis view
        logger.info("Segmenting off-axis view (%d frames)", len(data.off_frames))
        data.off_masks = _segment_view(
            predictor,
            data.off_frames,
            tip_points,
            use_text_prompts=use_text_prompts,
            base_offset=on_steps,
            total_steps=total_steps,
            on_progress=lambda c, t: on_progress("pass1_sam3", c, t, "SAM3 off-axis") if on_progress else None,
        )
        logger.info("Off-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.off_masks.items()})

    finally:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM3 model unloaded, GPU memory freed")

    if on_progress:
        on_progress("pass1_sam3", total_steps, total_steps, "SAM3 segmentation complete")
    logger.info("Pass 1 complete")
    return used_fallback
