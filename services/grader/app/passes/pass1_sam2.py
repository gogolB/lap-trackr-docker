"""Pass 1 — SAM2 video segmentation for per-instrument binary masks."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from app.passes.pass_data import PassData

logger = logging.getLogger("grader.passes.pass1_sam2")

_LABEL_OBJ_IDS = {"green_tip": 1, "pink_tip": 2}


def _load_tip_points(session_dir) -> dict[str, tuple[float, float, int]]:
    """Load tip init points from tip_init.json.

    Returns {label: (x, y, frame_idx)} for each instrument.
    """
    import json
    from pathlib import Path

    tip_init_path = session_dir / "tip_init.json"
    if not tip_init_path.exists():
        raise FileNotFoundError(f"tip_init.json not found at {tip_init_path}")

    tip_data = json.loads(tip_init_path.read_text())
    best: dict[str, tuple[float, float, int]] = {}

    for filename, detections in tip_data.items():
        for det in detections:
            label = det.get("label")
            if label not in _LABEL_OBJ_IDS:
                # Try color-based mapping
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
                best[label] = (x, y, 0)  # frame_idx=0 as init frame

    return best


def _encode_rle(mask: np.ndarray) -> np.ndarray:
    """Encode a binary mask as a simple run-length encoded array for memory efficiency."""
    flat = mask.ravel().astype(np.uint8)
    # Store as packed bytes — much smaller than full mask
    return np.packbits(flat)


def _decode_rle(rle: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Decode RLE back to binary mask."""
    flat = np.unpackbits(rle)[: shape[0] * shape[1]]
    return flat.reshape(shape).astype(np.uint8)


def _segment_view(
    predictor,
    frames: list[np.ndarray],
    tip_points: dict[str, tuple[float, float, int]],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[np.ndarray | None]]:
    """Run SAM2 segmentation on a single camera view.

    Returns {label: [rle_mask_or_None per frame]}.
    """
    import torch

    if not frames:
        return {}

    h, w = frames[0].shape[:2]
    n_frames = len(frames)

    # Initialize video state with inference_state context manager
    with torch.inference_mode():
        inference_state = predictor.init_state(video=frames)

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

        # Forward propagation
        forward_results: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state=inference_state,
        ):
            frame_masks = {}
            for i, oid in enumerate(obj_ids):
                mask = (masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                frame_masks[oid] = mask
            forward_results[frame_idx] = frame_masks
            if on_progress and (frame_idx + 1) % 10 == 0:
                on_progress(frame_idx + 1, n_frames * 2)

        # Backward propagation
        backward_results: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            inference_state=inference_state,
            reverse=True,
        ):
            frame_masks = {}
            for i, oid in enumerate(obj_ids):
                mask = (masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                frame_masks[oid] = mask
            backward_results[frame_idx] = frame_masks
            if on_progress and (frame_idx + 1) % 10 == 0:
                on_progress(n_frames + frame_idx + 1, n_frames * 2)

        predictor.reset_state(inference_state)

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
            # Encode as RLE for memory efficiency
            masks_by_label[label][fidx] = _encode_rle(merged)

    if on_progress:
        on_progress(n_frames * 2, n_frames * 2)

    return masks_by_label


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Execute Pass 1: SAM2 video segmentation."""
    import torch

    logger.info("Pass 1: SAM2 segmentation starting")

    tip_points = _load_tip_points(data.session_dir)
    if not tip_points:
        logger.warning("No tip init points found, skipping SAM2 pass")
        return

    logger.info("Tip init points: %s", {k: (v[0], v[1]) for k, v in tip_points.items()})

    # Load SAM2 model
    from app.config import SAM2_MODEL_PATH, SAM2_CONFIG_PATH

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        logger.error("sam2 package not installed, skipping Pass 1")
        return

    if on_progress:
        on_progress("pass1_sam2", 0, 1, "Loading SAM2 model")

    predictor = build_sam2_video_predictor(
        config_file=SAM2_CONFIG_PATH,
        ckpt_path=SAM2_MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        offload_video_model_to_cpu=True,
        offload_state_to_cpu=True,
    )

    try:
        # Segment on-axis view
        if on_progress:
            on_progress("pass1_sam2", 0, 4, "Segmenting on-axis view")
        logger.info("Segmenting on-axis view (%d frames)", len(data.on_frames))

        # Filter tip points for on-axis (use all points for now)
        data.on_masks = _segment_view(
            predictor,
            data.on_frames,
            tip_points,
            on_progress=lambda c, t: on_progress("pass1_sam2", c, t, "SAM2 on-axis") if on_progress else None,
        )
        logger.info("On-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.on_masks.items()})

        if on_progress:
            on_progress("pass1_sam2", 2, 4, "Segmenting off-axis view")

        # Segment off-axis view
        logger.info("Segmenting off-axis view (%d frames)", len(data.off_frames))
        data.off_masks = _segment_view(
            predictor,
            data.off_frames,
            tip_points,
            on_progress=lambda c, t: on_progress("pass1_sam2", c, t, "SAM2 off-axis") if on_progress else None,
        )
        logger.info("Off-axis masks: %s", {k: sum(1 for m in v if m is not None) for k, v in data.off_masks.items()})

    finally:
        # Unload SAM2 to free GPU memory
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM2 model unloaded, GPU memory freed")

    if on_progress:
        on_progress("pass1_sam2", 4, 4, "SAM2 segmentation complete")
    logger.info("Pass 1 complete")
