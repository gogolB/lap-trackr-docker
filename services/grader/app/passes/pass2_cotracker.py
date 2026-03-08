"""Pass 2 — CoTracker v3 point refinement using SAM2 mask seeds."""

from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np

from app.passes.pass_data import PassData
from app.passes.pass1_sam2 import _decode_rle

logger = logging.getLogger("grader.passes.pass2_cotracker")

_N_SEED_POINTS = 10
_VIS_THRESHOLD = 0.5


def _sample_seed_points_from_masks(
    masks: dict[str, list[np.ndarray | None]],
    frame_shape: tuple[int, int],
    n_seeds: int = _N_SEED_POINTS,
) -> dict[str, np.ndarray]:
    """Sample seed points from mask centroids across keyframes.

    Returns {label: (N, 3) array of [frame_idx, x, y]}.
    """
    seeds: dict[str, np.ndarray] = {}
    h, w = frame_shape

    for label, mask_list in masks.items():
        # Find frames with valid masks
        valid_frames: list[tuple[int, float, float]] = []
        for fidx, rle in enumerate(mask_list):
            if rle is None:
                continue
            mask = _decode_rle(rle, (h, w))
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            valid_frames.append((fidx, cx, cy))

        if not valid_frames:
            continue

        # Select evenly-spaced keyframes
        n_select = min(n_seeds, len(valid_frames))
        indices = np.linspace(0, len(valid_frames) - 1, n_select, dtype=int)
        selected = [valid_frames[i] for i in indices]

        seed_array = np.array(
            [[float(fidx), x, y] for fidx, x, y in selected],
            dtype=np.float32,
        )
        seeds[label] = seed_array
        logger.info("Sampled %d seed points for %s", len(seed_array), label)

    return seeds


def _track_view(
    model,
    frames: list[np.ndarray],
    seeds: dict[str, np.ndarray],
    chunk_size: int,
    overlap: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Track seed points through frames using CoTracker in chunks.

    Returns (tracks, visibility) where:
      tracks: {label: (T, 2)} visibility-weighted median tip position
      visibility: {label: (T,)} confidence scores
    """
    import torch

    n_frames = len(frames)
    tracks_out: dict[str, np.ndarray] = {}
    vis_out: dict[str, np.ndarray] = {}

    for label, seed_pts in seeds.items():
        # Process in chunks with overlap
        all_tracks = np.full((n_frames, 2), np.nan, dtype=np.float32)
        all_vis = np.zeros(n_frames, dtype=np.float32)
        weight_acc = np.zeros(n_frames, dtype=np.float32)

        chunk_start = 0
        chunk_idx = 0
        total_chunks = max(1, (n_frames - overlap) // (chunk_size - overlap) + 1)

        while chunk_start < n_frames:
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_frames = frames[chunk_start:chunk_end]
            chunk_len = len(chunk_frames)

            # Adjust seed frame indices for this chunk
            chunk_seeds = seed_pts.copy()
            chunk_seeds[:, 0] = np.clip(chunk_seeds[:, 0] - chunk_start, 0, chunk_len - 1)

            # Build video tensor: (1, T, 3, H, W)
            ct_device = os.environ.get("_GRADER_DEVICE", "")
            if not ct_device:
                ct_device = "cuda" if torch.cuda.is_available() else "cpu"

            video = np.stack(chunk_frames)
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()
            if ct_device != "cpu":
                video_tensor = video_tensor.to(ct_device)

            queries = torch.from_numpy(chunk_seeds).float().unsqueeze(0)
            if ct_device != "cpu":
                queries = queries.to(ct_device)

            with torch.no_grad():
                pred = model(video_tensor, queries=queries)

            # Extract tracks and visibility
            if isinstance(pred, tuple) and len(pred) >= 2:
                pred_tracks, pred_vis = pred[0], pred[1]
            elif isinstance(pred, dict):
                pred_tracks = pred["tracks"]
                pred_vis = pred["visibility"]
            else:
                pred_tracks = pred.tracks
                pred_vis = pred.visibility

            pred_tracks = pred_tracks[0].cpu().numpy()  # (T_chunk, N, 2)
            pred_vis = pred_vis[0].cpu().numpy()  # (T_chunk, N)

            # Compute visibility-weighted median position per frame
            for t in range(chunk_len):
                global_t = chunk_start + t
                vis_scores = pred_vis[t]  # (N,)
                valid = vis_scores > _VIS_THRESHOLD
                if not np.any(valid):
                    continue

                pts = pred_tracks[t][valid]  # (M, 2)
                ws = vis_scores[valid]

                # Weighted median: use visibility as weights
                total_w = np.sum(ws)
                wx = np.sum(pts[:, 0] * ws) / total_w
                wy = np.sum(pts[:, 1] * ws) / total_w
                mean_vis = float(np.mean(ws))

                # Blend with existing (for overlap regions)
                if weight_acc[global_t] > 0:
                    # Weighted blend in overlap region
                    old_w = weight_acc[global_t]
                    new_w = mean_vis
                    blend = new_w / (old_w + new_w)
                    all_tracks[global_t, 0] = (1 - blend) * all_tracks[global_t, 0] + blend * wx
                    all_tracks[global_t, 1] = (1 - blend) * all_tracks[global_t, 1] + blend * wy
                    all_vis[global_t] = max(all_vis[global_t], mean_vis)
                    weight_acc[global_t] += new_w
                else:
                    all_tracks[global_t] = [wx, wy]
                    all_vis[global_t] = mean_vis
                    weight_acc[global_t] = mean_vis

            chunk_idx += 1
            if on_progress:
                on_progress(chunk_idx, total_chunks)

            # Advance by (chunk_size - overlap) for the next chunk
            chunk_start += chunk_size - overlap
            if chunk_end >= n_frames:
                break

        tracks_out[label] = all_tracks
        vis_out[label] = all_vis

    return tracks_out, vis_out


def run(
    data: PassData,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """Execute Pass 2: CoTracker v3 point refinement."""
    import torch

    from app.config import COTRACKER_CHUNK_SIZE, COTRACKER_OVERLAP

    logger.info("Pass 2: CoTracker point refinement starting")

    if not data.on_masks and not data.off_masks:
        logger.warning("No masks from Pass 1, skipping CoTracker pass")
        return

    # Load CoTracker model — prefer env override (offline CLI), fall back to DB
    model_path = os.environ.get("_COTRACKER_MODEL_PATH", "")
    if not model_path:
        try:
            from app.db import get_active_model_info

            info = get_active_model_info(model_type="cotracker")
            if info is None:
                logger.warning("No active CoTracker model, skipping Pass 2")
                return
            model_path = info["file_path"]
        except Exception as exc:
            logger.warning("DB lookup failed and _COTRACKER_MODEL_PATH not set: %s", exc)
            return

    logger.info("Loading CoTracker from %s", model_path)

    if on_progress:
        on_progress("pass2_cotracker", 0, 1, "Loading CoTracker model")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if hasattr(checkpoint, "eval"):
        model = checkpoint
    else:
        from cotracker.predictor import CoTrackerPredictor
        model = CoTrackerPredictor(checkpoint=model_path)

    device = os.environ.get("_GRADER_DEVICE", "")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("CoTracker using device: %s", device)

    if hasattr(model, "eval"):
        model.eval()
    if device != "cpu":
        model = model.to(device)

    try:
        # Sample seed points from SAM2 masks
        if data.on_frames:
            on_shape = data.on_frames[0].shape[:2]
            on_seeds = _sample_seed_points_from_masks(data.on_masks, on_shape)
        else:
            on_seeds = {}

        if data.off_frames:
            off_shape = data.off_frames[0].shape[:2]
            off_seeds = _sample_seed_points_from_masks(data.off_masks, off_shape)
        else:
            off_seeds = {}

        # Track on-axis
        if on_seeds and data.on_frames:
            if on_progress:
                on_progress("pass2_cotracker", 0, 4, "Tracking on-axis")
            logger.info("Tracking on-axis with %d labels", len(on_seeds))
            data.on_tracks, data.on_visibility = _track_view(
                model,
                data.on_frames,
                on_seeds,
                COTRACKER_CHUNK_SIZE,
                COTRACKER_OVERLAP,
                on_progress=lambda c, t: on_progress("pass2_cotracker", c, t, "CoTracker on-axis") if on_progress else None,
            )
            for label in data.on_tracks:
                valid = np.sum(data.on_visibility.get(label, np.array([])) > _VIS_THRESHOLD)
                logger.info("On-axis %s: %d/%d visible frames", label, valid, len(data.on_frames))

        # Track off-axis
        if off_seeds and data.off_frames:
            if on_progress:
                on_progress("pass2_cotracker", 2, 4, "Tracking off-axis")
            logger.info("Tracking off-axis with %d labels", len(off_seeds))
            data.off_tracks, data.off_visibility = _track_view(
                model,
                data.off_frames,
                off_seeds,
                COTRACKER_CHUNK_SIZE,
                COTRACKER_OVERLAP,
                on_progress=lambda c, t: on_progress("pass2_cotracker", c, t, "CoTracker off-axis") if on_progress else None,
            )
            for label in data.off_tracks:
                valid = np.sum(data.off_visibility.get(label, np.array([])) > _VIS_THRESHOLD)
                logger.info("Off-axis %s: %d/%d visible frames", label, valid, len(data.off_frames))

    finally:
        del model
        if device != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("CoTracker model unloaded, GPU memory freed")

    if on_progress:
        on_progress("pass2_cotracker", 4, 4, "CoTracker tracking complete")
    logger.info("Pass 2 complete")
