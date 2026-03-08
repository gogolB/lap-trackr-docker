"""Shared tip initialization loading for Pass 1 (SAM2 and SAM3).

Resolves tip_init.json click coordinates to the correct sampled frame index
by reading frame index manifests (tip_init_samples.json or *_export.json).
Also provides HSV color validation for debugging incorrect tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("grader.passes.tip_loader")

_LABEL_OBJ_IDS = {"green_tip": 1, "pink_tip": 2}

# Expected HSV ranges for instrument tip colors
_HSV_RANGES = {
    # Green instruments read hue ~10-35 under warm surgical lighting (not pure green 60)
    "green_tip": {"h_low": 8, "h_high": 85, "s_min": 40, "v_min": 40},
    "pink_tip": {"h_low": 140, "h_high": 180, "s_min": 25, "v_min": 70},
}


def _load_frame_manifest(session_dir: Path) -> dict[str, dict]:
    """Load frame index manifest from tip_init_samples.json or *_export.json.

    Returns {filename: {"camera": str, "frame_idx": int}}.
    """
    # Primary: tip_init_samples.json (written by the UI when user clicks tips)
    manifest_path = session_dir / "tip_init_samples.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            if isinstance(data, dict):
                logger.info("Loaded frame manifest from tip_init_samples.json (%d entries)", len(data))
                return data
        except Exception:
            logger.warning("Failed to parse tip_init_samples.json", exc_info=True)

    # Fallback: aggregate from *_export.json files
    aggregated: dict[str, dict] = {}
    for camera_name in ("on_axis", "off_axis"):
        export_path = session_dir / f"{camera_name}_export.json"
        if not export_path.exists():
            continue
        try:
            export_meta = json.loads(export_path.read_text())
        except Exception:
            logger.warning("Failed to parse %s", export_path, exc_info=True)
            continue
        for entry in export_meta.get("sample_frames", []):
            filename = entry.get("filename")
            frame_idx = entry.get("frame_idx")
            if filename is not None and frame_idx is not None:
                aggregated[str(filename)] = {
                    "camera": camera_name,
                    "frame_idx": int(frame_idx),
                }

    if aggregated:
        logger.info("Loaded frame manifest from export metadata (%d entries)", len(aggregated))
    else:
        logger.warning("No frame manifest found (tip_init_samples.json or *_export.json)")

    return aggregated


def _resolve_frame_index(
    filename: str,
    camera_name: str,
    manifest: dict[str, dict],
) -> int | None:
    """Resolve the original video frame index for a tip_init sample filename."""
    entry = manifest.get(filename)
    if entry is None:
        return None

    cam = entry.get("camera", camera_name)
    if camera_name not in filename and cam != camera_name:
        return None

    try:
        return int(entry["frame_idx"])
    except (KeyError, ValueError, TypeError):
        return None


def validate_tip_color(
    frame: np.ndarray,
    x: float,
    y: float,
    label: str,
) -> bool:
    """Sample the pixel at (x,y) on a frame and check if it matches the expected color.

    Logs the actual HSV values for debugging.  Returns True if the color is plausible.
    """
    h, w = frame.shape[:2]
    px, py = int(round(x)), int(round(y))

    if px < 0 or px >= w or py < 0 or py >= h:
        logger.warning("Tip point (%d, %d) out of bounds (%dx%d) for %s", px, py, w, h, label)
        return False

    # Sample a small region (5x5) for robustness against noise
    y1, y2 = max(0, py - 2), min(h, py + 3)
    x1, x2 = max(0, px - 2), min(w, px + 3)
    region = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = hsv.mean(axis=(0, 1))

    logger.info(
        "Tip color check at (%d, %d) for %s: HSV=(%.0f, %.0f, %.0f)",
        px, py, label, mean_h, mean_s, mean_v,
    )

    ranges = _HSV_RANGES.get(label)
    if ranges is None:
        return True

    if label == "pink_tip":
        # Pink wraps around hue=0: either >= 140 or <= 10
        hue_ok = mean_h >= ranges["h_low"] or mean_h <= 10
    else:
        hue_ok = ranges["h_low"] <= mean_h <= ranges["h_high"]

    sat_ok = mean_s >= ranges["s_min"]
    val_ok = mean_v >= ranges["v_min"]
    is_valid = hue_ok and sat_ok and val_ok

    if not is_valid:
        logger.warning(
            "Color validation FAILED for %s at (%d, %d): HSV=(%.0f, %.0f, %.0f) — "
            "hue_ok=%s, sat_ok=%s, val_ok=%s",
            label, px, py, mean_h, mean_s, mean_v, hue_ok, sat_ok, val_ok,
        )
    else:
        logger.info("Color validation PASSED for %s", label)

    return is_valid


def load_tip_points(
    session_dir: Path,
    camera_name: str,
    sample_interval: int = 1,
    n_frames: int | None = None,
    frame_indices: list[int] | None = None,
) -> tuple[dict[str, tuple[float, float, int]], bool]:
    """Load tip init points for a specific camera view, resolving frame indices.

    Reads tip_init.json (or tip_detections.json fallback) and resolves the
    original frame index from tip_init_samples.json or *_export.json, then
    maps to the sampled frame position.

    Parameters
    ----------
    session_dir : Path
        Session directory containing tip_init.json and frame manifests.
    camera_name : str
        Camera view to load points for ("on_axis" or "off_axis").
    sample_interval : int
        Frame sampling interval used when loading video frames.
    n_frames : int, optional
        Number of frames available for this view (used to clamp indices).
    frame_indices : list[int], optional
        Actual original frame indices loaded by the sampler (from load_svo2).
        When provided, uses exact index lookup instead of arithmetic
        approximation, which is critical when labeled frames are injected
        into the sample set.

    Returns
    -------
    ({label: (x, y, sampled_frame_idx)}, used_fallback)
    """
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

    # Load frame index manifest
    frame_manifest = _load_frame_manifest(session_dir)

    # Best detection per label: (x, y, sampled_idx, confidence)
    best: dict[str, tuple[float, float, int, float]] = {}

    for filename, detections in tip_data.items():
        # Filter to only this camera's samples
        if camera_name not in filename:
            continue

        # Resolve the original video frame index
        original_idx = _resolve_frame_index(filename, camera_name, frame_manifest)
        if original_idx is None:
            logger.warning(
                "Could not resolve frame index for '%s' — skipping (no manifest entry)",
                filename,
            )
            continue

        # Map original frame index to sampled frame position
        if frame_indices is not None:
            # Exact lookup — labeled frames are injected into the sample set
            try:
                sampled_idx = frame_indices.index(original_idx)
            except ValueError:
                # Frame wasn't loaded (shouldn't happen if extra_frames was used)
                import bisect
                sampled_idx = bisect.bisect_left(frame_indices, original_idx)
                sampled_idx = min(sampled_idx, len(frame_indices) - 1)
                logger.warning(
                    "Frame %d not in loaded set, snapped to nearest: index %d (frame %d)",
                    original_idx, sampled_idx, frame_indices[sampled_idx],
                )
        else:
            # Arithmetic fallback when frame_indices not available
            sampled_idx = max(0, int(round(original_idx / max(sample_interval, 1))))

        # Clamp to available frames
        if n_frames is not None:
            sampled_idx = min(sampled_idx, n_frames - 1)

        logger.info(
            "Resolved %s: original_frame=%d -> sampled_frame=%d (interval=%d, exact=%s)",
            filename, original_idx, sampled_idx, sample_interval,
            frame_indices is not None,
        )

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
            if prev is None or confidence > prev[3]:
                best[label] = (x, y, sampled_idx, confidence)

    result = {label: (x, y, idx) for label, (x, y, idx, _) in best.items()}

    if result:
        logger.info(
            "Loaded %s tip points for %s: %s",
            len(result), camera_name,
            {k: (f"({v[0]:.1f}, {v[1]:.1f})", f"frame={v[2]}") for k, v in result.items()},
        )
    else:
        logger.warning("No tip points found for camera %s", camera_name)

    return result, used_fallback
