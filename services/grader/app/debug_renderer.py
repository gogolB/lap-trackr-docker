"""Debug visualization renderer for pipeline stages.

Renders per-pass overlay videos so you can visually inspect what each
stage produced. Activated via the _DEBUG_RENDER env var or --debug flag.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("grader.debug_renderer")

_MASK_COLORS = {
    "green_tip": (0, 255, 0),      # green
    "pink_tip": (180, 105, 255),    # pink/magenta
}
_TRACK_COLORS = {
    "green_tip": (0, 255, 0),
    "pink_tip": (255, 100, 220),
}
_DEFAULT_COLOR = (0, 200, 255)
_MASK_ALPHA = 0.4


def _create_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Create a VideoWriter with avc1 → mp4v fallback."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height),
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create VideoWriter for '{path}'")
    return writer


def render_segmentation_video(
    frames: list[np.ndarray],
    masks: dict[str, list[np.ndarray | None]],
    output_path: str,
    fps: float,
    camera_name: str = "",
    tip_points: dict[str, tuple[float, float, int]] | None = None,
) -> str:
    """Render a video with segmentation mask overlays.

    Each instrument's mask is shown as a colored transparent overlay,
    with mask centroid marked by a crosshair.  If tip_points is provided,
    the original click locations are drawn as diamond markers on the
    frames where they were placed.
    """
    from app.passes.pass1_sam2 import _decode_rle

    if not frames:
        logger.warning("No frames for segmentation debug video")
        return ""

    h, w = frames[0].shape[:2]
    n_frames = len(frames)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = _create_writer(output, w, h, fps)
    try:
        for fidx in range(n_frames):
            canvas = frames[fidx].copy()

            # Draw tip_init click points (diamond marker on the init frame)
            if tip_points:
                for label, (tx, ty, init_frame) in tip_points.items():
                    if fidx == init_frame:
                        color = _MASK_COLORS.get(label, _DEFAULT_COLOR)
                        px, py = int(round(tx)), int(round(ty))
                        # Diamond shape
                        size = 18
                        pts = np.array([
                            [px, py - size], [px + size, py],
                            [px, py + size], [px - size, py],
                        ], dtype=np.int32)
                        cv2.polylines(canvas, [pts], True, (255, 255, 255), 4, cv2.LINE_AA)
                        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
                        # Label
                        text = f"INIT {label} ({px}, {py})"
                        cv2.putText(canvas, text, (px + 22, py - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(canvas, text, (px + 22, py - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

            for label, mask_list in masks.items():
                if fidx >= len(mask_list) or mask_list[fidx] is None:
                    continue

                mask = _decode_rle(mask_list[fidx], (h, w))
                color = _MASK_COLORS.get(label, _DEFAULT_COLOR)

                # Semi-transparent mask overlay
                overlay = canvas.copy()
                overlay[mask > 0] = color
                cv2.addWeighted(overlay, _MASK_ALPHA, canvas, 1 - _MASK_ALPHA, 0, canvas)

                # Mask contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)

                # Centroid crosshair
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    size = 15
                    cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, 2, cv2.LINE_AA)
                    cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

                    # Label text
                    area = int(np.sum(mask > 0))
                    text = f"{label} area={area}"
                    cv2.putText(canvas, text, (cx + 12, cy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(canvas, text, (cx + 12, cy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Frame info
            info = f"Frame {fidx}/{n_frames}  {camera_name}  SEGMENTATION"
            cv2.putText(canvas, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(canvas)
    finally:
        writer.release()

    logger.info("Wrote segmentation debug video: %s", output)
    return str(output)


def render_cotracker_video(
    frames: list[np.ndarray],
    tracks: dict[str, np.ndarray],
    visibility: dict[str, np.ndarray],
    masks: dict[str, list[np.ndarray | None]],
    output_path: str,
    fps: float,
    camera_name: str = "",
    visibility_threshold: float = 0.3,
) -> str:
    """Render a video with CoTracker tracking points and optional mask underlay.

    Shows:
    - Faint mask underlay from Pass 1 (context)
    - CoTracker tracked point per instrument with trail
    - Visibility score next to each point
    - Red X where visibility is below threshold
    """
    from app.passes.pass1_sam2 import _decode_rle

    if not frames:
        logger.warning("No frames for CoTracker debug video")
        return ""

    h, w = frames[0].shape[:2]
    n_frames = len(frames)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    trail_length = 30
    trails: dict[str, list[tuple[int, int]]] = {label: [] for label in tracks}

    writer = _create_writer(output, w, h, fps)
    try:
        for fidx in range(n_frames):
            canvas = frames[fidx].copy()

            # Faint mask underlay from Pass 1
            for label, mask_list in masks.items():
                if fidx >= len(mask_list) or mask_list[fidx] is None:
                    continue
                mask = _decode_rle(mask_list[fidx], (h, w))
                color = _MASK_COLORS.get(label, _DEFAULT_COLOR)
                overlay = canvas.copy()
                overlay[mask > 0] = color
                cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

            # CoTracker points
            for label, positions in tracks.items():
                color = _TRACK_COLORS.get(label, _DEFAULT_COLOR)
                vis = visibility.get(label)
                v = float(vis[fidx]) if vis is not None and fidx < len(vis) else 0.0

                if fidx >= len(positions):
                    continue

                x, y = float(positions[fidx, 0]), float(positions[fidx, 1])
                if np.isnan(x) or np.isnan(y):
                    continue

                px, py = int(round(x)), int(round(y))

                if v >= visibility_threshold:
                    # Valid track point
                    trails[label].append((px, py))
                    if len(trails[label]) > trail_length:
                        trails[label] = trails[label][-trail_length:]

                    # Draw trail
                    if len(trails[label]) >= 2:
                        for i in range(1, len(trails[label])):
                            alpha = i / len(trails[label])
                            thickness = max(1, int(3 * alpha))
                            pt1 = trails[label][i - 1]
                            pt2 = trails[label][i]
                            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

                    # Point marker
                    cv2.circle(canvas, (px, py), 8, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 10, (255, 255, 255), 2, cv2.LINE_AA)

                    # Label + visibility
                    text = f"{label} vis={v:.2f}"
                    cv2.putText(canvas, text, (px + 14, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(canvas, text, (px + 14, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                else:
                    # Below threshold — red X
                    size = 10
                    cv2.line(canvas, (px - size, py - size), (px + size, py + size),
                             (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.line(canvas, (px - size, py + size), (px + size, py - size),
                             (0, 0, 255), 3, cv2.LINE_AA)
                    text = f"{label} vis={v:.2f} LOW"
                    cv2.putText(canvas, text, (px + 14, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Frame info
            info = f"Frame {fidx}/{n_frames}  {camera_name}  COTRACKER"
            cv2.putText(canvas, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(canvas)
    finally:
        writer.release()

    logger.info("Wrote CoTracker debug video: %s", output)
    return str(output)
