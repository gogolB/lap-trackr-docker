"""Render tracking overlay videos from detections."""

from __future__ import annotations

import csv
import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.backends.base import Detection

logger = logging.getLogger("grader.tracking_renderer")

_COLORS: dict[str, tuple[int, int, int]] = {
    "left_tip": (0, 255, 0),
    "right_tip": (255, 100, 0),
}
_DEFAULT_COLOR = (0, 200, 255)


def render_tracking_video(
    frames: list[np.ndarray],
    detections: list[list[Detection]],
    output_path: str,
    fps: float,
    trail_length: int = 30,
    on_progress: Callable[[int, int], None] | None = None,
) -> str:
    """Render a tracking overlay MP4 from frames and 2D detections."""
    if not frames:
        raise ValueError("No frames provided for tracking video rendering")

    frame_count = min(len(frames), len(detections))
    if frame_count == 0:
        raise ValueError("No detection frames provided for tracking video rendering")
    if len(frames) != len(detections):
        logger.warning(
            "Frame/detection count mismatch for %s: frames=%d detections=%d; rendering %d",
            output_path,
            len(frames),
            len(detections),
            frame_count,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    video_fps = float(fps) if fps and fps > 0 else 30.0
    writer = _create_writer(output, w, h, video_fps)

    trails: dict[str, deque[tuple[int, int]]] = {}
    try:
        for idx, (frame, frame_detections) in enumerate(
            zip(frames[:frame_count], detections[:frame_count]),
            start=1,
        ):
            canvas = frame.copy()
            current = _select_best_detections(frame_detections)

            for det in current.values():
                px = int(round(det.x))
                py = int(round(det.y))
                trails.setdefault(det.label, deque(maxlen=max(1, trail_length))).append((px, py))

            for label, points in trails.items():
                if len(points) < 2:
                    continue
                color = _COLORS.get(label, _DEFAULT_COLOR)
                poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [poly], False, color, 2, cv2.LINE_AA)

            for det in current.values():
                color = _COLORS.get(det.label, _DEFAULT_COLOR)
                px = int(round(det.x))
                py = int(round(det.y))
                cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)

                text = f"{det.label} {det.confidence:.2f}"
                origin = (px + 8, max(18, py - 8))
                cv2.putText(
                    canvas,
                    text,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    text,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            writer.write(canvas)
            if on_progress and (idx == frame_count or idx % 10 == 0):
                on_progress(idx, frame_count)
    finally:
        writer.release()

    logger.info("Rendered tracking overlay video: %s", output)
    return str(output)


def write_detection_csv(
    detections: list[list[Detection]],
    output_path: str,
    fps: float,
    camera_name: str,
) -> str:
    """Write per-frame tip detections to CSV for downstream playback/analysis."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_fps = float(fps) if fps and fps > 0 else 30.0

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_idx", "timestamp_s", "camera", "label", "x", "y", "confidence"]
        )
        for frame_idx, frame_detections in enumerate(detections):
            timestamp = frame_idx / frame_fps
            for det in frame_detections:
                writer.writerow(
                    [
                        frame_idx,
                        f"{timestamp:.6f}",
                        camera_name,
                        det.label,
                        f"{det.x:.3f}",
                        f"{det.y:.3f}",
                        f"{det.confidence:.4f}",
                    ]
                )

    logger.info("Wrote tracking detections CSV: %s", output)
    return str(output)


def write_pose_csv(
    poses: list[dict[str, Any]],
    output_path: str,
) -> str:
    """Write calculated world-space tip positions to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "timestamp_s",
                "left_tip_x_m",
                "left_tip_y_m",
                "left_tip_z_m",
                "right_tip_x_m",
                "right_tip_y_m",
                "right_tip_z_m",
            ]
        )
        for pose in poses:
            left_tip = pose.get("left_tip") or [None, None, None]
            right_tip = pose.get("right_tip") or [None, None, None]
            writer.writerow(
                [
                    pose.get("frame_idx"),
                    pose.get("timestamp"),
                    _format_pose_value(left_tip[0]),
                    _format_pose_value(left_tip[1]),
                    _format_pose_value(left_tip[2]),
                    _format_pose_value(right_tip[0]),
                    _format_pose_value(right_tip[1]),
                    _format_pose_value(right_tip[2]),
                ]
            )

    logger.info("Wrote calculated pose CSV: %s", output)
    return str(output)


def _create_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Create a VideoWriter with avc1 and mp4v fallback."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        logger.info("avc1 codec unavailable for %s, falling back to mp4v", path)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to create VideoWriter for '{path}': both avc1 and mp4v codecs failed"
        )
    return writer


def _select_best_detections(detections: list[Detection]) -> dict[str, Detection]:
    """Keep the highest-confidence detection per label."""
    best: dict[str, Detection] = {}
    for det in detections:
        prev = best.get(det.label)
        if prev is None or det.confidence >= prev.confidence:
            best[det.label] = det
    return best


def _format_pose_value(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"
