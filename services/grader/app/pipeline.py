"""Grading pipeline -- orchestrates all stages from SVO2 to metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from app.backends.base import Detection
from app.camera_transform import adjust_calibration
from app.color_detector import analyze_tip_frame
from app.config import FRAME_SAMPLE_INTERVAL, PIPELINE_MODE
from app.db import get_active_model_info
from app.exporter import _build_candidate_frame_indices
from app.metrics import calculate_metrics
from app.model_loader import get_backend
from app.pose_estimator import estimate_poses, estimate_poses_dual
from app.svo_loader import load_svo2
from app.tracking_renderer import (
    render_tracking_video,
    write_detection_csv,
    write_pose_csv,
)

ProgressCallback = Optional[Callable[[str, int, int, str], None]]

logger = logging.getLogger("grader.pipeline")
_TRACK_LABELS = ("green_tip", "pink_tip")
_QUERY_LABEL_ORDER = {
    "green_tip": 0,
    "pink_tip": 1,
    "left_tip": 0,
    "right_tip": 1,
}
_LEGACY_LABEL_MAP = {
    "left_tip": "green_tip",
    "right_tip": "pink_tip",
}
_MERGE_DISTANCE_RATIO = 0.03
_TRACKER_BLEND_WEIGHT = 0.8
_YOLO_BLEND_WEIGHT = 0.75
_COLOR_BLEND_WEIGHT = 0.9
_INTERPOLATE_MAX_GAP = 6
_INTERPOLATE_DISTANCE_RATIO = 0.08
_SMOOTH_WINDOW = 7
_SMOOTH_POLYORDER = 2


@dataclass(frozen=True)
class _DetectionMergeSummary:
    tracker_primary: int = 0
    yolo_primary: int = 0
    color_primary: int = 0
    missing: int = 0


@dataclass(frozen=True)
class _QueryPointSet:
    points: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class _TipInitCandidate:
    label: str
    frame_idx: int
    x: float
    y: float
    confidence: float


def _results_dir(job: dict) -> Path:
    """Return the per-session results directory for generated artifacts."""
    on_axis_path = job.get("on_axis_path")
    if not on_axis_path:
        raise FileNotFoundError("Cannot derive results directory without on_axis_path")
    results_dir = Path(on_axis_path).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _load_calibration(job: dict, key: str = "calibration_path") -> dict | None:
    """Load calibration JSON from the path specified in the job, if any."""
    calib_path = job.get(key)
    if not calib_path:
        return None

    p = Path(calib_path)
    if not p.exists():
        logger.warning("Calibration file not found: %s", calib_path)
        return None

    try:
        calibration = json.loads(p.read_text())
        logger.info("Loaded calibration from %s", calib_path)
        return calibration
    except Exception as exc:
        logger.warning("Failed to parse calibration %s: %s", calib_path, exc)
        return None


def _canonical_tip_label(label: str | None, color: str | None = None) -> str | None:
    if label:
        normalized = _LEGACY_LABEL_MAP.get(label, label)
        if normalized in _TRACK_LABELS:
            return normalized
    if color == "green":
        return "green_tip"
    if color == "pink":
        return "pink_tip"
    return None


def _session_dir_from_job(job: dict) -> Path | None:
    tip_init_path = job.get("tip_init_path")
    if tip_init_path:
        return Path(tip_init_path).parent
    on_axis_path = job.get("on_axis_path")
    if on_axis_path:
        return Path(on_axis_path).parent
    return None


def _load_tip_init_sample_manifest(session_dir: Path | None) -> dict[str, dict[str, Any]]:
    if session_dir is None:
        return {}

    manifest_path = session_dir / "tip_init_samples.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            logger.warning("Failed to parse %s", manifest_path, exc_info=True)

    aggregated: dict[str, dict[str, Any]] = {}
    for camera_name in ("on_axis", "off_axis"):
        export_meta_path = session_dir / f"{camera_name}_export.json"
        if not export_meta_path.exists():
            continue
        try:
            export_meta = json.loads(export_meta_path.read_text())
        except Exception:
            logger.warning("Failed to parse %s", export_meta_path, exc_info=True)
            continue
        for entry in export_meta.get("sample_frames", []):
            filename = entry.get("filename")
            frame_idx = entry.get("frame_idx")
            if filename is None or frame_idx is None:
                continue
            aggregated[str(filename)] = {
                "camera": camera_name,
                "frame_idx": int(frame_idx),
            }

    return aggregated


def _match_sample_frame_index(
    session_dir: Path,
    camera_name: str,
    filename: str,
) -> int | None:
    try:
        import cv2
    except ImportError:
        return None

    sample_path = session_dir / filename
    mp4_path = session_dir / f"{camera_name}_left.mp4"
    if not sample_path.exists() or not mp4_path.exists():
        return None

    sample = cv2.imread(str(sample_path))
    if sample is None:
        return None

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        candidates = _build_candidate_frame_indices(total_frames) or list(range(max(total_frames, 0)))
        sample_small = cv2.resize(sample, (160, 90), interpolation=cv2.INTER_AREA)
        best_idx: int | None = None
        best_score: float | None = None
        for frame_idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame_small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            score = float(np.mean(np.abs(frame_small.astype(np.float32) - sample_small.astype(np.float32))))
            if best_score is None or score < best_score:
                best_idx = int(frame_idx)
                best_score = score
        if best_idx is not None:
            logger.info(
                "Recovered sample frame index for %s/%s -> %d (score=%.3f)",
                camera_name,
                filename,
                best_idx,
                best_score or 0.0,
            )
        return best_idx
    finally:
        cap.release()


def _resolve_sample_frame_index(
    session_dir: Path | None,
    camera_name: str,
    filename: str,
    manifest: dict[str, dict[str, Any]],
) -> int | None:
    entry = manifest.get(filename)
    if entry is not None:
        try:
            camera = str(entry.get("camera", camera_name))
            if camera_name in filename or camera == camera_name:
                return int(entry["frame_idx"])
        except Exception:
            logger.warning("Invalid sample manifest entry for %s", filename, exc_info=True)

    if session_dir is None:
        return None

    recovered = _match_sample_frame_index(session_dir, camera_name, filename)
    if recovered is not None:
        manifest[filename] = {
            "camera": camera_name,
            "frame_idx": recovered,
        }
    return recovered


def _normalize_tip_init_detections(
    detections: list[dict[str, Any]],
) -> dict[str, _TipInitCandidate]:
    normalized: dict[str, _TipInitCandidate] = {}
    for det in detections:
        label = _canonical_tip_label(det.get("label"), det.get("color"))
        if label is None:
            continue
        confidence = float(det.get("confidence", 1.0))
        candidate = _TipInitCandidate(
            label=label,
            frame_idx=0,
            x=float(det["x"]),
            y=float(det["y"]),
            confidence=confidence,
        )
        previous = normalized.get(label)
        if previous is None or candidate.confidence >= previous.confidence:
            normalized[label] = candidate
    return normalized


def _load_query_points(
    job: dict,
    camera_name: str | None = None,
    sample_interval: int = FRAME_SAMPLE_INTERVAL,
) -> _QueryPointSet | None:
    """Load query points from tip_init.json in the session directory."""
    tip_init_path = job.get("tip_init_path")

    # Try explicit path first, then derive from on_axis_path
    if not tip_init_path:
        on_axis_path = job.get("on_axis_path")
        if on_axis_path:
            tip_init_path = str(Path(on_axis_path).parent / "tip_init.json")

    if not tip_init_path or not Path(tip_init_path).exists():
        logger.info("No tip_init.json found, using default detection")
        return None

    try:
        tip_data = json.loads(Path(tip_init_path).read_text())
        session_dir = _session_dir_from_job(job)
        manifest = _load_tip_init_sample_manifest(session_dir)
        best_by_label: dict[str, _TipInitCandidate] = {}
        items = list(tip_data.items())
        if camera_name:
            filtered_items = [
                (filename, detections)
                for filename, detections in items
                if camera_name in filename
            ]
            if not filtered_items:
                logger.info("No tip-init samples found for camera=%s", camera_name)
                return None
            items = filtered_items

        for filename, detections in items:
            if camera_name is None:
                inferred_camera = "off_axis" if "off_axis" in filename else "on_axis"
            else:
                inferred_camera = camera_name
            frame_idx = _resolve_sample_frame_index(
                session_dir,
                inferred_camera,
                filename,
                manifest,
            )
            if frame_idx is None:
                continue

            sampled_frame_idx = max(0, int(round(frame_idx / max(sample_interval, 1))))
            normalized_detections = _normalize_tip_init_detections(detections)
            for label, candidate in normalized_detections.items():
                frame_candidate = _TipInitCandidate(
                    label=label,
                    frame_idx=sampled_frame_idx,
                    x=candidate.x,
                    y=candidate.y,
                    confidence=candidate.confidence,
                )
                previous = best_by_label.get(label)
                if previous is None or (
                    frame_candidate.confidence,
                    -frame_candidate.frame_idx,
                ) > (
                    previous.confidence,
                    -previous.frame_idx,
                ):
                    best_by_label[label] = frame_candidate

        ordered_candidates = [
            best_by_label[label]
            for label in _TRACK_LABELS
            if label in best_by_label
        ]
        if ordered_candidates:
            qp = np.array(
                [
                    [float(candidate.frame_idx), candidate.x, candidate.y]
                    for candidate in ordered_candidates
                ],
                dtype=np.float32,
            )
            labels = tuple(candidate.label for candidate in ordered_candidates)
            logger.info(
                "Loaded %d query points from tip_init.json for camera=%s using labels=%s",
                len(ordered_candidates),
                camera_name or "any",
                labels,
            )
            return _QueryPointSet(points=qp, labels=labels)
    except Exception as exc:
        logger.warning("Failed to load tip_init.json: %s", exc)

    return None


def _load_stereo_calibration(job: dict) -> dict | None:
    """Load stereo calibration from session dir."""
    stereo_path = job.get("stereo_calibration_path")
    if not stereo_path:
        on_axis_path = job.get("on_axis_path")
        if on_axis_path:
            stereo_path = str(Path(on_axis_path).parent / "stereo_calibration.json")

    if not stereo_path or not Path(stereo_path).exists():
        return None

    try:
        data = json.loads(Path(stereo_path).read_text())
        logger.info("Loaded stereo calibration from %s", stereo_path)
        return data
    except Exception as exc:
        logger.warning("Failed to load stereo calibration: %s", exc)
        return None


def _load_off_axis_calibration(job: dict) -> dict | None:
    """Load off-axis camera calibration from session dir."""
    on_axis_path = job.get("on_axis_path")
    if not on_axis_path:
        return None
    calib_path = str(Path(on_axis_path).parent / "calibration_off_axis.json")
    if not Path(calib_path).exists():
        return None
    try:
        data = json.loads(Path(calib_path).read_text())
        logger.info("Loaded off-axis calibration from %s", calib_path)
        return data
    except Exception as exc:
        logger.warning("Failed to load off-axis calibration: %s", exc)
        return None


def _empty_detection_frames(frame_count: int) -> list[list[Detection]]:
    return [[] for _ in range(frame_count)]


def _best_by_label(detections: list[Detection]) -> dict[str, Detection]:
    best: dict[str, Detection] = {}
    for det in detections:
        previous = best.get(det.label)
        if previous is None or det.confidence >= previous.confidence:
            best[det.label] = det
    return best


def _ordered_labels(labels: set[str]) -> list[str]:
    return sorted(labels, key=lambda label: (_QUERY_LABEL_ORDER.get(label, 99), label))


def _distance_limit(frame_shape: tuple[int, int], ratio: float = _MERGE_DISTANCE_RATIO) -> float:
    height, width = frame_shape
    return hypot(width, height) * ratio


def _distance(a: Detection, b: Detection) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def _blend_detections(primary: Detection, secondary: Detection, primary_weight: float) -> Detection:
    secondary_weight = max(0.0, 1.0 - primary_weight)
    total_weight = primary_weight + secondary_weight
    if total_weight <= 0:
        return primary
    x = (primary.x * primary_weight + secondary.x * secondary_weight) / total_weight
    y = (primary.y * primary_weight + secondary.y * secondary_weight) / total_weight
    confidence = max(primary.confidence, secondary.confidence)
    source_parts = [primary.source, secondary.source]
    source = "+".join(
        part
        for idx, part in enumerate(source_parts)
        if part and part not in source_parts[:idx]
    )
    return Detection(
        x=float(x),
        y=float(y),
        confidence=float(confidence),
        label=primary.label,
        source=source or primary.source,
    )


def _detections_from_color_analysis(analysis: dict) -> list[Detection]:
    detections: list[Detection] = []
    for item in analysis.get("detections", []):
        detections.append(
            Detection(
                x=float(item["x"]),
                y=float(item["y"]),
                confidence=float(item.get("confidence", 0.0)),
                label=str(item["label"]),
                source="color",
            )
        )
    return detections


def _detect_color_frames(
    frames: list[np.ndarray],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[Detection]]:
    detections: list[list[Detection]] = []
    total = len(frames)
    for idx, frame in enumerate(frames, start=1):
        analysis = analyze_tip_frame(frame)
        detections.append(_detections_from_color_analysis(analysis))
        if on_progress and (idx == total or idx % 10 == 0):
            on_progress(idx, total)
    return detections


def _run_backend_stage(
    model_type: str,
    frames: list[np.ndarray],
    query_points: np.ndarray | None,
    query_labels: list[str] | tuple[str, ...] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    require_query_points: bool = False,
) -> tuple[list[list[Detection]], str | None]:
    if not frames:
        return [], f"No frames for {model_type} detection"

    if require_query_points and (query_points is None or len(query_points) == 0):
        return _empty_detection_frames(len(frames)), "Tip initialization required for CoTracker; skipping tracker stage"

    info = get_active_model_info(model_type=model_type)
    if info is None:
        return _empty_detection_frames(len(frames)), f"No active {model_type} model configured"

    backend = get_backend(model_type)
    if backend.__class__.__name__ == "PlaceholderBackend":
        return _empty_detection_frames(len(frames)), f"Active {model_type} model could not be loaded; using empty detections"

    detections = backend.detect(
        frames,
        query_points=query_points,
        query_labels=query_labels,
        on_progress=on_progress,
    )
    return detections, None


def _refine_detection_stream(
    detections: list[list[Detection]],
    frame_shape: tuple[int, int],
) -> tuple[list[list[Detection]], dict[str, int]]:
    labels = _ordered_labels(
        {
            det.label
            for frame_detections in detections
            for det in frame_detections
            if det.label
        }
    )
    if not labels:
        return detections, {"interpolated": 0, "smoothed": 0}

    refined_by_label: dict[str, list[Detection | None]] = {
        label: [_best_by_label(frame).get(label) for frame in detections]
        for label in labels
    }
    summary = {"interpolated": 0, "smoothed": 0}

    for label in labels:
        series = refined_by_label[label]
        interpolated, interpolated_count = _interpolate_short_gaps(series, label, frame_shape)
        smoothed, smoothed_count = _smooth_detection_series(interpolated)
        refined_by_label[label] = smoothed
        summary["interpolated"] += interpolated_count
        summary["smoothed"] += smoothed_count

    refined_frames: list[list[Detection]] = []
    for frame_idx in range(len(detections)):
        frame_detections = [
            det
            for label in labels
            if (det := refined_by_label[label][frame_idx]) is not None
        ]
        refined_frames.append(frame_detections)

    return refined_frames, summary


def _interpolate_short_gaps(
    series: list[Detection | None],
    label: str,
    frame_shape: tuple[int, int],
    max_gap: int = _INTERPOLATE_MAX_GAP,
) -> tuple[list[Detection | None], int]:
    if max_gap <= 0 or not series:
        return list(series), 0

    refined = list(series)
    distance_limit = _distance_limit(frame_shape, _INTERPOLATE_DISTANCE_RATIO)
    created = 0
    idx = 0
    while idx < len(refined):
        if refined[idx] is not None:
            idx += 1
            continue
        gap_start = idx
        while idx < len(refined) and refined[idx] is None:
            idx += 1
        gap_end = idx
        gap_len = gap_end - gap_start
        prev_idx = gap_start - 1
        next_idx = gap_end
        if (
            gap_len == 0
            or gap_len > max_gap
            or prev_idx < 0
            or next_idx >= len(refined)
            or refined[prev_idx] is None
            or refined[next_idx] is None
        ):
            continue

        prev_det = refined[prev_idx]
        next_det = refined[next_idx]
        if prev_det is None or next_det is None:
            continue
        travel = hypot(next_det.x - prev_det.x, next_det.y - prev_det.y)
        if travel > distance_limit * (gap_len + 1):
            continue

        for step in range(1, gap_len + 1):
            alpha = step / (gap_len + 1)
            refined[gap_start + step - 1] = Detection(
                x=float(prev_det.x + (next_det.x - prev_det.x) * alpha),
                y=float(prev_det.y + (next_det.y - prev_det.y) * alpha),
                confidence=float(min(prev_det.confidence, next_det.confidence) * 0.6),
                label=label,
                source="interpolated",
            )
            created += 1

    return refined, created


def _smooth_detection_series(
    series: list[Detection | None],
) -> tuple[list[Detection | None], int]:
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return list(series), 0

    refined = list(series)
    smoothed = 0
    start = 0
    while start < len(refined):
        while start < len(refined) and refined[start] is None:
            start += 1
        if start >= len(refined):
            break
        end = start
        while end < len(refined) and refined[end] is not None:
            end += 1
        segment = refined[start:end]
        segment_len = len(segment)
        if segment_len >= 5:
            window = min(_SMOOTH_WINDOW, segment_len if segment_len % 2 == 1 else segment_len - 1)
            if window >= 5 and window > _SMOOTH_POLYORDER:
                x_values = np.array([det.x for det in segment], dtype=np.float64)
                y_values = np.array([det.y for det in segment], dtype=np.float64)
                smooth_x = savgol_filter(x_values, window_length=window, polyorder=_SMOOTH_POLYORDER, mode="interp")
                smooth_y = savgol_filter(y_values, window_length=window, polyorder=_SMOOTH_POLYORDER, mode="interp")
                for offset, det in enumerate(segment):
                    if det is None:
                        continue
                    refined[start + offset] = Detection(
                        x=float(smooth_x[offset]),
                        y=float(smooth_y[offset]),
                        confidence=det.confidence,
                        label=det.label,
                        source=det.source,
                    )
                smoothed += segment_len
        start = end

    return refined, smoothed


def _merge_frame_detections(
    tracker_detections: list[Detection],
    yolo_detections: list[Detection],
    color_detections: list[Detection],
    frame_shape: tuple[int, int],
) -> tuple[list[Detection], _DetectionMergeSummary]:
    tracker_map = _best_by_label(tracker_detections)
    yolo_map = _best_by_label(yolo_detections)
    color_map = _best_by_label(color_detections)
    labels = _ordered_labels(set(tracker_map) | set(yolo_map) | set(color_map))
    distance_limit = _distance_limit(frame_shape)

    merged: list[Detection] = []
    summary = _DetectionMergeSummary()

    for label in labels:
        tracker_det = tracker_map.get(label)
        yolo_det = yolo_map.get(label)
        color_det = color_map.get(label)
        chosen: Detection | None = None

        if tracker_det is not None:
            chosen = tracker_det
            summary = _DetectionMergeSummary(
                tracker_primary=summary.tracker_primary + 1,
                yolo_primary=summary.yolo_primary,
                color_primary=summary.color_primary,
                missing=summary.missing,
            )
            if yolo_det is not None and _distance(tracker_det, yolo_det) <= distance_limit:
                chosen = _blend_detections(tracker_det, yolo_det, _TRACKER_BLEND_WEIGHT)
            elif color_det is not None and _distance(tracker_det, color_det) <= distance_limit:
                chosen = _blend_detections(tracker_det, color_det, _COLOR_BLEND_WEIGHT)
        elif yolo_det is not None:
            chosen = yolo_det
            summary = _DetectionMergeSummary(
                tracker_primary=summary.tracker_primary,
                yolo_primary=summary.yolo_primary + 1,
                color_primary=summary.color_primary,
                missing=summary.missing,
            )
            if color_det is not None and _distance(yolo_det, color_det) <= distance_limit:
                chosen = _blend_detections(yolo_det, color_det, _YOLO_BLEND_WEIGHT)
        elif color_det is not None:
            chosen = color_det
            summary = _DetectionMergeSummary(
                tracker_primary=summary.tracker_primary,
                yolo_primary=summary.yolo_primary,
                color_primary=summary.color_primary + 1,
                missing=summary.missing,
            )
        else:
            summary = _DetectionMergeSummary(
                tracker_primary=summary.tracker_primary,
                yolo_primary=summary.yolo_primary,
                color_primary=summary.color_primary,
                missing=summary.missing + 1,
            )

        if chosen is not None:
            merged.append(chosen)

    return merged, summary


def _merge_detection_streams(
    tracker_detections: list[list[Detection]],
    yolo_detections: list[list[Detection]],
    color_detections: list[list[Detection]],
    frame_shape: tuple[int, int],
) -> tuple[list[list[Detection]], dict[str, int]]:
    frame_count = max(len(tracker_detections), len(yolo_detections), len(color_detections))
    merged: list[list[Detection]] = []
    totals = {"tracker_primary": 0, "yolo_primary": 0, "color_primary": 0, "missing": 0}

    for idx in range(frame_count):
        tracker_frame = tracker_detections[idx] if idx < len(tracker_detections) else []
        yolo_frame = yolo_detections[idx] if idx < len(yolo_detections) else []
        color_frame = color_detections[idx] if idx < len(color_detections) else []
        merged_frame, summary = _merge_frame_detections(
            tracker_frame,
            yolo_frame,
            color_frame,
            frame_shape,
        )
        merged.append(merged_frame)
        totals["tracker_primary"] += summary.tracker_primary
        totals["yolo_primary"] += summary.yolo_primary
        totals["color_primary"] += summary.color_primary
        totals["missing"] += summary.missing

    return merged, totals


def _has_any_detections(detections: list[list[Detection]]) -> bool:
    return any(frame for frame in detections)


def run_pipeline(job: dict, on_progress: ProgressCallback = None) -> dict[str, Any]:
    """Run the full grading pipeline on a session's SVO2 files.

    Parameters
    ----------
    job : dict
        Must contain ``session_id``, ``on_axis_path`` and ``off_axis_path``.
        Optionally ``calibration_path``, ``stereo_calibration_path``,
        ``tip_init_path`` for enhanced tracking.
    on_progress : callable, optional
        ``(stage, current, total, detail)`` callback for live progress updates.

    Returns
    -------
    dict
        ``{"metrics": {...}, "poses": [...]}``
    """

    def _progress(stage: str, current: int, total: int, detail: str = "") -> None:
        if on_progress:
            on_progress(stage, current, total, detail)

    on_axis_path: str = job["on_axis_path"]
    off_axis_path: str = job["off_axis_path"]
    results_dir = _results_dir(job)
    warnings: list[str] = []

    if not Path(on_axis_path).exists():
        raise FileNotFoundError(f"On-axis SVO2 not found: {on_axis_path}")
    if not Path(off_axis_path).exists():
        raise FileNotFoundError(f"Off-axis SVO2 not found: {off_axis_path}")

    # Load calibrations
    on_calibration = _load_calibration(job, "calibration_path")
    off_calibration = _load_off_axis_calibration(job)
    stereo_calibration = _load_stereo_calibration(job)
    on_query_set = _load_query_points(job, "on_axis")
    off_query_set = _load_query_points(job, "off_axis")
    camera_config = job.get("camera_config")

    on_calibration = adjust_calibration(on_calibration, camera_config, "on_axis")
    off_calibration = adjust_calibration(off_calibration, camera_config, "off_axis")

    # Stage 1 -- load SVO2 files and extract frames + depth maps.
    _progress("load_on_axis", 0, 1, "Opening on-axis recording")
    logger.info("Stage 1: Loading SVO2 from %s", on_axis_path)
    frames, depth_maps, fps, _ = load_svo2(
        on_axis_path,
        on_progress=lambda current, total: _progress(
            "load_on_axis",
            current,
            total,
            "Loading on-axis frames",
        ),
        camera_config=camera_config,
    )
    logger.info(
        "  Loaded %d frames (%d depth maps) at %.1f fps",
        len(frames),
        len(depth_maps),
        fps,
    )
    _progress("load_on_axis", 1, 1, f"Loaded {len(frames)} sampled frames")

    _progress("load_off_axis", 0, 1, "Opening off-axis recording")
    logger.info("Stage 1b: Loading off-axis SVO2 from %s", off_axis_path)
    off_frames, off_depth, off_fps, _ = load_svo2(
        off_axis_path,
        on_progress=lambda current, total: _progress(
            "load_off_axis",
            current,
            total,
            "Loading off-axis frames",
        ),
        camera_config=camera_config,
    )
    logger.info("  Off-axis: %d frames at %.1f fps", len(off_frames), off_fps)
    _progress("load_off_axis", 1, 1, f"Loaded {len(off_frames)} sampled frames")

    # Stage 2 -- multi-pass 2D detection/tracking.
    logger.info("Stage 2: Running multi-pass 2D tracking")

    _progress("track_on_axis", 0, max(len(frames), 1), "CoTracker on-axis")
    on_tracker_detections, tracker_warning = _run_backend_stage(
        "cotracker",
        frames,
        query_points=on_query_set.points if on_query_set else None,
        query_labels=on_query_set.labels if on_query_set else None,
        on_progress=lambda current, total: _progress(
            "track_on_axis",
            current,
            total,
            "CoTracker on-axis",
        ),
        require_query_points=True,
    )
    if tracker_warning:
        warnings.append(f"On-axis tracker: {tracker_warning}")

    _progress("track_off_axis", 0, max(len(off_frames), 1), "CoTracker off-axis")
    off_tracker_detections, off_tracker_warning = _run_backend_stage(
        "cotracker",
        off_frames,
        query_points=off_query_set.points if off_query_set else None,
        query_labels=off_query_set.labels if off_query_set else None,
        on_progress=lambda current, total: _progress(
            "track_off_axis",
            current,
            total,
            "CoTracker off-axis",
        ),
        require_query_points=True,
    )
    if off_tracker_warning:
        warnings.append(f"Off-axis tracker: {off_tracker_warning}")

    _progress("detect_on_axis_yolo", 0, max(len(frames), 1), "YOLO on-axis")
    on_yolo_detections, on_yolo_warning = _run_backend_stage(
        "yolo",
        frames,
        query_points=None,
        on_progress=lambda current, total: _progress(
            "detect_on_axis_yolo",
            current,
            total,
            "YOLO on-axis",
        ),
    )
    if on_yolo_warning:
        warnings.append(f"On-axis YOLO: {on_yolo_warning}")

    _progress("detect_off_axis_yolo", 0, max(len(off_frames), 1), "YOLO off-axis")
    off_yolo_detections, off_yolo_warning = _run_backend_stage(
        "yolo",
        off_frames,
        query_points=None,
        on_progress=lambda current, total: _progress(
            "detect_off_axis_yolo",
            current,
            total,
            "YOLO off-axis",
        ),
    )
    if off_yolo_warning:
        warnings.append(f"Off-axis YOLO: {off_yolo_warning}")

    _progress("detect_on_axis_color", 0, max(len(frames), 1), "Color fallback on-axis")
    on_color_detections = _detect_color_frames(
        frames,
        on_progress=lambda current, total: _progress(
            "detect_on_axis_color",
            current,
            total,
            "Color fallback on-axis",
        ),
    )
    _progress("detect_off_axis_color", 0, max(len(off_frames), 1), "Color fallback off-axis")
    off_color_detections = _detect_color_frames(
        off_frames,
        on_progress=lambda current, total: _progress(
            "detect_off_axis_color",
            current,
            total,
            "Color fallback off-axis",
        ),
    )

    on_detections, on_merge_summary = _merge_detection_streams(
        on_tracker_detections,
        on_yolo_detections,
        on_color_detections,
        frames[0].shape[:2] if frames else (1, 1),
    )
    off_detections, off_merge_summary = _merge_detection_streams(
        off_tracker_detections,
        off_yolo_detections,
        off_color_detections,
        off_frames[0].shape[:2] if off_frames else (1, 1),
    )
    logger.info("  On-axis merged detections summary: %s", on_merge_summary)
    logger.info("  Off-axis merged detections summary: %s", off_merge_summary)
    warnings.append(f"On-axis detection mix: {on_merge_summary}")
    warnings.append(f"Off-axis detection mix: {off_merge_summary}")

    on_detections, on_refine_summary = _refine_detection_stream(
        on_detections,
        frames[0].shape[:2] if frames else (1, 1),
    )
    off_detections, off_refine_summary = _refine_detection_stream(
        off_detections,
        off_frames[0].shape[:2] if off_frames else (1, 1),
    )
    logger.info("  On-axis refined detections summary: %s", on_refine_summary)
    logger.info("  Off-axis refined detections summary: %s", off_refine_summary)
    warnings.append(f"On-axis refinement: {on_refine_summary}")
    warnings.append(f"Off-axis refinement: {off_refine_summary}")

    tracking_videos: list[str] = []
    tracking_csvs: list[str] = []

    # Persist per-frame 2D detections for playback/reanalysis.
    try:
        tracking_csvs.append(
            write_detection_csv(
                on_detections,
                str(results_dir / "tracking_on_axis.csv"),
                fps,
                "on_axis",
            )
        )
        tracking_csvs.append(
            write_detection_csv(
                off_detections,
                str(results_dir / "tracking_off_axis.csv"),
                off_fps,
                "off_axis",
            )
        )
        if _has_any_detections(on_tracker_detections):
            tracking_csvs.append(
                write_detection_csv(
                    on_tracker_detections,
                    str(results_dir / "tracking_on_axis_cotracker.csv"),
                    fps,
                    "on_axis_cotracker",
                )
            )
        if _has_any_detections(off_tracker_detections):
            tracking_csvs.append(
                write_detection_csv(
                    off_tracker_detections,
                    str(results_dir / "tracking_off_axis_cotracker.csv"),
                    off_fps,
                    "off_axis_cotracker",
                )
            )
        if _has_any_detections(on_yolo_detections):
            tracking_csvs.append(
                write_detection_csv(
                    on_yolo_detections,
                    str(results_dir / "tracking_on_axis_yolo.csv"),
                    fps,
                    "on_axis_yolo",
                )
            )
        if _has_any_detections(off_yolo_detections):
            tracking_csvs.append(
                write_detection_csv(
                    off_yolo_detections,
                    str(results_dir / "tracking_off_axis_yolo.csv"),
                    off_fps,
                    "off_axis_yolo",
                )
            )
        if _has_any_detections(on_color_detections):
            tracking_csvs.append(
                write_detection_csv(
                    on_color_detections,
                    str(results_dir / "tracking_on_axis_color.csv"),
                    fps,
                    "on_axis_color",
                )
            )
        if _has_any_detections(off_color_detections):
            tracking_csvs.append(
                write_detection_csv(
                    off_color_detections,
                    str(results_dir / "tracking_off_axis_color.csv"),
                    off_fps,
                    "off_axis_color",
                )
            )
    except Exception as exc:
        logger.warning("Failed to write tracking CSVs: %s", exc)
        warnings.append(f"Tracking CSV export failed: {exc}")

    # Stage 3 -- render tracking overlay video
    for stage_key, camera_name, cam_frames, cam_detections, cam_fps, filename in (
        ("render_on_axis", "on_axis", frames, on_detections, fps, "tracking_on_axis.mp4"),
        ("render_off_axis", "off_axis", off_frames, off_detections, off_fps, "tracking_off_axis.mp4"),
    ):
        _progress(
            stage_key,
            0,
            max(min(len(cam_frames), len(cam_detections)), 1),
            f"{camera_name.replace('_', '-')} camera",
        )
        try:
            tracking_videos.append(
                render_tracking_video(
                    cam_frames,
                    cam_detections,
                    str(results_dir / filename),
                    cam_fps,
                    on_progress=lambda current, total, _stage=stage_key, _camera=camera_name: _progress(
                        _stage,
                        current,
                        total,
                        f"{_camera.replace('_', '-')} camera",
                    ),
                )
            )
            _progress(
                stage_key,
                min(len(cam_frames), len(cam_detections)),
                max(min(len(cam_frames), len(cam_detections)), 1),
                f"Rendered {camera_name.replace('_', '-')} tracking video",
            )
        except Exception as exc:
            logger.warning("Failed to render %s tracking video: %s", camera_name, exc)
            warnings.append(f"{camera_name} tracking video render failed: {exc}")

    # Stage 4 -- 3D pose estimation
    _progress("estimate_poses", 0, 1, "Fusing 3D tip positions")
    logger.info("Stage 3: Estimating 3D poses")
    if stereo_calibration and on_calibration and off_calibration:
        logger.info("  Using dual-camera fusion")
        try:
            from app.fusion import StereoFusionError
            poses_3d = estimate_poses_dual(
                on_detections,
                off_detections,
                depth_maps,
                off_depth,
                fps,
                on_calibration=on_calibration,
                off_calibration=off_calibration,
                stereo_calibration=stereo_calibration,
            )
        except StereoFusionError as exc:
            logger.warning("Stereo fusion failed (%s), falling back to single-camera", exc)
            warnings.append(f"Stereo fusion failed: {exc}. Used single-camera estimation instead.")
            poses_3d = estimate_poses(on_detections, depth_maps, fps, calibration=on_calibration)
    else:
        logger.info("  Using single-camera (on-axis only)")
        poses_3d = estimate_poses(on_detections, depth_maps, fps, calibration=on_calibration)
    logger.info("  Estimated %d pose records", len(poses_3d))
    _progress("estimate_poses", 1, 1, f"Calculated {len(poses_3d)} fused pose records")

    try:
        tracking_csvs.append(
            write_pose_csv(
                poses_3d,
                str(results_dir / "tracked_positions_world.csv"),
            )
        )
    except Exception as exc:
        logger.warning("Failed to write world pose CSV: %s", exc)
        warnings.append(f"World pose CSV export failed: {exc}")

    # Stage 5 -- calculate skill metrics.
    _progress("calculate_metrics", 0, 1, "Calculating grading metrics")
    logger.info("Stage 4: Calculating metrics")
    metrics = calculate_metrics(poses_3d, fps)
    logger.info("  Metrics: %s", metrics)
    _progress("calculate_metrics", 1, 1, "Metrics complete")

    result: dict[str, Any] = {"metrics": metrics, "poses": poses_3d}
    if tracking_videos:
        result["tracking_videos"] = tracking_videos
    if tracking_csvs:
        result["tracking_csvs"] = tracking_csvs
    if warnings:
        result["warnings"] = warnings
    return result


# ---------------------------------------------------------------------------
# V2 pipeline: 6-pass offline grading
# ---------------------------------------------------------------------------


def _tracks_to_detections(
    tracks: dict[str, np.ndarray],
    visibility: dict[str, np.ndarray],
    visibility_threshold: float = 0.3,
) -> list[list[Detection]]:
    """Convert PassData track arrays to ``list[list[Detection]]`` for the renderer.

    For each frame, emits a Detection per label where visibility >= threshold
    and position is not NaN.
    """
    if not tracks:
        return []
    n_frames = max(len(t) for t in tracks.values())
    result: list[list[Detection]] = []
    for fidx in range(n_frames):
        frame_dets: list[Detection] = []
        for label, positions in tracks.items():
            if fidx >= len(positions):
                continue
            x, y = float(positions[fidx, 0]), float(positions[fidx, 1])
            if np.isnan(x) or np.isnan(y):
                continue
            vis = visibility.get(label)
            v = float(vis[fidx]) if vis is not None and fidx < len(vis) else 0.0
            if v < visibility_threshold:
                continue
            frame_dets.append(Detection(
                x=x, y=y, confidence=v, label=label, source="v2_pipeline",
            ))
        result.append(frame_dets)
    return result


def run_v2_pipeline(job: dict, on_progress: ProgressCallback = None) -> dict[str, Any]:
    """Run the 6-pass offline grading pipeline.

    Pass 1: SAM2 segmentation → per-instrument binary masks
    Pass 2: CoTracker v3 refinement → 2D tracks + visibility
    Pass 3: Adaptive color gap filling → fill gaps in tracks (CPU)
    Pass 4: Stereo triangulation → 3D positions (CPU)
    Pass 5: RTS smoothing → smoothed 3D trajectories (CPU)
    Pass 6: Identity verification → swap check (CPU)

    GPU models are loaded and unloaded sequentially (SAM2 then CoTracker).
    """
    import os
    import time

    from app.passes.pass_data import PassData
    from app.passes import pass2_cotracker, pass3_color
    from app.passes import pass4_triangulation, pass5_smoothing, pass6_identity

    # Select segmentation backend: "sam2" (default/production) or "sam3" (offline)
    seg_backend = os.environ.get("_SEGMENTATION_BACKEND", "sam2")
    if seg_backend == "sam3":
        from app.passes import pass1_sam3 as pass1_seg
        logger.info("Using SAM3 segmentation backend")
    else:
        from app.passes import pass1_sam2 as pass1_seg
        logger.info("Using SAM2 segmentation backend")

    def _progress(stage: str, current: int, total: int, detail: str = "") -> None:
        if on_progress:
            on_progress(stage, current, total, detail)

    on_axis_path: str = job["on_axis_path"]
    off_axis_path: str = job["off_axis_path"]
    results_dir = _results_dir(job)
    warnings: list[str] = []
    timings: dict[str, float] = {}

    # SVO2 files may not exist when running offline with MP4+NPZ exports.
    # The svo_loader derives session_dir from the path and falls back to
    # exported MP4+NPZ files automatically.
    session_dir_check = Path(on_axis_path).parent
    on_stem = Path(on_axis_path).stem
    off_stem = Path(off_axis_path).stem
    if not Path(on_axis_path).exists() and not (session_dir_check / f"{on_stem}_left.mp4").exists():
        raise FileNotFoundError(f"On-axis source not found (no SVO2 or MP4): {on_axis_path}")
    if not Path(off_axis_path).exists() and not (session_dir_check / f"{off_stem}_left.mp4").exists():
        raise FileNotFoundError(f"Off-axis source not found (no SVO2 or MP4): {off_axis_path}")

    camera_config = job.get("camera_config")

    # Load calibrations
    on_calibration = _load_calibration(job, "calibration_path")
    off_calibration = _load_off_axis_calibration(job)
    stereo_calibration = _load_stereo_calibration(job)

    on_calibration = adjust_calibration(on_calibration, camera_config, "on_axis")
    off_calibration = adjust_calibration(off_calibration, camera_config, "off_axis")

    # Determine labeled frame indices so they are always included in sampling
    session_dir = Path(on_axis_path).parent
    from app.passes.tip_loader import _load_frame_manifest
    _tip_manifest = _load_frame_manifest(session_dir)
    _on_extra: set[int] = set()
    _off_extra: set[int] = set()
    for filename, entry in _tip_manifest.items():
        cam = entry.get("camera", "")
        fidx = entry.get("frame_idx")
        if fidx is None:
            continue
        if "on_axis" in filename or cam == "on_axis":
            _on_extra.add(int(fidx))
        if "off_axis" in filename or cam == "off_axis":
            _off_extra.add(int(fidx))
    if _on_extra or _off_extra:
        logger.info(
            "Injecting labeled frames: on_axis=%s off_axis=%s",
            sorted(_on_extra), sorted(_off_extra),
        )

    # Load frames
    _progress("load_frames", 0, 2, "Loading on-axis frames")
    logger.info("V2 Pipeline: Loading frames")
    frames, depth_maps, fps, on_frame_indices = load_svo2(
        on_axis_path,
        on_progress=lambda c, t: _progress("load_frames", c, t, "Loading on-axis"),
        camera_config=camera_config,
        extra_frames=_on_extra or None,
    )
    _progress("load_frames", 1, 2, "Loading off-axis frames")
    off_frames, off_depth, off_fps, off_frame_indices = load_svo2(
        off_axis_path,
        on_progress=lambda c, t: _progress("load_frames", c, t, "Loading off-axis"),
        camera_config=camera_config,
        extra_frames=_off_extra or None,
    )
    logger.info("Loaded %d on-axis, %d off-axis frames at %.1f fps", len(frames), len(off_frames), fps)

    # Create PassData
    data = PassData(
        session_dir=session_dir,
        on_frames=frames,
        off_frames=off_frames,
        fps=fps,
        on_depth=depth_maps,
        off_depth=off_depth,
        stereo_calib=stereo_calibration,
        on_calib=on_calibration,
        off_calib=off_calibration,
        on_frame_indices=on_frame_indices,
        off_frame_indices=off_frame_indices,
    )

    # Pass 1: Segmentation (SAM2 or SAM3)
    pass1_key = f"pass1_{seg_backend}"
    t0 = time.monotonic()
    try:
        used_fallback = pass1_seg.run(data, on_progress=on_progress)
        if used_fallback:
            warnings.append(
                f"{seg_backend.upper()} used auto-detected tip positions (tip_init.json not found). "
                "For best results, confirm tip positions via the Initialize Tips page."
            )
    except Exception as exc:
        logger.warning("Pass 1 (%s) failed: %s", seg_backend.upper(), exc, exc_info=True)
        warnings.append(f"Pass 1 ({seg_backend.upper()}) failed: {exc}")
    timings[pass1_key] = time.monotonic() - t0
    logger.info("Pass 1 timing: %.1fs", timings[pass1_key])

    # Debug: render segmentation overlay videos (with tip init markers)
    if os.environ.get("_DEBUG_RENDER"):
        try:
            from app.debug_renderer import render_segmentation_video
            from app.passes.tip_loader import load_tip_points as _load_debug_tips

            _progress("debug_render", 0, 2, "Rendering segmentation debug videos")
            _si = int(os.environ.get("FRAME_SAMPLE_INTERVAL", "5"))

            if data.on_masks and frames:
                _on_tips, _ = _load_debug_tips(session_dir, "on_axis", _si, n_frames=len(frames))
                render_segmentation_video(
                    frames, data.on_masks,
                    str(results_dir / "debug_segmentation_on_axis.mp4"), fps,
                    camera_name="on_axis",
                    tip_points=_on_tips or None,
                )
            if data.off_masks and off_frames:
                _off_tips, _ = _load_debug_tips(session_dir, "off_axis", _si, n_frames=len(off_frames))
                render_segmentation_video(
                    off_frames, data.off_masks,
                    str(results_dir / "debug_segmentation_off_axis.mp4"), fps,
                    camera_name="off_axis",
                    tip_points=_off_tips or None,
                )
            _progress("debug_render", 2, 2, "Segmentation debug videos complete")
        except Exception as exc:
            logger.warning("Debug segmentation render failed: %s", exc, exc_info=True)

    # Pass 2: CoTracker point refinement
    t0 = time.monotonic()
    try:
        pass2_cotracker.run(data, on_progress=on_progress)
    except Exception as exc:
        logger.warning("Pass 2 (CoTracker) failed: %s", exc, exc_info=True)
        warnings.append(f"Pass 2 (CoTracker) failed: {exc}")
    timings["pass2_cotracker"] = time.monotonic() - t0
    logger.info("Pass 2 timing: %.1fs", timings["pass2_cotracker"])

    # Debug: render CoTracker overlay videos
    if os.environ.get("_DEBUG_RENDER"):
        try:
            from app.debug_renderer import render_cotracker_video
            _progress("debug_render", 0, 2, "Rendering CoTracker debug videos")
            if data.on_tracks and frames:
                render_cotracker_video(
                    frames, data.on_tracks, data.on_visibility, data.on_masks,
                    str(results_dir / "debug_cotracker_on_axis.mp4"), fps,
                    camera_name="on_axis",
                )
            if data.off_tracks and off_frames:
                render_cotracker_video(
                    off_frames, data.off_tracks, data.off_visibility, data.off_masks,
                    str(results_dir / "debug_cotracker_off_axis.mp4"), fps,
                    camera_name="off_axis",
                )
            _progress("debug_render", 2, 2, "CoTracker debug videos complete")
        except Exception as exc:
            logger.warning("Debug CoTracker render failed: %s", exc, exc_info=True)

    # Pass 3: Adaptive color gap filling (CPU)
    t0 = time.monotonic()
    try:
        pass3_color.run(data, on_progress=on_progress)
    except Exception as exc:
        logger.warning("Pass 3 (Color) failed: %s", exc, exc_info=True)
        warnings.append(f"Pass 3 (Color) failed: {exc}")
    timings["pass3_color"] = time.monotonic() - t0
    logger.info("Pass 3 timing: %.1fs", timings["pass3_color"])

    # Pass 4: Stereo triangulation (CPU)
    t0 = time.monotonic()
    try:
        pass4_triangulation.run(data, on_progress=on_progress)
    except Exception as exc:
        logger.warning("Pass 4 (Triangulation) failed: %s", exc, exc_info=True)
        warnings.append(f"Pass 4 (Triangulation) failed: {exc}")
    timings["pass4_triangulation"] = time.monotonic() - t0
    logger.info("Pass 4 timing: %.1fs", timings["pass4_triangulation"])

    # Pass 5: RTS smoothing (CPU)
    t0 = time.monotonic()
    try:
        pass5_smoothing.run(data, on_progress=on_progress)
    except Exception as exc:
        logger.warning("Pass 5 (Smoothing) failed: %s", exc, exc_info=True)
        warnings.append(f"Pass 5 (Smoothing) failed: {exc}")
    timings["pass5_smoothing"] = time.monotonic() - t0
    logger.info("Pass 5 timing: %.1fs", timings["pass5_smoothing"])

    # Pass 6: Identity verification (CPU)
    t0 = time.monotonic()
    try:
        pass6_identity.run(data, on_progress=on_progress)
    except Exception as exc:
        logger.warning("Pass 6 (Identity) failed: %s", exc, exc_info=True)
        warnings.append(f"Pass 6 (Identity) failed: {exc}")
    timings["pass6_identity"] = time.monotonic() - t0
    logger.info("Pass 6 timing: %.1fs", timings["pass6_identity"])

    # Render tracking overlay videos and CSVs
    t0 = time.monotonic()
    _progress("render_tracking", 0, 4, "Rendering tracking overlays")
    tracking_videos: list[str] = []
    tracking_csvs: list[str] = []
    try:
        on_detections = _tracks_to_detections(data.on_tracks, data.on_visibility)
        off_detections = _tracks_to_detections(data.off_tracks, data.off_visibility)

        if on_detections and frames:
            _progress("render_tracking", 0, 4, "Rendering on-axis tracking video")
            on_video_path = render_tracking_video(
                frames, on_detections,
                str(results_dir / "tracking_on_axis.mp4"), fps,
            )
            tracking_videos.append(on_video_path)

        _progress("render_tracking", 1, 4, "Rendering off-axis tracking video")
        if off_detections and off_frames:
            off_video_path = render_tracking_video(
                off_frames, off_detections,
                str(results_dir / "tracking_off_axis.mp4"), fps,
            )
            tracking_videos.append(off_video_path)

        _progress("render_tracking", 2, 4, "Writing detection CSVs")
        if on_detections:
            tracking_csvs.append(write_detection_csv(
                on_detections, str(results_dir / "detections_on_axis.csv"), fps, "on_axis",
            ))
        if off_detections:
            tracking_csvs.append(write_detection_csv(
                off_detections, str(results_dir / "detections_off_axis.csv"), fps, "off_axis",
            ))
    except Exception as exc:
        logger.warning("Tracking render failed: %s", exc, exc_info=True)
        warnings.append(f"Tracking render failed: {exc}")
    timings["render_tracking"] = time.monotonic() - t0
    logger.info("Render tracking timing: %.1fs", timings["render_tracking"])
    _progress("render_tracking", 4, 4, "Tracking render complete")

    # Convert PassData to poses format for metrics calculation
    # Use smoothed_3d if available, fall back to trajectories_3d
    traj_source = data.smoothed_3d if data.smoothed_3d else data.trajectories_3d
    labels = sorted(traj_source.keys())
    n_frames = max((len(t) for t in traj_source.values()), default=0)

    poses_3d: list[dict[str, Any]] = []
    for fidx in range(n_frames):
        timestamp = fidx / fps
        pose: dict[str, Any] = {
            "frame_idx": fidx,
            "timestamp": round(timestamp, 6),
        }
        for label in labels:
            traj = traj_source[label]
            if fidx < len(traj) and not np.any(np.isnan(traj[fidx])):
                pose[label] = [round(float(v), 6) for v in traj[fidx]]
            else:
                pose[label] = None
        poses_3d.append(pose)

    # Calculate metrics
    _progress("calculate_metrics", 0, 1, "Calculating grading metrics")
    metrics = calculate_metrics(poses_3d, fps)
    logger.info("V2 Metrics: %s", metrics)

    # Write pose CSV
    try:
        write_pose_csv(poses_3d, str(results_dir / "tracked_positions_world.csv"))
    except Exception as exc:
        logger.warning("Failed to write pose CSV: %s", exc, exc_info=True)
    _progress("calculate_metrics", 1, 1, "Metrics complete")

    # Build result
    result: dict[str, Any] = {
        "metrics": metrics,
        "poses": poses_3d,
        "pipeline_mode": "v2",
        "timings": timings,
        "tracking_videos": tracking_videos,
        "tracking_csvs": tracking_csvs,
    }
    if data.swap_map:
        warnings.append(f"Identity swap applied: {data.swap_map}")
    if warnings:
        result["warnings"] = warnings

    total_time = sum(timings.values())
    logger.info("V2 Pipeline complete in %.1fs — %s", total_time, timings)
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def grade(job: dict, on_progress: ProgressCallback = None) -> dict[str, Any]:
    """Run the grading pipeline, dispatching based on PIPELINE_MODE config.

    Parameters
    ----------
    job : dict
        Must contain ``session_id``, ``on_axis_path`` and ``off_axis_path``.
    on_progress : callable, optional
        ``(stage, current, total, detail)`` callback.

    Returns
    -------
    dict
        ``{"metrics": {...}, "poses": [...], ...}``
    """
    mode = PIPELINE_MODE
    logger.info("Grading pipeline mode: %s", mode)

    if mode == "v2":
        return run_v2_pipeline(job, on_progress=on_progress)
    else:
        return run_pipeline(job, on_progress=on_progress)
