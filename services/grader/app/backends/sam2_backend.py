"""SAM2 video segmentation backend.

Follows the ModelBackend interface for integration with the model loader.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from app.backends.base import Detection, ModelBackend

logger = logging.getLogger("grader.backends.sam2")


class SAM2Backend(ModelBackend):
    """SAM2 video segmentation via Meta's SAM 2.1 model."""

    def __init__(self) -> None:
        self._predictor: Any = None

    def load(self, path: str) -> None:
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor

            from app.config import SAM2_CONFIG_PATH

            logger.info("Loading SAM2 model from %s", path)
            self._predictor = build_sam2_video_predictor(
                config_file=SAM2_CONFIG_PATH,
                ckpt_path=path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                offload_video_model_to_cpu=True,
                offload_state_to_cpu=True,
            )
            logger.info("SAM2 model loaded")
        except Exception as exc:
            logger.error("Failed to load SAM2 model: %s", exc)
            raise

    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        query_labels: list[str] | tuple[str, ...] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]:
        """Run SAM2 segmentation and return centroid detections.

        This adapts the segmentation model to the Detection interface by
        returning mask centroids as detections.
        """
        if not frames or self._predictor is None:
            return [[] for _ in frames]

        if query_points is None or len(query_points) == 0:
            logger.warning("SAM2 requires query points for initialization")
            return [[] for _ in frames]

        try:
            import torch

            label_map = {0: "green_tip", 1: "pink_tip"}
            if query_labels:
                label_map = {i: str(query_labels[i]) for i in range(len(query_labels))}

            with torch.inference_mode():
                inference_state = self._predictor.init_state(video=frames)

                # Add points for each instrument
                for i in range(len(query_points)):
                    obj_id = i + 1
                    pts = np.array([[query_points[i, 1], query_points[i, 2]]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    self._predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=int(query_points[i, 0]),
                        obj_id=obj_id,
                        points=pts,
                        labels=labels,
                    )

                # Forward propagation
                results: dict[int, dict[int, np.ndarray]] = {}
                for frame_idx, obj_ids, masks in self._predictor.propagate_in_video(
                    inference_state=inference_state,
                ):
                    frame_masks = {}
                    for j, oid in enumerate(obj_ids):
                        mask = (masks[j, 0] > 0.0).cpu().numpy().astype(np.uint8)
                        frame_masks[oid] = mask
                    results[frame_idx] = frame_masks
                    if on_progress and (frame_idx + 1) % 10 == 0:
                        on_progress(frame_idx + 1, len(frames))

                self._predictor.reset_state(inference_state)

            # Convert masks to centroid detections
            all_detections: list[list[Detection]] = []
            for fidx in range(len(frames)):
                frame_dets: list[Detection] = []
                frame_masks = results.get(fidx, {})
                for i in range(len(query_points)):
                    obj_id = i + 1
                    mask = frame_masks.get(obj_id)
                    if mask is None:
                        continue
                    ys, xs = np.where(mask > 0)
                    if len(xs) == 0:
                        continue
                    cx = float(np.mean(xs))
                    cy = float(np.mean(ys))
                    area_ratio = float(len(xs)) / max(1, mask.shape[0] * mask.shape[1])
                    confidence = min(1.0, area_ratio * 100)  # Rough confidence
                    label = label_map.get(i, f"instrument_{i + 1}_tip")
                    frame_dets.append(Detection(
                        x=cx, y=cy,
                        confidence=confidence,
                        label=label,
                        source="sam2",
                    ))
                all_detections.append(frame_dets)
                if on_progress and (fidx + 1) == len(frames):
                    on_progress(len(frames), len(frames))

            return all_detections

        except Exception as exc:
            logger.warning("SAM2 inference failed: %s", exc)
            return [[] for _ in frames]

    def unload(self) -> None:
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("SAM2 model unloaded")
