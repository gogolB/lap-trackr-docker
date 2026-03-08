"""Static catalog of downloadable ML models.

Each entry defines a model that can be downloaded to the NVMe drive and
activated as the instrument-detection backend in the grading pipeline.
"""

from __future__ import annotations

MODEL_CATALOG: list[dict] = [
    {
        "slug": "yolo11l-pose-1088",
        "name": "YOLO11L Pose 1088",
        "model_type": "yolo",
        "description": "Primary laparoscopic pose model for tip detection and reacquisition support.",
        "version": "11l-1088",
        "local_path": "yolov11-pose/yolo11l_pose_1088.pt",
        "auto_activate": True,
    },
    {
        "slug": "cotracker-v3-offline",
        "name": "CoTracker 3 Offline",
        "model_type": "cotracker",
        "description": "Offline point tracker for exported grading videos. Use it with green and pink tip initialization.",
        "version": "3.0",
        "local_path": "cotracker/cotracker-v3-offline/scaled_offline.pth",
        "auto_activate": True,
        "download_url": "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth",
    },
]
