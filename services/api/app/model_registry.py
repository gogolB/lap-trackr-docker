"""Static catalog of downloadable ML models.

Each entry defines a model that can be downloaded to the NVMe drive and
activated as the instrument-detection backend in the grading pipeline.
"""

from __future__ import annotations

MODEL_CATALOG: list[dict] = [
    {
        "slug": "cotracker-v2",
        "name": "Co-Tracker v2",
        "model_type": "cotracker",
        "description": "Zero-shot point tracking for instrument tips across video frames.",
        "version": "2.0",
        "download_url": "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth",
        "file_size_bytes": 120_000_000,
    },
    {
        "slug": "sam2-hiera-large",
        "name": "SAM 2 Hiera Large",
        "model_type": "sam2",
        "description": "Video segmentation model for instrument mask extraction — large variant.",
        "version": "2.0",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "file_size_bytes": 898_000_000,
    },
    {
        "slug": "sam2-hiera-small",
        "name": "SAM 2 Hiera Small",
        "model_type": "sam2",
        "description": "Lightweight SAM 2 variant for faster segmentation.",
        "version": "2.0",
        "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "file_size_bytes": 185_000_000,
    },
    {
        "slug": "tapir-v1",
        "name": "TAPIR v1",
        "model_type": "tapir",
        "description": "DeepMind point tracking model for long-range instrument tip tracking.",
        "version": "1.0",
        "download_url": "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt",
        "file_size_bytes": 280_000_000,
    },
    {
        "slug": "yolov8n",
        "name": "YOLOv8 Nano",
        "model_type": "yolo",
        "description": "Ultralytics YOLOv8 nano baseline — fast object detection. Replace with a custom surgical model.",
        "version": "8.0",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
        "file_size_bytes": 6_200_000,
    },
]
