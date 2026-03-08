"""Configuration loaded from environment variables."""

import os


REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# The API service may use an asyncpg:// scheme; the sync worker needs
# psycopg2, so we normalise the URL at import time.
_raw_db_url: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/laptrackr",
)
DATABASE_URL: str = (
    _raw_db_url
    .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    .replace("postgres://", "postgresql+psycopg2://")
)

DATA_DIR: str = os.environ.get("DATA_DIR", "/data")
MODELS_DIR: str = os.environ.get("MODELS_DIR", "/data/models")

# SVO2 loader settings
FRAME_SAMPLE_INTERVAL: int = int(os.environ.get("FRAME_SAMPLE_INTERVAL", "5"))

# Placeholder camera intrinsics (ZED 2i typical values at 720p)
CAMERA_FX: float = float(os.environ.get("CAMERA_FX", "700.0"))
CAMERA_FY: float = float(os.environ.get("CAMERA_FY", "700.0"))
CAMERA_CX: float = float(os.environ.get("CAMERA_CX", "640.0"))
CAMERA_CY: float = float(os.environ.get("CAMERA_CY", "360.0"))

# Target FPS assumed for timing calculations when SVO2 metadata is
# unavailable (ZED cameras typically record at 30 fps).
DEFAULT_FPS: float = float(os.environ.get("DEFAULT_FPS", "30.0"))

# Pipeline mode: "v2" (6-pass offline) or "legacy" (original multi-pass)
PIPELINE_MODE: str = os.environ.get("PIPELINE_MODE", "v2")

# SAM2 model paths
SAM2_MODEL_PATH: str = os.environ.get(
    "SAM2_MODEL_PATH", "/data/models/sam2/sam2.1_hiera_large.pt"
)
SAM2_CONFIG_PATH: str = os.environ.get(
    "SAM2_CONFIG_PATH", "configs/sam2.1/sam2.1_hiera_l.yaml"
)

# CoTracker v3 chunked processing settings
COTRACKER_CHUNK_SIZE: int = int(os.environ.get("COTRACKER_CHUNK_SIZE", "300"))
COTRACKER_OVERLAP: int = int(os.environ.get("COTRACKER_OVERLAP", "30"))
