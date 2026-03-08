# ML Backends

The grading pipeline uses a pluggable backend system for instrument tip detection. Multiple ML models are supported, and users can switch between them or upload custom models via the web UI.

## Backend Interface

All backends implement the `ModelBackend` abstract class defined in `services/grader/app/backends/base.py`:

```python
@dataclass(frozen=True)
class Detection:
    x: float              # pixel x coordinate
    y: float              # pixel y coordinate
    confidence: float     # [0.0, 1.0]
    label: str            # "left_tip", "right_tip", etc.

class ModelBackend(ABC):
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""

    @abstractmethod
    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of BGR images (H, W, 3) uint8
            query_points: Optional (N, 3) array of [frame_idx, x, y] for point tracking
            on_progress: Callback(current, total) for progress reporting

        Returns:
            List of detection lists, one per frame
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
```

## Available Backends

### YOLO (`yolo_backend.py`)

**Type:** Object detection
**Model:** Ultralytics YOLOv8/v11
**File:** `.pt` checkpoint

**How it works:**
1. Runs YOLO inference on each frame independently
2. Extracts bounding box centers as tip positions
3. Maps detections to `left_tip` / `right_tip` based on class names or spatial position

**Tip assignment logic:**
- If model has classes with "tip" in the name: uses those directly
- Otherwise: takes top-2 highest-confidence detections, assigns left/right by x-coordinate (leftmost = left_tip)

**Best for:** Custom YOLO models trained on surgical instrument tips. Fast inference.

**Catalog model:** `yolov8n` (6.2 MB) -- general-purpose nano model, not specifically trained for surgical instruments.

### CoTracker (`cotracker_backend.py`)

**Type:** Point tracking
**Model:** Meta Co-Tracker v2
**Requires:** Query points (initial tip positions from tip_init.json)

**How it works:**
1. Stacks all frames into a video tensor (1, T, 3, H, W)
2. Tracks query points through time using the Co-Tracker model
3. Returns tracked positions with visibility scores
4. Filters by visibility > 0.5

**Label assignment:** Even query indices = `left_tip`, odd = `right_tip`

**Best for:** Tracking known initial positions through a video. Handles occlusion well.

**Catalog model:** `cotracker-v2` (120 MB)

### TAPIR (`tapir_backend.py`)

**Type:** Point tracking (DeepMind)
**Model:** TAPIR (Tracking Any Point with per-frame Initialization and temporal Refinement)
**Requires:** Query points (optional -- defaults to center-left and center-right)

**How it works:**
1. Normalizes frames to [0, 1], stacks to video tensor
2. Runs TAPIR model with query points
3. Returns tracks with optional occlusion scores
4. Filters by occlusion < 0.5

**Default query points** (when none provided):
- Left tip: (frame 0, 35% width, 50% height)
- Right tip: (frame 0, 65% width, 50% height)

**Best for:** Long-range point tracking with temporal refinement.

**Catalog model:** `tapir-v1` (280 MB)

### SAM2 (`sam2_backend.py`)

**Type:** Segmentation
**Model:** SAM 2 (Segment Anything Model 2)

**How it works:**
1. Runs segmentation on each frame
2. Thresholds mask at 0.5
3. Extracts tip positions from mask extrema (leftmost point = left_tip, rightmost = right_tip)
4. Fixed confidence: 0.85

**Best for:** When instruments are visually distinct from background but no point tracking is needed.

**Catalog models:** `sam2-hiera-large` (898 MB), `sam2-hiera-small` (185 MB)

### Placeholder (`placeholder_backend.py`)

**Type:** Synthetic (no ML)

**How it works:** Generates deterministic pseudo-random detections based on frame index (seeded with frame_idx). Used automatically when no real model is available.

**When used:**
- No active model in the database
- Backend fails to load (automatic fallback)
- Dev mode (no PyTorch available)

**Detection positions:**
- Left tip: x in [20-50% width], y in [20-80% height]
- Right tip: x in [50-80% width], y in [20-80% height]
- Confidence: uniform [0.85, 0.99]

## Model Loading and Caching

The `model_loader.py` module manages backend lifecycle:

1. On each grading job, calls `get_backend()` which checks the database for the active model
2. If the active model matches the cached backend, returns the cached instance (no reload)
3. If a different model is now active, unloads the old backend and loads the new one
4. If no model is active, returns `PlaceholderBackend`
5. If loading fails, logs the error and falls back to `PlaceholderBackend`

**GPU memory management:** All backends call `del self._model` and `torch.cuda.empty_cache()` in their `unload()` method to free GPU memory.

## Model Catalog

Pre-configured models available for download via the UI:

| Slug | Name | Type | Size | Description |
|------|------|------|------|-------------|
| `cotracker-v2` | Co-Tracker v2 | point_tracking | 120 MB | Zero-shot point tracking (Meta) |
| `sam2-hiera-large` | SAM 2 Hiera Large | segmentation | 898 MB | Video segmentation, large variant |
| `sam2-hiera-small` | SAM 2 Hiera Small | segmentation | 185 MB | Video segmentation, small variant |
| `tapir-v1` | TAPIR v1 | point_tracking | 280 MB | DeepMind point tracking |
| `yolov8n` | YOLOv8 Nano | detection | 6.2 MB | Baseline object detection |

Models are downloaded to `{MODELS_DIR}/{model_type}/{slug}/` and activated via the Models page.

## Custom Models

Users can upload custom `.pt` model files via the Models page:

1. Click "Upload Model" and select a `.pt` file (max 500 MB)
2. The model is saved as `{MODELS_DIR}/custom/custom-{filename}/`
3. Custom models get slug `custom-{filename}` and type based on the YOLO backend
4. Activate the model to use it for grading

## Adding a New Backend

To add support for a new ML model type:

### 1. Create the Backend Class

Create `services/grader/app/backends/my_backend.py`:

```python
import numpy as np
from .base import ModelBackend, Detection

class MyBackend(ModelBackend):
    def __init__(self):
        self._model = None

    def load(self, path: str) -> None:
        # Load your model from path
        import my_framework
        self._model = my_framework.load(path)

    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        on_progress=None,
    ) -> list[list[Detection]]:
        results = []
        for i, frame in enumerate(frames):
            # Run inference
            detections = self._run_inference(frame)
            results.append(detections)
            if on_progress:
                on_progress(i + 1, len(frames))
        return results

    def _run_inference(self, frame: np.ndarray) -> list[Detection]:
        # Your detection logic here
        # Must return list of Detection(x, y, confidence, label)
        pass

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
```

### 2. Register in Model Loader

Edit `services/grader/app/model_loader.py`:

```python
# Add to the backend type mapping
if model_type == "my_type":
    from app.backends.my_backend import MyBackend
    backend = MyBackend()
```

### 3. Add to Model Catalog (Optional)

If you want the model to appear in the download catalog, edit `services/api/app/model_registry.py`:

```python
MODEL_CATALOG.append({
    "slug": "my-model-v1",
    "name": "My Model v1",
    "model_type": "my_type",
    "description": "Description of what it does",
    "version": "1.0.0",
    "download_url": "https://example.com/my-model.pt",
    "file_size_bytes": 50_000_000,
})
```

### 4. Update Dependencies

Add any new Python packages to `services/grader/requirements.txt` and rebuild:

```bash
docker compose build grader
docker compose up -d grader exporter
```

## Pipeline Integration

The detection pipeline in `pipeline.py` uses backends as follows:

```
Load frames (svo_loader)
  -> Load query points from tip_init.json (if point tracking backend)
  -> backend.detect(frames, query_points)
  -> 2D detections per frame
  -> Render tracking overlay video
  -> Back-project to 3D using depth + calibration
  -> If dual camera: fuse with stereo triangulation
  -> Calculate surgical metrics
```

Query points are sourced from `tip_init.json` in the session directory. Point tracking backends (CoTracker, TAPIR) use these to initialize tracking. Detection backends (YOLO) and segmentation backends (SAM2) ignore query points and detect tips independently per frame.
