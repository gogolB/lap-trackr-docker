# ML Backends

Lap-Trackr grading is no longer described as a single pluggable detector running frame-by-frame. The target design is an offline multi-pass grading pipeline that uses different model families for different jobs.

The primary grading stack is:

- **SAM2** for dense per-view video segmentation
- **CoTracker3** for tip-level point tracking and occlusion-aware refinement
- **Color analysis** for gap filling and identity verification

Optional support models can still be useful:

- **YOLO11 pose** for initialization hints, re-acquisition after long losses, and QA checks

See [Offline Grading Pipeline](offline-grading-pipeline.md) for the full design.

## Model Roles

### SAM2

**Role:** Primary per-view segmentation model

SAM2 should produce dense green and pink tool masks for each exported camera stream. In the offline pipeline it is expected to use bidirectional video context, not real-time single-frame prompting.

SAM2 is responsible for:

- tool-region segmentation
- long-range spatial consistency
- defining mask support for identity checks
- supplying gap intervals when the tool is fully occluded or leaves frame

### CoTracker3

**Role:** Primary tip tracker

CoTracker3 consumes user-confirmed tip initialization points and tracks the exact tip location through time. It is the preferred point-level tracker because it provides sub-pixel tracks and visibility signals.

CoTracker3 is responsible for:

- precise tip trajectories
- handling partial occlusion
- preserving temporal continuity
- validating that the tracked tip remains inside the SAM2 mask support

### Color Analysis

**Role:** Conservative offline fallback and identity check

Color analysis should use the green and pink tape as a strong identity cue. It is most useful when both SAM2 and CoTracker3 lose confidence or when identity must be verified through ambiguous crossings.

Color is responsible for:

- filling short detection gaps
- reinforcing green / pink identity
- catching obvious identity swaps
- supporting low-confidence review flags

### YOLO11 Pose

**Role:** Optional auxiliary detector

YOLO11 pose is not the primary grader. Its best use is as a support model for:

- tip proposals on difficult frames
- re-acquisition after long losses
- QA against the SAM2 + CoTracker3 stack

The current built-in auxiliary model is the large laparoscopic pose checkpoint in `services/api/app/model_registry.py`.

## Runtime Contract

Backends still implement the shared `ModelBackend` interface in `services/grader/app/backends/base.py`.

```python
@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    confidence: float
    label: str         # "green_tip" or "pink_tip"
    source: str        # "cotracker", "yolo", "color", etc.

class ModelBackend(ABC):
    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def detect(
        self,
        frames: list[np.ndarray],
        query_points: np.ndarray | None = None,
        query_labels: list[str] | tuple[str, ...] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[Detection]]: ...

    @abstractmethod
    def unload(self) -> None: ...
```

Important conventions:

- labels should normalize to `green_tip` and `pink_tip`
- query points are `[frame_idx, x, y]`
- `source` must indicate which pass or backend produced the detection
- offline graders should preserve provenance for later confidence weighting

## Active / Planned Model Catalog

Current built-in models:

| Slug | Name | Type | Role |
|------|------|------|------|
| `cotracker-v3-offline` | CoTracker 3 Offline | `cotracker` | Primary tip tracker |
| `yolo11l-pose-1088` | YOLO11L Pose 1088 | `yolo` | Auxiliary pose detector |

Target primary segmentation family:

| Family | Intended Role |
|--------|---------------|
| `sam2` | Primary dense video segmentation for offline grading |

Models are stored under `{MODELS_DIR}` and can be activated by type. The offline grading design assumes SAM2 and CoTracker3 can both participate in the same grading run.

## Initialization Contract

Initialization points come from the exported sample-frame workflow:

- the exporter writes representative sample JPGs
- the user confirms green and pink tip locations in `tip_init.json`
- the exporter also writes `tip_init_samples.json` so the grader can recover the original source frame indices

Those frame indices matter. Offline trackers must start from the correct sampled frame, not assume all prompts belong to frame `0`.

## Pipeline Integration

At the system level, backends are not interchangeable replacements for one another. They serve different passes:

1. **SAM2** segments the taped instruments per view
2. **CoTracker3** refines tip points from confirmed prompts
3. **Color analysis** fills or verifies low-confidence gaps
4. **Multi-view triangulation** reconstructs 3D positions
5. **Smoothing / optimization** produces the final trajectories

This is intentionally different from a real-time detector-first design.

## Adding a New Backend

New models should only be added if they have a clear role in the offline pipeline. Before adding one, decide whether it is meant for:

- segmentation
- point tracking
- auxiliary detection / re-acquisition
- identity verification

Then:

1. Create the backend class in `services/grader/app/backends/`
2. Register it in `services/grader/app/model_loader.py`
3. Add it to `services/api/app/model_registry.py` if it should appear in the managed catalog
4. Document exactly which pass it serves in the offline pipeline

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
