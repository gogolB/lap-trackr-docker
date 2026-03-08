# Data Model

This document describes the durable schema, ephemeral job-state keys, and on-disk artifacts produced for each recorded session.

For migration mechanics and operational guidance, see [Database & Migrations](database.md).

## PostgreSQL Schema

### `users`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default `uuid4()` | User identifier |
| `username` | VARCHAR(255) | UNIQUE, indexed, NOT NULL | Login name |
| `hashed_password` | VARCHAR(255) | NOT NULL | Password hash |
| `created_at` | TIMESTAMPTZ | NOT NULL, default now | Account creation time |

Deleting a user cascades to their sessions.

### `sessions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default `uuid4()` | Session identifier |
| `user_id` | UUID | FK -> `users.id`, NOT NULL | Owning user |
| `name` | VARCHAR(255) | NOT NULL | Human-readable session name |
| `started_at` | TIMESTAMPTZ | NOT NULL | Recording start time |
| `stopped_at` | TIMESTAMPTZ | NULLABLE | Recording stop time |
| `status` | `sessionstatus` enum | NOT NULL | Durable session lifecycle state |
| `on_axis_path` | TEXT | NULLABLE | Absolute path to the on-axis `.svo2` file |
| `off_axis_path` | TEXT | NULLABLE | Absolute path to the off-axis `.svo2` file |
| `created_at` | TIMESTAMPTZ | NOT NULL, default now | Row creation time |

`sessionstatus` values:

- `recording`
- `completed`
- `exporting`
- `export_failed`
- `awaiting_init`
- `grading`
- `graded`
- `failed`

Deleting a session cascades to `grading_results` and any session-scoped `calibrations`.

### `grading_results`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default `uuid4()` | Result identifier |
| `session_id` | UUID | FK -> `sessions.id`, UNIQUE, NOT NULL | One grading result per session |
| `workspace_volume` | FLOAT | NULLABLE | Convex-hull volume of fused tip positions |
| `avg_speed` | FLOAT | NULLABLE | Mean tip speed |
| `max_jerk` | FLOAT | NULLABLE | Peak jerk |
| `path_length` | FLOAT | NULLABLE | Total tip travel distance |
| `economy_of_motion` | FLOAT | NULLABLE | Direct distance / total path |
| `total_time` | FLOAT | NULLABLE | Graded duration in seconds |
| `completed_at` | TIMESTAMPTZ | NULLABLE | When grading finished |
| `error` | TEXT | NULLABLE | Fatal grading error |
| `warnings` | JSONB | NULLABLE | Non-fatal pipeline warnings |

### `calibrations`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default `uuid4()` | Calibration identifier |
| `camera_name` | VARCHAR(32) | NOT NULL | `on_axis` or `off_axis` |
| `is_default` | BOOLEAN | NOT NULL | Whether this is the default calibration for that camera |
| `session_id` | UUID | FK -> `sessions.id`, NULLABLE | Session-scoped calibration, if any |
| `fx`, `fy`, `cx`, `cy` | FLOAT | NOT NULL | Camera intrinsics |
| `k1`, `k2`, `k3`, `p1`, `p2` | FLOAT | NULLABLE | Distortion coefficients |
| `image_width`, `image_height` | INTEGER | NOT NULL | Calibration image size |
| `extrinsic_matrix` | JSONB | NULLABLE | 4x4 camera-to-board transform |
| `board_rows`, `board_cols` | INTEGER | NOT NULL | ChArUco board dimensions |
| `square_size_mm`, `marker_size_mm` | FLOAT | NOT NULL | Board geometry |
| `aruco_dict` | VARCHAR(32) | NOT NULL | Dictionary name, for example `DICT_5X5_100` |
| `reprojection_error` | FLOAT | NULLABLE | RMS reprojection error in pixels |
| `num_frames_used` | INTEGER | NULLABLE | Number of calibration captures used |
| `is_global` | BOOLEAN | NOT NULL, default false | Global calibration flag |
| `calibration_path` | TEXT | NULLABLE | Path to the saved JSON file |
| `created_at` | TIMESTAMPTZ | NOT NULL, default now | Creation time |

Constraints:

- partial unique index: only one row per `camera_name` may have `is_default = true`

### `ml_models`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default `uuid4()` | Model identifier |
| `slug` | VARCHAR(64) | UNIQUE, NOT NULL | Stable model key |
| `name` | VARCHAR(255) | NOT NULL | Display name |
| `model_type` | VARCHAR(32) | NOT NULL | `detection`, `point_tracking`, or `segmentation` |
| `description` | TEXT | NULLABLE | Human-readable description |
| `version` | VARCHAR(64) | NULLABLE | Version string |
| `download_url` | TEXT | NULLABLE | Source URL for downloadable models |
| `file_size_bytes` | BIGINT | NULLABLE | Expected download size |
| `file_path` | TEXT | NULLABLE | Local model weights path |
| `status` | `modelstatus` enum | NOT NULL | Lifecycle state |
| `is_active` | BOOLEAN | NOT NULL, default false | Active model used by the grader |
| `is_custom` | BOOLEAN | NOT NULL, default false | User-uploaded model flag |
| `created_at` | TIMESTAMPTZ | NOT NULL, default now | Creation time |
| `updated_at` | TIMESTAMPTZ | NOT NULL, updated automatically | Last mutation time |

`modelstatus` values:

- `available`
- `downloading`
- `ready`
- `active`
- `custom`
- `failed`

### `camera_config`

Single-row table. The enforced row id is always `1`.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PK, check `id = 1` | Singleton row key |
| `on_axis_serial` | VARCHAR(32) | NOT NULL | On-axis ZED serial number |
| `off_axis_serial` | VARCHAR(32) | NOT NULL | Off-axis ZED serial number |
| `on_axis_swap_eyes`, `off_axis_swap_eyes` | BOOLEAN | NOT NULL, default false | Use right-eye intrinsics/video instead of left |
| `on_axis_rotation`, `off_axis_rotation` | INTEGER | NOT NULL, default 0 | Camera rotation in degrees (`0`, `90`, `180`, `270`) |
| `on_axis_flip_h`, `on_axis_flip_v` | BOOLEAN | NOT NULL, default false | Horizontal/vertical flips |
| `off_axis_flip_h`, `off_axis_flip_v` | BOOLEAN | NOT NULL, default false | Horizontal/vertical flips |
| `updated_at` | TIMESTAMPTZ | NULLABLE | Last config update time |

## Session Lifecycle

The database and filesystem move together during a session:

1. `recording`: session row is created, directory is created, camera service writes `on_axis.svo2` and `off_axis.svo2`
2. `exporting`: exporter writes MP4, depth, sample JPEG, and `tip_detections.json`
3. `awaiting_init`: user must confirm tip positions
4. `completed`: `tip_init.json` is saved and grading may start
5. `grading`: grader writes `results/` artifacts and `grading_results`
6. `graded`: session is fully processed

`re-export` may move a session back to `exporting` while preserving whether the post-export state should be `awaiting_init`, `completed`, or `graded`.

## Redis Job State

Redis stores worker coordination, not authoritative session history.

### Queue keys

| Key | Type | Description |
|-----|------|-------------|
| `export_jobs` | list | Export jobs queued by the API |
| `grading_jobs` | list | Grading jobs queued by the API |

### Progress keys

| Key Pattern | Type | Description |
|-------------|------|-------------|
| `job_progress:{session_id}` | hash | Overall progress fields plus per-stage payloads under `stage__*` |
| `export_cancel:{session_id}` | string | Export cancellation flag |
| `model_download:{model_id}` | hash | Model download progress |

`job_progress:{session_id}` stores fields such as:

- `stage`
- `current`
- `total`
- `percent`
- `detail`
- `updated_at`
- `stage_started_at`
- `stage__export_on_axis`
- `stage__export_off_axis`
- `stage__detect_tips`
- `stage__load_on_axis`
- `stage__detect_on_axis`
- and other stage-specific JSON blobs

## Session Directory Layout

Per-session files live under:

```text
/data/users/<user_id>/<YYYY-MM-DD_HH-MM-SS>/
```

Typical layout after export and grading:

```text
/data/users/<user_id>/<timestamp>/
├── session_metadata.json
├── on_axis.svo2
├── off_axis.svo2
├── calibration_on_axis.json
├── calibration_off_axis.json
├── stereo_calibration.json
├── on_axis_left.mp4
├── on_axis_right.mp4
├── off_axis_left.mp4
├── off_axis_right.mp4
├── on_axis_depth.npz
├── off_axis_depth.npz
├── on_axis_sample_0.jpg
├── on_axis_sample_1.jpg
├── off_axis_sample_0.jpg
├── off_axis_sample_1.jpg
├── tip_detections.json
├── tip_init.json
└── results/
    ├── metrics.json
    ├── poses.json
    ├── tracking_on_axis.mp4
    ├── tracking_off_axis.mp4
    ├── tracking_on_axis.csv
    ├── tracking_off_axis.csv
    └── tracked_positions_world.csv
```

Notes:

- `tip_init.json` only exists after the user confirms tips
- `results/` only exists after grading runs
- re-export overwrites exported MP4/NPZ/sample artifacts for that session directory

## File Formats

### `metrics.json`

Produced by the grading pipeline. Contains surgical skill metrics.

```json
{
  "workspace_volume": 125.4,
  "avg_speed": 15.2,
  "max_jerk": 450.1,
  "path_length": 342.8,
  "economy_of_motion": 0.72,
  "total_time": 900.0
}
```

| Metric | Unit | Description |
|--------|------|-------------|
| `workspace_volume` | cm^3 | Convex hull volume of all 3D tip positions. Requires at least 4 valid points |
| `avg_speed` | mm/s | Mean speed across all frames for both tips |
| `max_jerk` | mm/s^3 | Peak third derivative of position |
| `path_length` | mm | Total Euclidean distance traveled |
| `economy_of_motion` | 0-1 | Direct distance divided by total path |
| `total_time` | seconds | Duration from first to last processed frame |

### `poses.json`

Per-frame fused 3D instrument positions.

```json
[
  {
    "frame_idx": 0,
    "timestamp": 0.0,
    "left_tip": [0.15, -0.03, 0.42],
    "right_tip": [0.22, -0.01, 0.38]
  },
  {
    "frame_idx": 5,
    "timestamp": 0.167,
    "left_tip": [0.16, -0.02, 0.41],
    "right_tip": null
  }
]
```

Coordinates are in metres. When calibration is available, they are in the board/world frame; otherwise they remain in the active camera frame.

### `calibration_*.json`

Saved per-session camera calibration.

```json
{
  "intrinsics": {
    "fx": 1065.0,
    "fy": 1065.0,
    "cx": 960.0,
    "cy": 600.0,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    "image_width": 1920,
    "image_height": 1200
  },
  "extrinsic_matrix": [
    [0.99, 0.01, -0.02, 0.15],
    [-0.01, 0.99, 0.03, -0.08],
    [0.02, -0.03, 0.99, 0.42],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "board_config": {
    "rows": 9,
    "cols": 14,
    "square_size_mm": 20.0,
    "marker_size_mm": 15.0,
    "aruco_dict": "DICT_5X5_100"
  },
  "reprojection_error": 0.45,
  "num_frames_used": 7
}
```

### `stereo_calibration.json`

Inter-camera transform and per-camera calibration snapshots.

```json
{
  "T_on_to_off": [
    [0.99, 0.01, -0.02, 0.25],
    [-0.01, 0.99, 0.03, -0.05],
    [0.02, -0.03, 0.99, 0.00],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "on_axis": { "...calibration data..." },
  "off_axis": { "...calibration data..." }
}
```

`T_on_to_off` is the homogeneous transform from the on-axis camera frame into the off-axis camera frame.

### `tip_detections.json`

Auto-detected tip positions from color-based detection on exported sample frames.

```json
{
  "on_axis_sample_0.jpg": [
    {"label": "left_tip", "x": 450.2, "y": 320.1, "confidence": 0.85, "color": "green"},
    {"label": "right_tip", "x": 890.5, "y": 310.3, "confidence": 0.92, "color": "pink"}
  ]
}
```

### `tip_init.json`

User-confirmed tip positions created in the Tip Initialization workflow.

```json
{
  "on_axis_sample_0.jpg": [
    {"label": "left_tip", "x": 455.0, "y": 318.0},
    {"label": "right_tip", "x": 892.0, "y": 312.0}
  ]
}
```

### Depth NPZ files

`{camera}_depth.npz` is a ZIP archive of numpy arrays:

```text
frame_000000.npy  # float32, shape (H, W), values in metres
frame_000001.npy
frame_000002.npy
...
```

Load in Python:

```python
import numpy as np
import zipfile

with zipfile.ZipFile("on_axis_depth.npz", "r") as z:
    names = sorted(z.namelist())
    depth_frame = np.load(z.open(names[0]))  # shape (1200, 1920), dtype float32
```

### Tracking CSVs

`tracking_{camera}.csv` stores 2D detections:

```csv
frame_idx,timestamp_s,camera,label,x,y,confidence
0,0.000,on_axis,left_tip,450.2,320.1,0.92
0,0.000,on_axis,right_tip,890.5,310.3,0.88
5,0.167,on_axis,left_tip,452.1,318.5,0.91
```

`tracked_positions_world.csv` stores fused 3D tip positions:

```csv
frame_idx,timestamp_s,left_tip_x_m,left_tip_y_m,left_tip_z_m,right_tip_x_m,right_tip_y_m,right_tip_z_m
0,0.000,0.15,-0.03,0.42,0.22,-0.01,0.38
5,0.167,0.16,-0.02,0.41,,,
```

Empty values indicate that the tip was not available in that frame.

### Tracking videos

`tracking_on_axis.mp4` and `tracking_off_axis.mp4` are rendered overlay videos showing:

- the current tip detections
- trailing polylines for recent motion
- per-tip labels with confidence
