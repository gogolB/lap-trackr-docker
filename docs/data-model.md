# Data Model

## Database Schema

### users

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid4 | User identifier |
| username | VARCHAR | UNIQUE, NOT NULL | Login name |
| hashed_password | VARCHAR | NOT NULL | bcrypt hash |
| created_at | TIMESTAMP | NOT NULL, default now | Account creation time |

### sessions

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid4 | Session identifier |
| user_id | UUID | FK -> users.id, NOT NULL | Owning user |
| name | VARCHAR | NULLABLE | Optional session name |
| started_at | TIMESTAMP | NOT NULL | Recording start time |
| stopped_at | TIMESTAMP | NULLABLE | Recording stop time |
| status | ENUM | NOT NULL | Session lifecycle state |
| on_axis_path | VARCHAR | NULLABLE | Path to on-axis SVO2 file |
| off_axis_path | VARCHAR | NULLABLE | Path to off-axis SVO2 file |
| created_at | TIMESTAMP | NOT NULL, default now | Record creation time |

**Status enum values:** `recording`, `exporting`, `export_failed`, `awaiting_init`, `completed`, `grading`, `graded`, `failed`

**Cascade:** Deleting a user deletes all their sessions.

### grading_results

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid4 | Result identifier |
| session_id | UUID | FK -> sessions.id, UNIQUE | One result per session |
| workspace_volume | FLOAT | NULLABLE | Convex hull volume of tip positions (cm^3) |
| avg_speed | FLOAT | NULLABLE | Mean instrument speed (mm/s) |
| max_jerk | FLOAT | NULLABLE | Peak acceleration derivative (mm/s^3) |
| path_length | FLOAT | NULLABLE | Total distance traveled (mm) |
| economy_of_motion | FLOAT | NULLABLE | Ratio: direct distance / total path [0-1] |
| total_time | FLOAT | NULLABLE | Session duration (s) |
| completed_at | TIMESTAMP | NULLABLE | When grading finished |
| error | TEXT | NULLABLE | Error message if grading failed |
| warnings | JSONB | NULLABLE | List of non-fatal warning strings |

**Cascade:** Deleting a session deletes its grading result.

### calibrations

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid4 | Calibration identifier |
| camera_name | VARCHAR | NOT NULL | `on_axis` or `off_axis` |
| is_default | BOOLEAN | NOT NULL | Whether this is the global default |
| fx | FLOAT | NULLABLE | Focal length X (pixels) |
| fy | FLOAT | NULLABLE | Focal length Y (pixels) |
| cx | FLOAT | NULLABLE | Principal point X (pixels) |
| cy | FLOAT | NULLABLE | Principal point Y (pixels) |
| k1, k2, k3 | FLOAT | NULLABLE | Radial distortion coefficients |
| p1, p2 | FLOAT | NULLABLE | Tangential distortion coefficients |
| image_width | INTEGER | NULLABLE | Calibrated image width |
| image_height | INTEGER | NULLABLE | Calibrated image height |
| extrinsic_matrix | JSONB | NULLABLE | 4x4 camera-to-board transform |
| board_rows | INTEGER | NULLABLE | ChArUco board rows |
| board_cols | INTEGER | NULLABLE | ChArUco board columns |
| board_square_size_mm | FLOAT | NULLABLE | Square size in mm |
| board_marker_size_mm | FLOAT | NULLABLE | Marker size in mm |
| board_aruco_dict | VARCHAR | NULLABLE | ArUco dictionary name |
| reprojection_error | FLOAT | NULLABLE | RMS reprojection error (pixels) |
| num_frames_used | INTEGER | NULLABLE | Number of calibration frames |
| is_global | BOOLEAN | default false | Whether calibration applies globally |
| calibration_path | VARCHAR | NULLABLE | Path to saved JSON file |

### ml_models

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid4 | Model identifier |
| slug | VARCHAR | UNIQUE, NOT NULL | URL-safe identifier (e.g., `yolov8n`) |
| name | VARCHAR | NOT NULL | Display name |
| model_type | VARCHAR | NOT NULL | `detection`, `point_tracking`, `segmentation` |
| description | TEXT | NULLABLE | Human-readable description |
| version | VARCHAR | NULLABLE | Model version string |
| download_url | VARCHAR | NULLABLE | URL to download weights |
| file_size_bytes | BIGINT | NULLABLE | Expected download size |
| file_path | VARCHAR | NULLABLE | Local path to downloaded weights |
| status | ENUM | NOT NULL | Lifecycle state |
| is_active | BOOLEAN | default false | Whether this is the active model |
| is_custom | BOOLEAN | default false | Whether user-uploaded |
| created_at | TIMESTAMP | NOT NULL | Record creation time |
| updated_at | TIMESTAMP | NULLABLE | Last update time |

**Status enum values:** `available`, `downloading`, `ready`, `active`, `custom`, `failed`

### camera_config

Single-row table (id=1). Auto-created if missing.

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| id | INTEGER | 1 | Always 1 |
| on_axis_serial | VARCHAR | NULL | On-axis camera serial |
| off_axis_serial | VARCHAR | NULL | Off-axis camera serial |
| on_axis_swap_eyes | BOOLEAN | false | Swap left/right on on-axis |
| off_axis_swap_eyes | BOOLEAN | false | Swap left/right on off-axis |
| on_axis_rotation | INTEGER | 0 | Rotation degrees (0/90/180/270) |
| off_axis_rotation | INTEGER | 0 | Rotation degrees |
| on_axis_flip_h | BOOLEAN | false | Horizontal flip |
| on_axis_flip_v | BOOLEAN | false | Vertical flip |
| off_axis_flip_h | BOOLEAN | false | Horizontal flip |
| off_axis_flip_v | BOOLEAN | false | Vertical flip |
| updated_at | TIMESTAMP | now | Last update time |

## File Formats

### metrics.json

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
| workspace_volume | cm^3 | Convex hull volume of all 3D tip positions. Requires >= 4 points |
| avg_speed | mm/s | Mean speed across all frames for both tips |
| max_jerk | mm/s^3 | Peak 3rd derivative of position. Lower = smoother |
| path_length | mm | Total Euclidean distance traveled by both tips |
| economy_of_motion | 0-1 | Direct distance / total path. 1.0 = perfectly straight |
| total_time | seconds | Duration from first to last frame |

### poses.json

Per-frame 3D instrument tip positions.

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

Coordinates are in metres, in the camera frame (or board frame if extrinsic calibration is available). `null` values indicate the tip was not detected in that frame.

### calibration_*.json

Camera calibration saved per-session.

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

### stereo_calibration.json

Inter-camera transform.

```json
{
  "T_on_to_off": [
    [r00, r01, r02, tx],
    [r10, r11, r12, ty],
    [r20, r21, r22, tz],
    [0, 0, 0, 1]
  ],
  "on_axis": { "...calibration data..." },
  "off_axis": { "...calibration data..." }
}
```

`T_on_to_off` is a 4x4 homogeneous transform from on-axis camera frame to off-axis camera frame, derived as: `T_off_to_board @ inv(T_on_to_board)`.

### tip_detections.json

Auto-detected tip positions from color-based detection on sample frames.

```json
{
  "on_axis_sample_0.jpg": [
    {"label": "left_tip", "x": 450.2, "y": 320.1, "confidence": 0.85, "color": "green"},
    {"label": "right_tip", "x": 890.5, "y": 310.3, "confidence": 0.92, "color": "pink"}
  ]
}
```

### tip_init.json

User-confirmed tip positions (created via the Tip Init page).

```json
{
  "on_axis_sample_0.jpg": [
    {"label": "left_tip", "x": 455.0, "y": 318.0},
    {"label": "right_tip", "x": 892.0, "y": 312.0}
  ]
}
```

### depth NPZ files

`{camera}_depth.npz` is a ZIP archive of numpy arrays:

```
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

### tracking CSVs

**tracking_{camera}.csv** (2D detections):

```csv
frame_idx,timestamp_s,camera,label,x,y,confidence
0,0.000,on_axis,left_tip,450.2,320.1,0.92
0,0.000,on_axis,right_tip,890.5,310.3,0.88
5,0.167,on_axis,left_tip,452.1,318.5,0.91
```

**tracked_positions_world.csv** (3D poses):

```csv
frame_idx,timestamp_s,left_tip_x_m,left_tip_y_m,left_tip_z_m,right_tip_x_m,right_tip_y_m,right_tip_z_m
0,0.000,0.15,-0.03,0.42,0.22,-0.01,0.38
5,0.167,0.16,-0.02,0.41,,,
```

Empty values indicate the tip was not detected in that frame.
