# API Reference

The API service is a FastAPI application running on port 8000. Nginx proxies requests from `/api/*` on port 80, stripping the `/api` prefix before forwarding.

Base URL: `http://<host>/api` (via nginx) or `http://<host>:8000` (direct).

## Authentication

All endpoints except `/auth/register`, `/auth/login`, and `/health` require a JWT Bearer token.

```
Authorization: Bearer <token>
```

Tokens are obtained via the login endpoint and expire after `JWT_EXPIRY_MINUTES` (default: 1440 minutes / 24 hours).

Some endpoints (sample frame serving) also accept `?token=<jwt>` as a query parameter for embedding in `<img>` tags.

---

## Auth Endpoints

### POST /auth/register

Create a new user account.

**Request:**
```json
{
  "username": "surgeon1",
  "password": "securepassword"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "username": "surgeon1",
  "created_at": "2026-03-08T10:00:00Z"
}
```

**Errors:** 409 if username already taken.

### POST /auth/login

Authenticate and receive a JWT token.

**Request:**
```json
{
  "username": "surgeon1",
  "password": "securepassword"
}
```

**Response (200):**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

**Errors:** 401 if credentials are invalid.

### GET /auth/me

Get the current authenticated user.

**Response (200):**
```json
{
  "id": "uuid",
  "username": "surgeon1",
  "created_at": "2026-03-08T10:00:00Z"
}
```

---

## Session Endpoints

Session status lifecycle:

- `recording -> exporting -> awaiting_init -> completed -> grading -> graded`
- `export_failed` and `failed` are retryable failure states
- `re-export` can move an existing session back to `exporting` without creating a new session

### GET /sessions/

List sessions for the authenticated user.

**Query Parameters:**
- `skip` (int, default 0): Pagination offset
- `limit` (int, default 100): Max results

**Response (200):**
```json
[
  {
    "id": "uuid",
    "user_id": "uuid",
    "name": "Practice Session 1",
    "started_at": "2026-03-08T10:00:00Z",
    "stopped_at": "2026-03-08T10:15:00Z",
    "status": "graded",
    "on_axis_path": "/data/users/.../on_axis.svo2",
    "off_axis_path": "/data/users/.../off_axis.svo2",
    "created_at": "2026-03-08T10:00:00Z"
  }
]
```

### GET /sessions/{id}

Get session details including grading result (if available).

**Response (200):**
```json
{
  "id": "uuid",
  "name": "Practice Session 1",
  "status": "graded",
  "started_at": "...",
  "stopped_at": "...",
  "grading_result": {
    "id": "uuid",
    "session_id": "uuid",
    "workspace_volume": 125.4,
    "avg_speed": 15.2,
    "max_jerk": 450.1,
    "path_length": 342.8,
    "economy_of_motion": 0.72,
    "total_time": 900.0,
    "completed_at": "...",
    "error": null,
    "warnings": []
  }
}
```

### POST /sessions/start

Start a new recording session.

**Request (optional):**
```json
{
  "name": "Peg Transfer Practice"
}
```

**What it does:**
1. Creates session directory at `/data/users/{user_id}/{timestamp}/`
2. Copies global default calibration files into the session directory
3. Calls camera service `POST /record/start` to begin SVO2 recording
4. Returns the session with status `recording`

**Response (201):** SessionOut object.

### POST /sessions/{id}/stop

Stop recording and queue export.

**What it does:**
1. Acquires database row lock (`SELECT ... FOR UPDATE`)
2. Calls camera service `POST /record/stop`
3. Pushes export job to Redis queue
4. Sets status to `exporting`

**Response (200):** SessionOut with status `exporting`.

### POST /sessions/{id}/grade

Queue the session for ML grading. Requires status `completed`.

**Response (200):** SessionOut with status `grading`.

### POST /sessions/{id}/re-export

Re-run the export pipeline for an existing session.

Behavior:

- rejects sessions in `recording` or `grading`
- if the session is already `exporting`, sets a Redis cancel flag and re-queues export
- clears the old progress hash before queueing the new export
- preserves the intended post-export state:
  - `awaiting_init` if no `tip_init.json` exists
  - `completed` if `tip_init.json` already exists
  - `graded` if the session was already graded

**Response (200):** SessionOut with status `exporting`.

### POST /sessions/{id}/retry

Retry a failed session.

- `export_failed` sessions re-enter `exporting`
- `failed` sessions re-enter `grading`

**Response (200):** SessionOut with status `exporting` or `grading`.

### GET /sessions/{id}/progress

Get real-time progress of an active export or grading job.

**Response (200):**
```json
{
  "session_id": "uuid",
  "status": "grading",
  "stage": "detect_on_axis",
  "current": 45,
  "total": 200,
  "percent": 22.5,
  "detail": "Processing frame 45/200",
  "updated_at": 1709884805.4,
  "stage_started_at": 1709884800.0,
  "stages": {
    "load_on_axis": {
      "stage": "load_on_axis",
      "current": 200,
      "total": 200,
      "percent": 100,
      "detail": "Loaded 200 sampled frames",
      "status": "completed",
      "updated_at": 1709884802.1,
      "started_at": 1709884797.0
    },
    "detect_on_axis": {
      "stage": "detect_on_axis",
      "current": 45,
      "total": 200,
      "percent": 22.5,
      "detail": "On-axis camera",
      "status": "running",
      "updated_at": 1709884805.4,
      "started_at": 1709884803.0
    }
  }
}
```

Export stage keys commonly include:

- `export_on_axis`
- `export_off_axis`
- `detect_tips`

Grading stage keys commonly include:

- `load_on_axis`
- `load_off_axis`
- `detect_on_axis`
- `detect_off_axis`
- `render_on_axis`
- `render_off_axis`
- `estimate_poses`
- `calculate_metrics`

### GET /sessions/{id}/download

Download a ZIP archive of the session directory. The archive may include:

- source `.svo2` files
- exported left/right MP4 files for both cameras
- depth `.npz` files
- calibration JSON
- sample JPEGs
- `tip_detections.json`
- `tip_init.json`
- `results/` artifacts including metrics, poses, tracking CSVs, and tracking videos

**Response:** `application/zip` stream. Symlinks are excluded for security.

### DELETE /sessions/{id}

Delete a session and all its files. Blocked while status is `recording`, `exporting`, or `grading`.

---

## Tip Initialization Endpoints

### GET /sessions/{id}/tip-init

Get auto-detected tip positions and sample frame filenames.

**Response (200):**
```json
{
  "detections": {
    "on_axis_sample_0.jpg": [
      {"label": "left_tip", "x": 450.2, "y": 320.1, "confidence": 0.85, "color": "green"},
      {"label": "right_tip", "x": 890.5, "y": 310.3, "confidence": 0.92, "color": "pink"}
    ]
  },
  "sample_frames": ["on_axis_sample_0.jpg", "on_axis_sample_1.jpg", "on_axis_sample_2.jpg"]
}
```

### GET /sessions/{id}/sample-frame/{filename}

Serve a sample frame JPEG. Accepts Bearer token or `?token=` query parameter.

### PUT /sessions/{id}/tip-init

Save user-confirmed tip positions and advance status to `completed`.

**Request:**
```json
{
  "tips": {
    "on_axis_sample_0.jpg": [
      {"label": "left_tip", "x": 455.0, "y": 318.0},
      {"label": "right_tip", "x": 892.0, "y": 312.0}
    ]
  }
}
```

---

## Results Endpoints

### GET /results/{session_id}

Fetch grading result summary from database.

### GET /results/{session_id}/metrics

Stream the full `metrics.json` file from disk.

### GET /results/{session_id}/poses

Stream the full `poses.json` file from disk.

---

## Model Management Endpoints

### GET /models/

List all ML models (built-in catalog + custom uploads).

**Response (200):**
```json
[
  {
    "id": "uuid",
    "slug": "yolov8n",
    "name": "YOLOv8 Nano",
    "model_type": "detection",
    "description": "Baseline object detection model",
    "version": "8.1.0",
    "download_url": "https://...",
    "file_size_bytes": 6200000,
    "file_path": null,
    "status": "available",
    "is_active": false,
    "is_custom": false
  }
]
```

### POST /models/{id}/download

Start downloading a model from its `download_url`. Runs as a background task.

**Requires:** status = `available` or `failed`.

### GET /models/{id}/progress

Poll download progress.

**Response (200):**
```json
{
  "model_id": "uuid",
  "status": "downloading",
  "downloaded_bytes": 52428800,
  "total_bytes": 120000000,
  "percent": 43.7,
  "error": null
}
```

### POST /models/{id}/activate

Activate a model (deactivates other non-custom models). The grader will use this model for subsequent jobs.

### DELETE /models/{id}

Delete a model. Catalog models reset to `available`; custom models are removed from the database.

### POST /models/upload

Upload a custom `.pt` model file.

**Request:** `multipart/form-data` with a `file` field.

**Constraints:**
- File must end with `.pt`
- Max 500 MB
- Filename max 200 characters

---

## Camera Configuration Endpoints

### GET /camera-config/

Get current camera configuration.

**Response (200):**
```json
{
  "on_axis_serial": "51127155",
  "off_axis_serial": "57422551",
  "on_axis_swap_eyes": false,
  "off_axis_swap_eyes": false,
  "on_axis_rotation": 0,
  "off_axis_rotation": 0,
  "on_axis_flip_h": false,
  "on_axis_flip_v": false,
  "off_axis_flip_h": false,
  "off_axis_flip_v": false,
  "updated_at": "2026-03-08T10:00:00Z"
}
```

### PUT /camera-config/

Update camera configuration (partial update -- only include fields you want to change).

### POST /camera-config/apply

Save current config to database AND push to the running camera service.

---

## Calibration Endpoints

### Single Camera

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/calibration/capture/{camera_name}` | POST | Capture frame and detect ChArUco corners |
| `/calibration/compute/{camera_name}` | POST | Compute extrinsic from accumulated captures |
| `/calibration/reset/{camera_name}` | POST | Reset accumulated captures |

`camera_name` must be `on_axis` or `off_axis`.

### Stereo

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/calibration/capture/stereo` | POST | Capture from both cameras simultaneously |
| `/calibration/compute/stereo` | POST | Compute per-camera extrinsics + inter-camera transform |
| `/calibration/reset/stereo` | POST | Reset captures for both cameras |

### Status and Defaults

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/calibration/status` | GET | Get capture counts and board config |
| `/calibration/defaults` | GET | List all saved default calibrations |
| `/calibration/defaults/{camera_name}` | GET | Get specific default calibration |
| `/calibration/defaults/{camera_name}` | DELETE | Delete a default calibration |

---

## Health Endpoint

### GET /health/system

Aggregate health check for all services. No authentication required.

**Response (200):**
```json
{
  "overall": "healthy",
  "services": {
    "api": {"status": "ok"},
    "database": {"status": "ok", "version": "PostgreSQL 15.4", "latency_ms": 2.1},
    "redis": {"status": "ok", "memory_used": "1.2M"},
    "camera": {"status": "ok", "recording": false, "cameras": 2},
    "grader": {"status": "ok", "pending_jobs": 0}
  }
}
```

Overall is `degraded` if any service reports an error.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Human-readable error message"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (invalid parameters, wrong session status) |
| 401 | Unauthorized (missing/invalid/expired JWT) |
| 403 | Forbidden (accessing another user's session) |
| 404 | Not found |
| 409 | Conflict (duplicate username, concurrent operation) |
| 422 | Validation error (malformed request body) |
| 502 | Bad gateway (camera service unreachable) |
