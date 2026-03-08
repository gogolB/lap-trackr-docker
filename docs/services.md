# Services Deep Dive

Detailed internals of each microservice for developers extending the system.

## API Service (`services/api/`)

### Technology

- Python 3.10, FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2
- Runs via `uvicorn` on port 8000
- Uses `asyncpg` for PostgreSQL, `redis` for Redis, `httpx` for outgoing HTTP

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app init, lifespan (startup/shutdown), CORS, router registration |
| `app/core/config.py` | Pydantic BaseSettings, reads `.env` |
| `app/core/auth.py` | JWT creation/verification, password hashing (bcrypt), auth dependencies |
| `app/core/database.py` | Async SQLAlchemy engine (pool_size=20, max_overflow=10), session factory |
| `app/models/models.py` | ORM models: User, Session, GradingResult, Calibration, MLModel, CameraConfig |
| `app/schemas/schemas.py` | Pydantic request/response schemas |
| `app/model_registry.py` | Static catalog of available ML models (with download URLs) |
| `app/routers/*.py` | Route handlers for each domain |

### Startup Behavior

1. **JWT guard**: Refuses to start if `JWT_SECRET` matches known defaults
2. **Schema initialization**: Runs Alembic automatically at startup
3. **Legacy bootstrap fallback**: If the DB has no `alembic_version`, bootstraps the current schema and stamps head
4. **Model seeding**: Inserts catalog models from `MODEL_CATALOG` if not already present
5. **Stale-session sweep**: Marks sessions stuck in `exporting` or `grading` for >30 minutes as failed
6. **Periodic sweep**: Re-runs the stale-session sweep every 5 minutes

### Database Migrations

Alembic is configured for async migrations. Migration files live in `services/api/migrations/versions/`.

```bash
# Generate a new migration
docker compose exec api alembic revision --autogenerate -m "description"

# Run pending migrations
docker compose exec api alembic upgrade head
```

In normal operation, the API applies pending migrations automatically on startup. Manual `alembic upgrade head` is mainly for debugging, preflight checks, or repairing a DB before the API is restarted.

All new schema changes should be migration-backed. Do not add new one-off schema mutations in application startup code.

### Adding a New Router

1. Create `app/routers/my_feature.py` with a FastAPI `APIRouter`
2. Add schemas to `app/schemas/schemas.py`
3. Import and register in `app/main.py`:
   ```python
   from app.routers import my_feature
   app.include_router(my_feature.router, prefix="/my-feature", tags=["my-feature"])
   ```

---

## Camera Service (`services/camera/`)

### Technology

- Python 3.10, FastAPI, OpenCV, ZED SDK (production only)
- Runs via `uvicorn` on port 8001
- Auto-detects ZED SDK at startup; falls back to `MockCameraManager` if unavailable

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, SDK auto-detection, all endpoints |
| `app/camera_manager.py` | Real ZED camera control: open, record, stream, calibrate |
| `app/camera_manager_mock.py` | Mock replacement generating test patterns |
| `app/calibrator.py` | ChArUco board detection and extrinsic computation |
| `app/config.py` | Environment variables |

### Camera Manager Interface

Both `CameraManager` and `MockCameraManager` implement the same interface:

```python
open_cameras() -> None
close() -> None
start_recording(session_dir: str) -> dict  # returns {on_axis_path, off_axis_path}
stop_recording() -> None
get_frame(camera_name: str, eye: str = "left") -> bytes  # JPEG
capture_calibration_frame(camera_name: str) -> tuple[bytes, ndarray]
apply_config(config: dict) -> None
get_intrinsics(camera_name: str) -> dict
get_camera_info() -> dict
list_cameras() -> list
status() -> dict
```

### Recording Architecture

The real `CameraManager` uses a background grab thread:

1. `start_recording()` enables SVO2 recording (H264 compression) on each camera, then starts `_grab_loop()` in a daemon thread
2. `_grab_loop()` continuously calls `grab()` on all cameras and stores the latest left-eye image in `_latest_images` for MJPEG streaming
3. `get_frame()` reads from `_latest_images` when recording (the grab thread owns `grab()`). When idle, it calls `grab()` directly.
4. Per-camera locks prevent grab/retrieve race conditions
5. Error tracking: logs on 1st error and every 100th; stops after 500 consecutive errors per camera

### Image Transforms

Transforms are applied in this order: **rotation -> flip_h -> flip_v**

- Rotation: 0, 90, 180, 270 degrees (cv2.rotate)
- Flip H: cv2.flip(img, 1)
- Flip V: cv2.flip(img, 0)
- Eye swap: when `swap_eyes=True`, left retrieves right-eye data and vice versa

### Adding a New Camera Type

To support a camera other than ZED X:

1. Create `app/camera_manager_new.py` implementing the same interface as `CameraManager`
2. Update the auto-detection logic in `app/main.py` to try importing the new manager
3. Match the same method signatures so all endpoints work unchanged

---

## Grader Service (`services/grader/`)

### Technology

- Production: ZED SDK runtime image (`stereolabs/zed:5.2-py-runtime-jetson-jp5.1.2`, Python 3.8), Redis (BRPOP worker), SQLAlchemy (sync/psycopg2), scipy, numpy
- Development: lightweight Python image, no ZED SDK, no PyTorch -- falls back to synthetic data and placeholder backend
- Production ML stack includes PyTorch and optional Ultralytics YOLO models

### Key Files

| File | Purpose |
|------|---------|
| `app/worker.py` | Main grading worker loop (Redis BRPOP on `grading_jobs`) |
| `app/export_worker.py` | Export worker loop (Redis BRPOP on `export_jobs`) |
| `app/pipeline.py` | Orchestrates: load -> detect -> render -> pose -> metrics |
| `app/exporter.py` | SVO2 to MP4 + NPZ conversion with hardware/software encoding |
| `app/svo_loader.py` | Loads SVO2 via ZED SDK, fallback to MP4+NPZ, fallback to synthetic |
| `app/model_loader.py` | Factory: loads active ML backend from DB, caches instance |
| `app/pose_estimator.py` | 2D detections + depth -> 3D world coordinates |
| `app/fusion.py` | Dual-camera fusion (monocular depth + stereo triangulation) |
| `app/metrics.py` | Surgical skill metrics calculation |
| `app/tracking_renderer.py` | Renders overlay videos and exports CSV detection/pose records |
| `app/color_detector.py` | HSV-based tip detection (green/pink tape) |
| `app/db.py` | Synchronous DB operations (psycopg2) |
| `app/config.py` | Environment variables |
| `app/backends/*.py` | ML model backends |

### Worker Architecture

The grader runs two separate worker processes (separate containers, same image):

1. **Export worker** (`python3 -m app.export_worker`): Listens on `export_jobs` Redis queue
2. **Grading worker** (`python3 -m app.worker`): Listens on `grading_jobs` Redis queue

Both use `BRPOP` (blocking pop) with a 5-second timeout. Jobs are JSON strings pushed via `LPUSH` by the API.

The exporter processes `on_axis` and `off_axis` in parallel within a single session job, then runs initial color-based tip detection across the extracted sample frames.

### Pipeline Stages

The grading pipeline in `pipeline.py` runs these stages sequentially:

1. **load_on_axis** / **load_off_axis**: Load video frames and depth maps
2. **detect_on_axis** / **detect_off_axis**: Run ML backend on frames
3. **render_on_axis** / **render_off_axis**: Render tracking overlay videos
4. **estimate_poses**: Convert 2D detections to 3D using depth + calibration
5. **calculate_metrics**: Compute surgical skill metrics

Each stage publishes progress to Redis (`job_progress:{session_id}` hash) for real-time tracking by the frontend.

### SVO2 Loading Fallback Chain

`svo_loader.py` tries three strategies in order:

1. **ZED SDK**: Opens SVO2 directly, uses neural depth estimation (GPU), samples every Nth frame
2. **Exported files**: Reads `{camera}_left.mp4` (OpenCV) + `{camera}_depth.npz` (numpy)
3. **Synthetic data**: Generates random frames and depth maps (seed=42 for reproducibility)

### Export Process

`exporter.py` converts one SVO2 file to:

- `{camera}_left.mp4` and `{camera}_right.mp4` (left/right eye videos)
- `{camera}_depth.npz` (depth frames as numpy arrays in a ZIP archive)
- `*_sample_*.jpg` (sample frames used for tip initialization)

Video encoding tries:
1. **GStreamer + NVENC** (Jetson hardware encoder, 20 Mbps H.264)
2. **OpenCV avc1** (software H.264)
3. **OpenCV mp4v** (software MPEG-4)

After both cameras finish exporting, the worker writes `tip_detections.json` from the extracted sample frames and advances the session to `awaiting_init`, `completed`, or `graded` depending on existing session artifacts.

### Database Access

The grader uses synchronous SQLAlchemy (psycopg2) because it runs in a single-threaded worker loop, not an async web server. The `DATABASE_URL` from `.env` is automatically converted from `asyncpg://` to `psycopg2://` at import time in `config.py`.

---

## Frontend Service (`services/frontend/`)

### Technology

- React 18, TypeScript, Vite, TailwindCSS v3, React Router v6, TanStack Query v5, Recharts

### Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Router setup, error boundary, React Query client |
| `src/api/client.ts` | Centralized API client (fetch-based, JWT handling) |
| `src/hooks/useAuth.ts` | Auth context (login, register, logout, current user) |
| `src/components/Layout.tsx` | Nav bar, sidebar, user menu |
| `src/components/ProtectedRoute.tsx` | Auth guard wrapper |
| `src/pages/Dashboard.tsx` | Session stats, recent sessions |
| `src/pages/LiveView.tsx` | Camera feeds, recording controls, calibration panel |
| `src/pages/SessionList.tsx` | Session table with status badges |
| `src/pages/SessionDetail.tsx` | Session management, progress tracking, metrics display |
| `src/pages/TipInitPage.tsx` | Canvas-based tip position editor |
| `src/pages/ConfigPage.tsx` | Camera serial/rotation/flip configuration |
| `src/pages/ModelsPage.tsx` | ML model download/upload/activate |
| `src/pages/HealthPage.tsx` | System health dashboard |
| `src/pages/LoginPage.tsx` | Login/register form |

### Build Process

The frontend is a **build-only container**. It:

1. Runs `npm ci && npm run build` during `docker build`
2. At container start, copies the built files to a shared volume (`frontend_dist`)
3. Nginx serves these static files

For local development with hot reload, see [Development Setup - Frontend Hot Reload](setup-dev.md#frontend-development-with-hot-reload).

### API Client Pattern

All API calls go through `src/api/client.ts`. Key features:

- Base URL is empty string (relies on nginx or Vite proxy)
- JWT stored in `localStorage` as `lap_trackr_token`
- 30-second timeout on all requests
- 401 responses automatically clear the token and redirect to `/login`
- Model uploads use separate `multipart/form-data` fetch

### Adding a New Page

1. Create `src/pages/MyPage.tsx`
2. Add a route in `src/App.tsx`:
   ```tsx
   <Route path="/my-page" element={<ProtectedRoute><MyPage /></ProtectedRoute>} />
   ```
3. Add a nav link in `src/components/Layout.tsx`
4. Add API functions to `src/api/client.ts` if needed
5. Rebuild: `dc-dev build frontend && dc-dev up -d frontend nginx`

---

## Nginx (`services/nginx/`)

### Configuration

`nginx.conf` defines two server blocks:

- **Port 80**: Main HTTP server (frontend + API + camera WebSocket routes)
- **Port 8081**: Dedicated MJPEG stream server

### Key Behaviors

- `/api/*` requests: prefix stripped, forwarded to `api:8000` with 300s timeout
- `/api/auth/*`: rate-limited to 5 req/s (burst 10)
- `/ws/camera/*`: prefix stripped, forwarded to `camera:8001` with no buffering and 86400s timeout
- `/stream/*` on port 8081: forwarded to `camera:8001` with no buffering
- Frontend assets: `try_files $uri $uri/ /index.html` for SPA routing
- `/assets/*`: 1-year immutable cache headers
- Gzip enabled for common text types
- Security headers applied globally

### Modifying Routes

Edit `services/nginx/nginx.conf`. Changes take effect after:

```bash
docker compose restart nginx
```

No rebuild needed since the config is bind-mounted (`:ro`).

---

## Database (`db`)

PostgreSQL 15 with data stored at `/data/postgres`.

See [Database & Migrations](database.md) for schema ownership, migration workflow, and backup/restore guidance.

### Connection Details

- Host: `db` (Docker service name)
- Port: 5432
- Database: `laptrackr` (configurable)
- User: `laptrackr` (configurable)

### Health Check

`pg_isready -U ${POSTGRES_USER}` every 5 seconds.

---

## Redis

Redis 7 with data stored at `/data/redis`.

### Queues

| Queue | Consumer | Job Format |
|-------|----------|------------|
| `export_jobs` | Export worker | `{"session_id": "uuid", "on_axis_path": "...", "off_axis_path": "...", ...}` |
| `grading_jobs` | Grading worker | `{"session_id": "uuid", "on_axis_path": "...", "off_axis_path": "...", ...}` |

### Progress Hashes

| Key Pattern | Used By | Purpose |
|-------------|---------|---------|
| `job_progress:{session_id}` | Workers | Overall stage plus per-stage JSON payloads (`stage__*`) with counts, timestamps, percent, detail, and status |
| `model_download:{model_id}` | API | Downloaded bytes, total bytes, percent |
| `export_cancel:{session_id}` | Export worker | Cancellation signal |
