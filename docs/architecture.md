# Architecture

## System Overview

Lap-Trackr is a Docker-containerized system that runs on an NVIDIA Jetson AGX Orin. It captures stereo video from two ZED X cameras, exports recordings to portable formats, runs an offline multi-pass grading pipeline, estimates 3D poses, and computes surgical skill metrics.

## Service Topology

```
                    Browser
                      |
               ┌──────┴──────┐
               │    nginx     │
               │  :80  :8081  │
               └──┬──────┬───┘
          /api/*  │      │  /ws/camera/*
                  │      │  /stream/* (8081)
           ┌──────┴──┐ ┌─┴──────────┐
           │   API   │ │   Camera   │
           │  :8000  │ │   :8001    │
           └──┬──┬───┘ └────────────┘
              │  │
     ┌────────┘  └────────┐
     │                    │
┌────┴─────┐        ┌────┴────┐
│ Postgres │        │  Redis  │
│  :5432   │        │  :6379  │
└──────────┘        └────┬────┘
                         │
              ┌──────────┼──────────┐
              │                     │
        ┌─────┴─────┐        ┌─────┴─────┐
        │  Grader   │        │ Exporter  │
        │ (worker)  │        │ (worker)  │
        └───────────┘        └───────────┘

  React SPA is served as static files by nginx.
  Frontend container only runs at build time to produce assets.
```

## Services

| Service | Image | Port | Role |
|---------|-------|------|------|
| **nginx** | `nginx:1.25-bookworm` | 80, 8081 | Reverse proxy. Serves frontend static files, routes `/api/*` to API, `/ws/camera/*` and `/stream/*` to camera |
| **api** | Custom (Python 3.10 slim) | 8000 | FastAPI REST backend. Auth, session lifecycle, model management, calibration, job dispatch |
| **camera** | Custom (ZED SDK base on Jetson; Python slim in dev) | 8001 | MJPEG streaming, SVO2 recording, ChArUco calibration |
| **grader** | Custom (Python 3.10 + PyTorch) | -- | Redis worker. Consumes exported artifacts, runs offline grading passes, triangulates and smooths 3D trajectories, computes metrics |
| **exporter** | Custom (ZED SDK runtime on Jetson) | -- | Redis worker. Converts SVO2 to MP4 + NPZ, extracts sample frames, writes export metadata, runs initial color-based tip detection |
| **frontend** | Node 20 build -> Alpine copy | -- | Build-only container. Compiles React/Vite app, outputs to shared volume |
| **db** | `postgres:15-bookworm` | 5432 | PostgreSQL. Users, sessions, grading results, calibrations, model catalog |
| **redis** | `redis:7-bookworm` | 6379 | Job queues (`export_jobs`, `grading_jobs`), progress tracking, model download progress |

## Networking

All services communicate over a Docker bridge network named `internal`. No service ports are exposed to the host except nginx (80, 8081).

### Nginx Routing

| Path | Upstream | Notes |
|------|----------|-------|
| `/api/auth/*` | `api:8000` | Rate-limited (5 req/s). Prefix `/api` stripped before forwarding |
| `/api/*` | `api:8000` | 300s timeout for long operations. Prefix `/api` stripped |
| `/ws/camera/*` | `camera:8001` | No buffering, 86400s timeout for MJPEG. Prefix `/ws/camera` stripped |
| `/stream/*` (port 8081) | `camera:8001` | Dedicated port for MJPEG streams to avoid browser connection pool exhaustion |
| `/assets/*` | Static files | 1-year immutable cache |
| `/*` | Static files | SPA fallback: `try_files $uri $uri/ /index.html` |

### Why Two Ports?

Browsers limit concurrent connections per host (typically 6). MJPEG streams hold connections open indefinitely. By serving streams on port 8081 and the app on port 80, the browser treats them as separate origins with independent connection pools. This prevents live camera feeds from blocking API requests.

## Session Lifecycle

```
┌────────────┐     ┌────────────┐     ┌──────────────┐     ┌───────────┐     ┌─────────┐     ┌────────┐
│  recording  │────▸│  exporting  │────▸│ awaiting_init │────▸│ completed │────▸│ grading │────▸│ graded │
└────────────┘     └─────┬──────┘     └──────────────┘     └───────────┘     └────┬────┘     └────────┘
                         │                                                        │
                         ▼                                                        ▼
                  ┌──────────────┐                                          ┌──────────┐
                  │ export_failed │                                          │  failed  │
                  └──────────────┘                                          └──────────┘
```

1. **recording**: Camera service is writing SVO2 files
2. **exporting**: Export worker converts SVO2 to MP4 + NPZ depth
3. **awaiting_init**: Export complete; user must confirm instrument tip positions on sample frames
4. **completed**: Ready for grading (tip positions confirmed or auto-detected)
5. **grading**: Grading worker is processing the offline multi-pass pipeline (segmentation, tracking, fusion, smoothing, metrics)
6. **graded**: Metrics available

If tip auto-detection is confident (tip_init.json already exists from a previous run), the session skips `awaiting_init` and goes directly to `completed`.

## Data Flow

### Recording
```
User clicks Start ──▸ API POST /sessions/start
  ──▸ API creates session dir, copies default calibrations
  ──▸ API calls Camera POST /record/start
  ──▸ Camera opens SVO2 writers, starts grab thread
  ──▸ SVO2 files written to /data/users/{user_id}/{timestamp}/
```

### Export
```
User clicks Stop ──▸ API POST /sessions/{id}/stop
  ──▸ API calls Camera POST /record/stop
  ──▸ API LPUSH export_jobs (Redis)
  ──▸ Export worker BRPOP export_jobs
  ──▸ SVO2 ──▸ MP4 (hardware NVENC or software) + NPZ (depth)
  ──▸ Extract representative sample frames for initialization
  ──▸ Color-detect tips on samples (green/pink HSV thresholding)
  ──▸ Save tip_detections.json + tip_init_samples.json
  ──▸ Status ──▸ awaiting_init or completed
```

### Grading
```
User clicks Grade ──▸ API POST /sessions/{id}/grade
  ──▸ API LPUSH grading_jobs (Redis)
  ──▸ Grading worker BRPOP grading_jobs
  ──▸ Load exported MP4+NPZ artifacts and initialization metadata
  ──▸ Pass 1: SAM2 per-view segmentation
  ──▸ Pass 2: CoTracker3 tip refinement from confirmed tip-init points
  ──▸ Pass 3: Color-based gap fill and identity checks
  ──▸ Pass 4: Multi-view triangulation with reprojection residuals
  ──▸ Pass 5: Full-trajectory smoothing / optimization
  ──▸ Pass 6: Final green/pink identity verification
  ──▸ Render tracking overlay videos
  ──▸ Compute metrics (workspace volume, speed, jerk, path length, economy, duration)
  ──▸ Save results to DB + JSON files
  ──▸ Status ──▸ graded
```

See [Offline Grading Pipeline](offline-grading-pipeline.md) for the detailed target design.

## File System Layout

```
/data/
├── users/{user_id}/{YYYY-MM-DD_HH-MM-SS}/
│   ├── on_axis.svo2                    # Raw stereo recording (on-axis camera)
│   ├── off_axis.svo2                   # Raw stereo recording (off-axis camera)
│   ├── on_axis_left.mp4                # Exported left-eye video
│   ├── on_axis_right.mp4               # Exported right-eye video
│   ├── off_axis_left.mp4
│   ├── off_axis_right.mp4
│   ├── on_axis_depth.npz               # Depth frames (numpy arrays)
│   ├── off_axis_depth.npz
│   ├── on_axis_sample_0.jpg            # Sample frame (first)
│   ├── on_axis_sample_1.jpg            # Sample frame (middle)
│   ├── on_axis_sample_2.jpg            # Sample frame (last)
│   ├── off_axis_sample_*.jpg
│   ├── calibration_on_axis.json        # Camera intrinsics + extrinsic
│   ├── calibration_off_axis.json
│   ├── stereo_calibration.json         # Inter-camera transform
│   ├── tip_detections.json             # Auto-detected tip positions from color
│   ├── tip_init.json                   # User-confirmed tip positions
│   ├── tip_init_samples.json           # Sample filename -> source frame metadata
│   ├── session_metadata.json           # Session info, camera serials, SDK version
│   └── results/
│       ├── metrics.json                # Skill metrics
│       ├── poses.json                  # Per-frame 3D positions
│       ├── tracking_on_axis.csv        # 2D detections per frame
│       ├── tracking_off_axis.csv
│       ├── tracking_on_axis_cotracker.csv
│       ├── tracking_off_axis_cotracker.csv
│       ├── tracking_on_axis_yolo.csv
│       ├── tracking_off_axis_yolo.csv
│       ├── tracking_on_axis_color.csv
│       ├── tracking_off_axis_color.csv
│       ├── tracked_positions_world.csv # 3D world positions per frame
│       ├── tracking_on_axis.mp4        # Overlay video with detection trails
│       └── tracking_off_axis.mp4
├── calibration/
│   └── default/
│       ├── on_axis.json                # Global default on-axis calibration
│       ├── off_axis.json               # Global default off-axis calibration
│       └── stereo_calibration.json     # Global default stereo transform
├── models/
│   ├── cotracker/cotracker-v3-offline/ # Offline CoTracker3 weights
│   ├── yolov11-pose/                   # Auxiliary YOLO pose weights
│   ├── sam2/                           # SAM2 segmentation weights
│   └── custom/{slug}/                  # User-uploaded models
├── postgres/                           # PostgreSQL data directory
└── redis/                              # Redis persistence
```

## Security Model

- **Authentication**: JWT Bearer tokens (HS256), required on all endpoints except `/auth/register`, `/auth/login`, and `/health`
- **Startup guard**: API refuses to start if `JWT_SECRET` is left at the default placeholder value
- **Authorization**: Users can only access their own sessions; path traversal guards on all file-serving endpoints
- **CORS**: Configurable via `CORS_ORIGINS`. Credentials only enabled with explicit origin list (not `*`)
- **Rate limiting**: Auth endpoints rate-limited at 5 req/s via nginx
- **Security headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy, Content-Security-Policy
- **File uploads**: Only `.pt` files accepted, max 500 MB, path escape validation
- **Database**: SELECT ... FOR UPDATE locks on stop/delete/grade to prevent race conditions
