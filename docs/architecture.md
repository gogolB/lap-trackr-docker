# Architecture

## System Overview

Lap-Trackr is a Docker-containerized system that runs on an NVIDIA Jetson AGX Orin. It captures stereo video from two ZED X cameras, exports recordings to portable formats, detects instrument tips using ML models, estimates 3D poses, and computes surgical skill metrics.

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
| **grader** | Custom (ZED SDK + PyTorch on Jetson; Python slim in dev) | -- | Redis worker. Loads SVO2, runs ML detection, estimates 3D poses, computes metrics |
| **exporter** | Same image as grader, different entrypoint | -- | Redis worker. Converts SVO2 to MP4 + NPZ, extracts sample frames, runs color-based tip detection |
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
5. **grading**: Grading worker is processing (ML detection, pose estimation, metrics)
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
  ──▸ Extract 3 sample frames (first, middle, last)
  ──▸ Color-detect tips on samples (green/pink HSV thresholding)
  ──▸ Save tip_detections.json
  ──▸ Status ──▸ awaiting_init or completed
```

### Grading
```
User clicks Grade ──▸ API POST /sessions/{id}/grade
  ──▸ API LPUSH grading_jobs (Redis)
  ──▸ Grading worker BRPOP grading_jobs
  ──▸ Load SVO2 (or MP4+NPZ fallback, or synthetic)
  ──▸ Load active ML backend (YOLO/CoTracker/SAM2/TAPIR/Placeholder)
  ──▸ Detect instrument tips (2D) on sampled frames
  ──▸ Render tracking overlay videos
  ──▸ Back-project 2D detections + depth ──▸ 3D poses
  ──▸ If dual-camera: fuse with stereo calibration
  ──▸ Compute metrics (workspace volume, speed, jerk, path length, economy, duration)
  ──▸ Save results to DB + JSON files
  ──▸ Status ──▸ graded
```

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
│   ├── session_metadata.json           # Session info, camera serials, SDK version
│   └── results/
│       ├── metrics.json                # Skill metrics
│       ├── poses.json                  # Per-frame 3D positions
│       ├── tracking_on_axis.csv        # 2D detections per frame
│       ├── tracking_off_axis.csv
│       ├── tracked_positions_world.csv # 3D world positions per frame
│       ├── tracking_on_axis.mp4        # Overlay video with detection trails
│       └── tracking_off_axis.mp4
├── calibration/
│   └── default/
│       ├── on_axis.json                # Global default on-axis calibration
│       ├── off_axis.json               # Global default off-axis calibration
│       └── stereo_calibration.json     # Global default stereo transform
├── models/
│   ├── point_tracking/cotracker-v2/    # Downloaded model weights
│   ├── segmentation/sam2-hiera-large/
│   ├── detection/yolov8n/
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
