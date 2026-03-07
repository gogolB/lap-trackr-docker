# Lap-Trackr

Capture and grade laparoscopic surgical training sessions using stereo cameras, instrument detection, and 3D kinematic analysis.

## Architecture

```
┌────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐
│ nginx  │──▸│   API    │──▸│ Camera  │   │ Grader  │
│  :80   │   │  :8000   │   │  :8001  │   │ (worker)│
└────┬───┘   └────┬─────┘   └─────────┘   └────┬────┘
     │            │                             │
     │       ┌────┴─────┐                  ┌────┴────┐
     │       │ Postgres │                  │  Redis  │
     │       │  :5432   │                  │  :6379  │
     │       └──────────┘                  └─────────┘
     │
   React SPA (static files)
```

| Service | Role |
|---------|------|
| **nginx** | Reverse proxy. Serves frontend, routes `/api/*` to API and `/ws/camera/*` to camera service |
| **api** | FastAPI. Auth, sessions, grading jobs, calibration, model management |
| **camera** | FastAPI + ZED SDK. MJPEG streaming, SVO2 recording, ChArUco calibration |
| **grader** | Redis worker. Reads SVO2 files, runs instrument detection, computes 3D poses and skill metrics |
| **frontend** | React 18 + Vite + Tailwind. Build-only container that produces static assets |
| **db** | PostgreSQL 15 |
| **redis** | Redis 7. Job queue for grading |

## Running on the Jetson (production)

Requires Jetson AGX Orin with JetPack 5.1.2+ and two ZED X cameras (GMSL2).

```bash
cp .env.example .env
# Edit .env: set ZED_SERIAL_ON_AXIS, ZED_SERIAL_OFF_AXIS, JWT_SECRET

docker compose build
docker compose up -d
```

Open `http://<jetson-ip>` in a browser.

## Running locally for development (macOS / Linux x86)

You can run the full stack on a Mac or Linux workstation for **playback, grading, and UI development**. Recording and live camera capture are disabled — the camera service runs in mock mode with test-pattern video.

### Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- ~2 GB free disk space for images

### Quick start

```bash
git clone <repo-url> && cd lap-trackr

# Create env file
cp .env.example .env

# Build and start with the dev compose override
docker compose -f docker-compose.yml -f docker-compose.dev.yml build
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Open [http://localhost](http://localhost).

### What works in dev mode

| Feature | Status |
|---------|--------|
| Frontend UI | Full |
| Auth (register/login) | Full |
| Session management | Full (mock recordings create placeholder SVO2 files) |
| MJPEG live view | Test-pattern streams for all 4 views |
| Grading | Runs with synthetic data (no real SVO2 decoding) |
| Calibration panel | UI works; ChArUco detection runs against test frames |
| Health dashboard | Full |
| Model management | Full |

### What does NOT work in dev mode

- **Real camera capture** — no ZED SDK, so streams show test patterns
- **SVO2 recording** — files are placeholder stubs
- **SVO2 playback in grader** — falls back to synthetic frames/depth automatically
- **GPU-accelerated detection** — grader dev image has no CUDA or PyTorch; uses the placeholder backend

### Convenience alias

Add to your shell profile:

```bash
alias dc-dev="docker compose -f docker-compose.yml -f docker-compose.dev.yml"
```

Then: `dc-dev build`, `dc-dev up -d`, `dc-dev logs -f api`, etc.

### Rebuilding the frontend

The frontend builds at image build time. After changing React/TS source:

```bash
dc-dev build frontend && dc-dev up -d frontend nginx
```

### Viewing logs

```bash
dc-dev logs -f api camera grader
```

### Resetting everything

```bash
dc-dev down -v   # stops containers and removes volumes (DB data lost)
```

## Project structure

```
lap-trackr/
├── docker-compose.yml          # Production compose
├── docker-compose.dev.yml      # Dev override (mock camera, no GPU)
├── .env.example                # Environment template
├── services/
│   ├── api/                    # FastAPI backend
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── core/           # config, auth, database
│   │   │   ├── models/         # SQLAlchemy models
│   │   │   ├── schemas/        # Pydantic schemas
│   │   │   └── routers/        # auth, sessions, results, calibration, models, health
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── camera/                 # Camera service
│   │   ├── app/
│   │   │   ├── main.py         # FastAPI app (auto-detects ZED SDK vs mock)
│   │   │   ├── camera_manager.py       # Real ZED camera control
│   │   │   ├── camera_manager_mock.py  # Mock for dev
│   │   │   ├── calibrator.py           # ChArUco board detection
│   │   │   └── config.py
│   │   ├── Dockerfile          # Production (ZED SDK base image)
│   │   └── Dockerfile.dev      # Dev (plain Python, no ZED)
│   ├── grader/                 # Grading worker
│   │   ├── app/
│   │   │   ├── worker.py       # Redis BRPOP loop
│   │   │   ├── pipeline.py     # Orchestrates load → detect → pose → metrics
│   │   │   ├── svo_loader.py   # ZED SVO2 reader (falls back to synthetic)
│   │   │   ├── pose_estimator.py
│   │   │   ├── metrics.py
│   │   │   ├── model_loader.py
│   │   │   ├── db.py
│   │   │   └── backends/       # yolo, cotracker, sam2, tapir, placeholder
│   │   ├── Dockerfile          # Production (ZED + PyTorch for Jetson)
│   │   └── Dockerfile.dev      # Dev (no ZED, no PyTorch)
│   ├── frontend/               # React SPA
│   │   ├── src/
│   │   │   ├── App.tsx
│   │   │   ├── api/client.ts
│   │   │   ├── pages/          # Dashboard, LiveView, Sessions, Health, Models
│   │   │   └── components/
│   │   ├── Dockerfile
│   │   └── package.json
│   └── nginx/
│       └── nginx.conf
└── data/                       # Runtime data (NVMe on Jetson, local volume in dev)
    ├── users/{user_id}/{session_timestamp}/
    │   ├── on_axis.svo2
    │   ├── off_axis.svo2
    │   ├── calibration_on_axis.json
    │   ├── session_metadata.json
    │   └── results/
    │       ├── metrics.json
    │       └── poses.json
    ├── calibration/default/    # Global default calibrations
    └── models/                 # ML model weights
```

## Calibration

The system uses a ChArUco board to calibrate camera extrinsics (camera-to-workspace transform) so that 3D measurements are in a consistent coordinate frame.

Board parameters (configured via `.env`):
- **9x14** checkerboard, **20mm** squares, **15mm** ArUco markers
- Dictionary: **DICT_5X5_50**

### Workflow

1. Mount cameras in final position
2. Print the ChArUco board at actual size (verify with a ruler)
3. In **Live View**, open the calibration panel
4. Capture 5+ frames with the board at varied positions and angles
5. Click **Compute** — check reprojection error is < 1.0 px
6. Repeat for the second camera

Calibration is saved as a global default and automatically copied into each new session directory.

### Generating the board image

```python
python3 -c "
import cv2
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
b = cv2.aruco.CharucoBoard((9, 14), 0.020, 0.015, d)
img = b.generateImage((900, 1400))
cv2.imwrite('charuco_9x14.png', img)
"
```

## Session data

Each session directory is **self-contained** — everything needed for offline replay:

| File | Contents |
|------|----------|
| `on_axis.svo2` | On-axis stereo recording |
| `off_axis.svo2` | Off-axis stereo recording |
| `calibration_on_axis.json` | Camera intrinsics + extrinsic transform + board config |
| `session_metadata.json` | Session info, camera serials, SDK version |
| `results/metrics.json` | Skill metrics (after grading) |
| `results/poses.json` | Frame-by-frame 3D instrument tip positions |
