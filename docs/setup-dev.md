# Development Setup

Run the full Lap-Trackr stack on your local machine for UI development, API work, and testing -- no ZED cameras or NVIDIA GPU required.

## What Works in Dev Mode

| Feature | Status |
|---------|--------|
| Frontend UI (all pages) | Full |
| Auth (register/login) | Full |
| Session management (start/stop/delete) | Full (mock recordings create placeholder SVO2 files) |
| MJPEG live view | Test-pattern streams for all 4 camera views |
| Export pipeline | Runs (creates placeholder MP4/NPZ files) |
| Grading pipeline | Runs with synthetic data (placeholder ML backend) |
| Calibration UI | UI works; ChArUco detection runs against test frames |
| Health dashboard | Full |
| Model management (download/upload/activate) | Full |
| Camera config | Full |

## What Does NOT Work in Dev Mode

- **Real camera capture** -- no ZED SDK, streams show animated test patterns
- **Real SVO2 recording/playback** -- files are placeholder stubs
- **GPU-accelerated ML detection** -- grader dev image has no CUDA or PyTorch; uses the placeholder backend which generates synthetic detections
- **Hardware video encoding** -- falls back to software (OpenCV) encoding

## Prerequisites

### All Platforms

- **Docker Desktop** (Windows/macOS) or **Docker Engine + Compose v2** (Linux)
- **Git**
- ~2 GB free disk space for Docker images
- ~200 MB for the `/data` directory

### Platform-Specific Notes

#### Windows

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. In Docker Desktop settings:
   - Enable WSL 2 backend (recommended over Hyper-V)
   - Allocate at least 4 GB RAM to WSL
3. Clone and run from a WSL 2 terminal for best performance (avoid `/mnt/c/` paths)
4. Create the data directory: `sudo mkdir -p /data && sudo chown $USER:$USER /data`

#### macOS

1. Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Allocate at least 4 GB RAM in Docker Desktop preferences
3. Create the data directory: `sudo mkdir -p /data && sudo chown $USER /data`

#### Linux (with NVIDIA GPU)

If you have an NVIDIA GPU and want to test GPU-accelerated ML backends locally:

1. Install [Docker Engine](https://docs.docker.com/engine/install/)
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. You can optionally use the production Dockerfiles instead of the dev overrides, but you'll still need to mock the ZED SDK (no GMSL2 cameras on desktop GPUs). See [Hybrid GPU Setup](#hybrid-gpu-setup) below.

#### Linux (without NVIDIA GPU)

1. Install [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose v2](https://docs.docker.com/compose/install/)
2. Create the data directory: `sudo mkdir -p /data && sudo chown $USER:$USER /data`

## Quick Start

```bash
# Clone the repository
git clone <repo-url> && cd lap-trackr

# Create environment file
cp .env.example .env

# (Optional) Generate a real JWT secret -- the dev stack will work with the
# default but will print a warning at startup.
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Paste the output as JWT_SECRET in .env

# Build and start with the dev compose override
docker compose -f docker-compose.yml -f docker-compose.dev.yml build
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Open [http://localhost](http://localhost) in your browser.

## Convenience Alias

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, or PowerShell `$PROFILE`):

```bash
alias dc-dev="docker compose -f docker-compose.yml -f docker-compose.dev.yml"
```

Then:

```bash
dc-dev build          # Build all images
dc-dev up -d          # Start all services
dc-dev logs -f api    # Follow API logs
dc-dev down           # Stop all services
dc-dev down -v        # Stop and delete all data (DB included)
```

## What the Dev Override Changes

The `docker-compose.dev.yml` file overrides production settings:

| Service | Change |
|---------|--------|
| **camera** | Uses `Dockerfile.dev` (Python slim, no ZED SDK). Mock serials `MOCK-00001`/`MOCK-00002`. No privileged mode, no GPU runtime, no device mounts |
| **grader** | Uses `Dockerfile.dev` (no ZED SDK, no PyTorch, no ultralytics). Falls back to synthetic data and placeholder backend |
| **exporter** | Uses `Dockerfile.dev`. Same fallbacks as grader |
| **nginx** | Relaxed health check dependency on camera (started vs healthy) |

Services that are **unchanged** in dev mode: api, db, redis, frontend.

## Rebuilding After Code Changes

### Frontend changes (React/TypeScript)

```bash
dc-dev build frontend && dc-dev up -d frontend nginx
```

The frontend is a build-only container -- it compiles the React app and writes static files to a shared volume that nginx serves. You must rebuild after any frontend source change.

### API changes (Python)

```bash
dc-dev build api && dc-dev up -d api
```

### Camera service changes

```bash
dc-dev build camera && dc-dev up -d camera
```

### Grader/exporter changes

Both use the same Docker image:

```bash
dc-dev build grader && dc-dev up -d grader exporter
```

## Viewing Logs

```bash
# All services
dc-dev logs -f

# Specific services
dc-dev logs -f api camera
dc-dev logs -f grader exporter
```

## Database Access

```bash
# Connect to PostgreSQL directly
dc-dev exec db psql -U laptrackr laptrackr

# Run a query
dc-dev exec db psql -U laptrackr laptrackr -c "SELECT id, username FROM users;"
```

## Resetting State

```bash
# Stop containers and remove volumes (database, redis, frontend dist)
dc-dev down -v

# Also clear session/model data (if using /data on host)
sudo rm -rf /data/users /data/models /data/calibration
sudo mkdir -p /data/users /data/models /data/calibration
```

## Hybrid GPU Setup

If you have a Linux workstation with an NVIDIA GPU and want GPU-accelerated ML inference (YOLO, CoTracker, etc.) but no ZED cameras:

1. Install Docker + NVIDIA Container Toolkit (see Linux GPU prerequisites above)

2. Create a `docker-compose.override.yml` in the project root:

```yaml
services:
  # Keep camera in mock mode (no ZED cameras on desktop)
  camera:
    build:
      context: ./services/camera
      dockerfile: Dockerfile.dev
    environment:
      ZED_SERIAL_ON_AXIS: MOCK-00001
      ZED_SERIAL_OFF_AXIS: MOCK-00002

  # Use production grader image with GPU support
  grader:
    build:
      context: ./services/grader
      dockerfile: Dockerfile
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: all

  exporter:
    build:
      context: ./services/grader
      dockerfile: Dockerfile.dev
```

3. Then run with just the base compose file (the override is auto-loaded):

```bash
docker compose build
docker compose up -d
```

This gives you:
- Mock camera streams (no ZED SDK needed)
- Real ML inference on your GPU (SAM2, CoTracker3, and optional YOLO support)
- Software video encoding (no NVENC on desktop GPUs)
- Synthetic SVO2 data (grader generates fake frames/depth for testing)

## Offline Grading (No Docker / No Jetson)

Process exported session videos on any workstation with a GPU — no Redis, PostgreSQL, Docker, or ZED SDK required.

### Prerequisites

- Python 3.10+
- PyTorch with CUDA (Linux/Windows) or MPS (macOS)
- SAM2 and CoTracker installed from source

### Install

```bash
cd services/grader

# Install PyTorch first (pick one):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
pip install torch torchvision  # macOS MPS

# Install grader dependencies
pip install -r requirements.offline.txt

# Install ML model packages
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/facebookresearch/co-tracker.git
```

### Download Models

Download the SAM2 checkpoint (`sam2.1_hiera_large.pt`) from the [SAM2 releases](https://github.com/facebookresearch/sam2/releases).

Download the CoTracker checkpoint from the [CoTracker releases](https://github.com/facebookresearch/co-tracker/releases).

### Copy Session Data from Jetson

From the Jetson, copy the session directory to your workstation:

```bash
scp -r enmed@jetson:/data/users/<user_id>/<session_dir> ./my_session
```

The directory should contain:
- `on_axis_left.mp4`, `off_axis_left.mp4` (exported video)
- `on_axis_depth.npz`, `off_axis_depth.npz` (depth maps)
- `tip_init.json` or `tip_detections.json` (instrument tip positions)

### Run

```bash
cd services/grader

python -m app.grade_offline ./my_session \
    --sam2-model /path/to/sam2.1_hiera_large.pt \
    --cotracker-model /path/to/cotracker_v3.pth \
    --device cuda
```

Options:
- `--device cuda|mps|cpu` — compute device (auto-detected if omitted)
- `--sample-interval N` — process every Nth frame (default: 1 = all frames)
- `--sam2-config NAME` — SAM2 Hydra config name (default: `configs/sam2.1/sam2.1_hiera_l.yaml`)

Results are written to `<session_dir>/results/`:
- `metrics.json`, `poses.json`, `timings.json`
- `tracking_on_axis.mp4`, `tracking_off_axis.mp4`
- `detections_on_axis.csv`, `detections_off_axis.csv`
- `tracked_positions_world.csv`

## Frontend Development with Hot Reload

For faster frontend iteration, run Vite's dev server locally instead of rebuilding the Docker image:

### Prerequisites

- Node.js 18+ and npm

### Steps

```bash
cd services/frontend
npm install
npm run dev
```

Vite dev server starts on `http://localhost:5173` with hot module replacement. The `vite.config.ts` proxies:
- `/api/*` to `http://localhost:8000` (API service)
- `/ws/camera/*` to `http://localhost:8001` (camera service)

You still need the backend services running in Docker:

```bash
# In the project root
dc-dev up -d db redis api camera grader exporter
```

Then open `http://localhost:5173` instead of `http://localhost`.

## Troubleshooting Dev Setup

### "Cannot connect to the Docker daemon"

Docker Desktop isn't running. Start it from your system tray (Windows/macOS) or run `sudo systemctl start docker` (Linux).

### Port 80 already in use

Another service (Apache, IIS, etc.) is using port 80. Either stop it or change the nginx port in `docker-compose.yml`:

```yaml
nginx:
  ports:
    - "8080:80"
    - "8081:8081"
```

Then access the app at `http://localhost:8080`.

### API refuses to start (JWT_SECRET error)

Generate a real secret and set it in `.env`:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### `/data` directory permission denied

```bash
sudo mkdir -p /data
sudo chown -R $USER:$USER /data
```

On Windows (WSL 2), run these commands inside the WSL terminal, not PowerShell.

### Docker build fails with "no space left on device"

```bash
docker system prune -f
docker builder prune -f
```

### Camera streams show "connection refused"

Wait 10-15 seconds after `docker compose up`. The camera service takes a moment to initialize. Check logs: `dc-dev logs camera`.
