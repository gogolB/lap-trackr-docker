# Environment Variables

All configuration is managed through environment variables, defined in the `.env` file at the project root and passed to services via `docker-compose.yml`.

## Quick Setup

```bash
cp .env.example .env
# Edit .env with your values
```

## Complete Reference

### Database

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `POSTGRES_USER` | `laptrackr` | Yes | PostgreSQL username |
| `POSTGRES_PASSWORD` | `changeme` | Yes | PostgreSQL password. **Change in production** |
| `POSTGRES_DB` | `laptrackr` | Yes | Database name |
| `DATABASE_URL` | `postgresql+asyncpg://laptrackr:changeme@db:5432/laptrackr` | Yes | Full connection string. Must use `asyncpg` driver (the grader automatically converts to `psycopg2` internally) |

### Authentication

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `JWT_SECRET` | `GENERATE_A_RANDOM_SECRET` | **Yes** | HMAC signing key. API **refuses to start** if left as default. Generate with: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `JWT_ALGORITHM` | `HS256` | No | JWT signing algorithm |
| `JWT_EXPIRY_MINUTES` | `1440` (24 hours) | No | Token lifetime in minutes |

### Camera

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `ZED_SERIAL_ON_AXIS` | *(empty)* | Production only | Serial number of the on-axis ZED X camera. Leave blank or set to `MOCK-*` for dev mode |
| `ZED_SERIAL_OFF_AXIS` | *(empty)* | Production only | Serial number of the off-axis ZED X camera |
| `CAMERA_HOST` | `0.0.0.0` | No | Camera service bind address |
| `CAMERA_PORT` | `8001` | No | Camera service port |

### Paths

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `DATA_DIR` | `/data` | No | Base directory for all runtime data (sessions, models, calibration) |
| `MODELS_DIR` | `/data/models` | No | ML model weights storage |
| `CALIBRATION_DIR` | `/data/calibration` | No | Calibration file storage |

### CORS

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `CORS_ORIGINS` | `*` | No | Comma-separated allowed origins. `*` allows all origins but disables credentials. Set explicit origins (e.g., `http://localhost,https://your-domain.com`) to enable credentials |

### Redis

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | No | Redis connection string |

### API

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `API_HOST` | `0.0.0.0` | No | API service bind address |
| `API_PORT` | `8000` | No | API service port |
| `CAMERA_SERVICE_URL` | `http://camera:8001` | No | Internal URL for camera service (set in docker-compose.yml, not .env) |

### Calibration Board

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `CHARUCO_ROWS` | `14` | No | Number of rows on the ChArUco board |
| `CHARUCO_COLS` | `9` | No | Number of columns on the ChArUco board |
| `CHARUCO_SQUARE_SIZE_MM` | `20.0` | No | Checkerboard square size in millimeters |
| `CHARUCO_MARKER_SIZE_MM` | `15.0` | No | ArUco marker size in millimeters |
| `CHARUCO_DICT` | `DICT_5X5_50` | No | ArUco dictionary name. Supported: `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`, `DICT_5X5_50`, `DICT_5X5_100`, `DICT_5X5_250`, `DICT_5X5_1000`, `DICT_6X6_50`, `DICT_6X6_100`, `DICT_6X6_250`, `DICT_7X7_50`, `DICT_7X7_100` |

### Grader Internals

These are set in `services/grader/app/config.py` and not typically overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_SAMPLE_INTERVAL` | `5` | Process every Nth frame from recordings |
| `CAMERA_FX` | `700.0` | Fallback focal length X (pixels) when no calibration file |
| `CAMERA_FY` | `700.0` | Fallback focal length Y (pixels) |
| `CAMERA_CX` | `640.0` | Fallback principal point X (pixels) |
| `CAMERA_CY` | `360.0` | Fallback principal point Y (pixels) |
| `DEFAULT_FPS` | `30.0` | Fallback frame rate when SVO2 metadata unavailable |

## Example `.env` for Production (Jetson)

```env
POSTGRES_USER=laptrackr
POSTGRES_PASSWORD=s3cur3-p4ssw0rd-h3r3
POSTGRES_DB=laptrackr
DATABASE_URL=postgresql+asyncpg://laptrackr:s3cur3-p4ssw0rd-h3r3@db:5432/laptrackr

JWT_SECRET=xK9mQ2vR7wY4tF6hN8jP0sL3aE5dC1bG
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=1440

ZED_SERIAL_ON_AXIS=51127155
ZED_SERIAL_OFF_AXIS=57422551

DATA_DIR=/data
MODELS_DIR=/data/models
CALIBRATION_DIR=/data/calibration
CORS_ORIGINS=*
REDIS_URL=redis://redis:6379/0

CHARUCO_ROWS=9
CHARUCO_COLS=14
CHARUCO_SQUARE_SIZE_MM=20.0
CHARUCO_MARKER_SIZE_MM=15.0
CHARUCO_DICT=DICT_5X5_100
```

## Example `.env` for Development

```env
POSTGRES_USER=laptrackr
POSTGRES_PASSWORD=changeme
POSTGRES_DB=laptrackr
DATABASE_URL=postgresql+asyncpg://laptrackr:changeme@db:5432/laptrackr

JWT_SECRET=dev-secret-not-for-production-use-only
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=1440

ZED_SERIAL_ON_AXIS=
ZED_SERIAL_OFF_AXIS=

DATA_DIR=/data
MODELS_DIR=/data/models
CALIBRATION_DIR=/data/calibration
CORS_ORIGINS=*
REDIS_URL=redis://redis:6379/0

CHARUCO_ROWS=14
CHARUCO_COLS=9
CHARUCO_SQUARE_SIZE_MM=20.0
CHARUCO_MARKER_SIZE_MM=15.0
CHARUCO_DICT=DICT_5X5_50
```
