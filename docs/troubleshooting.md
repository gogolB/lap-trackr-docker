# Troubleshooting

## Service Won't Start

### API: "JWT_SECRET is insecure" error

The API refuses to start if `JWT_SECRET` is left at a known default value.

**Fix:** Generate a real secret and set it in `.env`:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### API: Database connection refused

The API depends on PostgreSQL being healthy. Check:

```bash
docker compose ps db
docker compose logs db
```

Common causes:
- `/data/postgres` directory doesn't exist or has wrong permissions
- PostgreSQL data was corrupted (delete `/data/postgres` and restart to reinitialize)
- `POSTGRES_PASSWORD` in `.env` doesn't match what was used when the DB was first created

### Camera: "ZED SDK not available -- running in MOCK mode"

This is expected in dev mode. In production, ensure:
- The ZED SDK is installed on the host at `/usr/local/zed`
- The camera container is using the production `Dockerfile` (not `Dockerfile.dev`)
- The host volumes are mounted: `/usr/local/zed:/usr/local/zed`

### Camera: No cameras detected

```bash
# Check if cameras are physically connected
ls /dev/video*

# Check ZED diagnostic
docker compose exec camera python3 -c "import pyzed.sl; print(pyzed.sl.Camera.get_device_list())"

# Verify serial numbers in .env match actual cameras
curl http://localhost:8001/cameras
```

### Nginx: "502 Bad Gateway"

Nginx can't reach the upstream service. Check:

```bash
docker compose ps          # Are api/camera healthy?
docker compose logs api    # API startup errors?
docker compose logs camera # Camera startup errors?
```

### Grader/Exporter: Not processing jobs

```bash
# Check if worker is running
docker compose logs grader
docker compose logs exporter

# Check Redis queue length
docker compose exec redis redis-cli LLEN export_jobs
docker compose exec redis redis-cli LLEN grading_jobs
```

## Recording Issues

### Camera streams show black/frozen

- Check camera connection: `docker compose logs camera`
- Restart camera service: `docker compose restart camera`
- If using ZED cameras, check the grab thread error count in logs

### SVO2 files are 0 bytes

The grab thread may have failed to start. Check:

```bash
docker compose logs camera | grep -i error
```

Ensure the camera container has access to hardware devices (privileged mode, device mounts).

### Session stuck in "exporting"

The export worker may have crashed or the job was lost.

```bash
# Check exporter logs
docker compose logs exporter

# The API's periodic sweep will mark it as export_failed after 30 minutes.
# You can also re-export or retry manually:
curl -X POST http://localhost/api/sessions/{id}/re-export \
  -H "Authorization: Bearer $TOKEN"

curl -X POST http://localhost/api/sessions/{id}/retry \
  -H "Authorization: Bearer $TOKEN"
```

If the UI shows progress stuck on one camera, check `GET /api/sessions/{id}/progress` or the exporter logs. Export now tracks `export_on_axis`, `export_off_axis`, and `detect_tips` separately.

### Session stuck in "grading"

Same as above, but for the grading worker. Sessions stuck in `grading` for >30 minutes are automatically marked `failed`.

```bash
docker compose logs grader
```

## Grading Issues

### "No frames loaded" error

The grader couldn't load video data. In order of fallback:

1. SVO2 via ZED SDK (production only)
2. Exported MP4 + NPZ files
3. Synthetic data

If all three fail, check that:
- Export completed successfully (status was `completed` before grading)
- MP4/NPZ files exist in the session directory
- File permissions allow the grader container to read them

### Placeholder detections (random-looking results)

The PlaceholderBackend is being used. This means no real ML model is active.

**Fix:** Go to the Models page, download a model, and activate it.

### GPU out of memory

The offline grader may need multiple heavyweight passes, especially SAM2 and CoTracker3. Large segmentation models can consume significant GPU memory.

**Fix:** Use a smaller SAM2 checkpoint, reduce concurrent grading work, or move the offline grading job to a roomier GPU environment.

### Metrics are all zero or NaN

Depth data may be missing or all detections failed to back-project.

- Check if `{camera}_depth.npz` exists and has valid data
- Check if calibration files are present in the session directory
- Review `poses.json` -- if most entries are `null`, detection quality was poor

## Database Issues

### Reset the database

```bash
# WARNING: This deletes all data
docker compose down
sudo rm -rf /data/postgres
docker compose up -d
```

### Run migrations manually

```bash
docker compose exec api sh -lc 'cd /app && alembic upgrade head'
```

Normally this is not required. The API runs Alembic automatically on startup. If you added a migration or repaired schema drift, restarting the API is usually enough:

```bash
docker compose restart api
```

### Check the current migration revision

```bash
docker compose exec api sh -lc 'cd /app && alembic current && printf "\\n---\\n" && alembic heads'
```

### Enum value missing after a deploy

Symptoms include errors like:

```text
invalid input value for enum sessionstatus: "awaiting_init"
```

That means the live DB schema drifted from the expected migration history.

Checks:

```bash
docker compose exec db psql -U laptrackr -d laptrackr -c "SELECT version_num FROM alembic_version;"
docker compose exec db psql -U laptrackr -d laptrackr -c "
  SELECT e.enumlabel
  FROM pg_type t
  JOIN pg_enum e ON t.oid = e.enumtypid
  WHERE t.typname = 'sessionstatus'
  ORDER BY e.enumsortorder;
"
```

Fix:

1. ensure the latest migration files are present in the API image
2. restart the API or run `alembic upgrade head`
3. if the DB revision says it is current but the schema is missing pieces, add a new forward repair migration instead of editing old revisions

### Check database size

```bash
docker compose exec db psql -U laptrackr -c "
  SELECT pg_size_pretty(pg_database_size('laptrackr'));
"
```

## Disk Space Issues

### Check disk usage

```bash
# Overall
df -h /data

# Docker images and containers
docker system df

# Session data
du -sh /data/users/* | sort -rh | head -20

# Largest files
find /data/users -type f -size +100M -exec ls -lh {} \; | sort -k5 -rh | head -20
```

### Free up space

```bash
# Remove unused Docker images
docker system prune -f

# Remove old sessions via the UI (or API)
# Sessions include SVO2 files which can be 1-10 GB each

# Clean Docker build cache
docker builder prune -f
```

## Network Issues

### Can't access the UI from another machine

Check if nginx is listening and the firewall allows it:

```bash
# Is nginx running?
docker compose ps nginx

# Is port 80 open?
sudo ss -tlnp | grep :80

# Test from the Jetson itself
curl http://localhost

# Check firewall
sudo ufw status
```

### CORS errors in browser console

Set `CORS_ORIGINS` in `.env` to include your frontend origin:

```env
CORS_ORIGINS=http://192.168.1.100,http://jetson-hostname
```

Restart the API: `docker compose restart api`

## Performance Issues

### Slow export

- Check if hardware NVENC is available: `ls /dev/nvhost-msenc`
- Ensure the exporter container has device mounts for NVENC
- Software encoding is ~5-10x slower than hardware

### Slow grading

- Check GPU utilization: `tegrastats` (on Jetson) or `nvidia-smi`
- Consider using `FRAME_SAMPLE_INTERVAL` > 5 to process fewer frames
- Use a lighter model (YOLOv8n vs SAM2 large)

### Slow MJPEG streams

- Each stream holds an HTTP connection open
- The dedicated port 8081 prevents stream connections from blocking API requests
- If streams are laggy, check CPU/GPU load and camera grab thread errors

## Development Issues

### Frontend changes not appearing

The frontend is built at Docker image build time. After changes:

```bash
dc-dev build frontend && dc-dev up -d frontend nginx
```

For faster iteration, use [frontend hot reload](setup-dev.md#frontend-development-with-hot-reload).

### API changes not taking effect

Rebuild and restart:

```bash
dc-dev build api && dc-dev up -d api
```

### Docker build fails on ARM Mac (M1/M2/M3)

Some images may not have ARM builds. If you see platform warnings:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
dc-dev build
```

This runs containers under emulation (slower but functional).

## Log Locations

| Service | Command |
|---------|---------|
| API | `docker compose logs -f api` |
| Camera | `docker compose logs -f camera` |
| Grader | `docker compose logs -f grader` |
| Exporter | `docker compose logs -f exporter` |
| Nginx | `docker compose logs -f nginx` |
| PostgreSQL | `docker compose logs -f db` |
| Redis | `docker compose logs -f redis` |
| All | `docker compose logs -f` |

### Filtering logs

```bash
# Last 100 lines
docker compose logs --tail 100 api

# Since a specific time
docker compose logs --since 2026-03-08T10:00:00 api

# Errors only
docker compose logs api 2>&1 | grep -i error
```
