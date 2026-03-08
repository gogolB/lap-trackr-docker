# Database & Migrations

This document covers the persistent state of Lap-Trackr: what is stored in PostgreSQL, what is stored only in Redis, and how schema changes are applied.

For the field-by-field schema, see [Data Model](data-model.md).

## Storage Responsibilities

| Store | Durability | Used For |
|-------|------------|----------|
| PostgreSQL (`db`) | Durable | Users, sessions, grading results, calibrations, model catalog, camera config |
| Redis (`redis`) | Ephemeral | Export and grading queues, per-job progress, export cancel flags, model download progress |
| Session directory on disk (`/data/users/...`) | Durable | SVO2 files, exported MP4/NPZ files, sample frames, calibration JSON, tip initialization data, grading artifacts |

## PostgreSQL Ownership

The API service owns the database schema.

- ORM models live in [services/api/app/models/models.py](/home/enmed/lap-trackr/services/api/app/models/models.py)
- Alembic revisions live in `services/api/migrations/versions/`
- The grader writes rows through its own sync DB helper, but it must match the API schema exactly

Core durable entities:

| Table | Purpose |
|-------|---------|
| `users` | Local user accounts |
| `sessions` | Recording lifecycle and file locations |
| `grading_results` | Final metrics, errors, and warnings per session |
| `calibrations` | Per-camera intrinsic/extrinsic calibration records |
| `ml_models` | Catalog and custom model metadata |
| `camera_config` | Single-row active camera config pushed to the camera service |

## Session Lifecycle in the Database

`sessions.status` is the main durable job-state field.

| Status | Set By | Meaning |
|--------|--------|---------|
| `recording` | API | Camera service is actively recording SVO2 files |
| `exporting` | API / exporter | Export job is converting SVO2 to MP4/NPZ and generating sample detections |
| `export_failed` | exporter / stale-session sweep | Export failed or stalled long enough to be marked failed |
| `awaiting_init` | exporter | Export finished; user must confirm tip positions |
| `completed` | tip-init API | Tip initialization is saved and the session is ready to grade |
| `grading` | API / grader | Grading job is in progress |
| `graded` | grader | Final metrics were written successfully |
| `failed` | API / grader / stale-session sweep | Grading failed or recording startup failed |

Important transitions:

- `recording -> exporting`: `POST /sessions/{id}/stop`
- `exporting -> awaiting_init`: export completed and no `tip_init.json` exists yet
- `exporting -> completed`: re-export completed and `tip_init.json` already exists
- `exporting -> graded`: re-export completed for a session that was already graded
- `awaiting_init -> completed`: `PUT /sessions/{id}/tip-init`
- `completed -> grading`: `POST /sessions/{id}/grade`
- `grading -> graded`: grading pipeline finished successfully
- `export_failed -> exporting`: `POST /sessions/{id}/retry` or `POST /sessions/{id}/re-export`
- `failed -> grading`: `POST /sessions/{id}/retry`

## Redis Keys and Queues

Redis does not hold authoritative session state. It is used for worker coordination and progress reporting only.

### Queues

| Key | Type | Producer | Consumer |
|-----|------|----------|----------|
| `export_jobs` | list | API | exporter worker |
| `grading_jobs` | list | API | grader worker |

Jobs are JSON payloads. The session status in PostgreSQL is still the durable source of truth.

### Progress Keys

| Key Pattern | Type | Writer | Notes |
|-------------|------|--------|-------|
| `job_progress:{session_id}` | hash | exporter / grader | Overall stage, current, total, percent, detail, timestamps, and per-stage JSON blobs |
| `export_cancel:{session_id}` | string | API | Signals the exporter to cancel an in-flight re-export |
| `model_download:{model_id}` | hash | API | Tracks model download progress |

`job_progress:{session_id}` is intentionally short-lived. Workers set a TTL so stale progress disappears after completion or abandonment.

## Migration Strategy

Lap-Trackr now applies Alembic migrations automatically when the API starts.

### Startup behavior

On API startup:

1. If `alembic_version` already exists, the API runs `alembic upgrade head`
2. If the database exists without Alembic metadata, the API bootstraps the current schema and stamps the DB to head
3. On a fresh database, the API bootstraps the schema and stamps the current head revision

This is implemented in [services/api/app/main.py](/home/enmed/lap-trackr/services/api/app/main.py).

### Rules for schema changes

- Every schema change must have an Alembic revision
- Do not rely on ad hoc `ALTER TABLE` logic in service code for new changes
- Do not edit or delete applied revision files
- If the grader needs a new field, add it through the API migration chain first

### Common commands

```bash
# Show the current DB revision from inside the API container
docker compose exec api sh -lc 'cd /app && alembic current'

# Show the repository head revision
docker compose exec api sh -lc 'cd /app && alembic heads'

# Generate a new revision after editing models
docker compose exec api sh -lc 'cd /app && alembic revision --autogenerate -m "describe change"'

# Apply pending revisions manually
docker compose exec api sh -lc 'cd /app && alembic upgrade head'
```

### Drift recovery

If the DB schema and `alembic_version` drift apart:

1. inspect the live schema in PostgreSQL
2. add a forward-only repair migration if needed
3. restart the API or run `alembic upgrade head`

Do not hand-edit `alembic_version` unless you are intentionally stamping a known-good schema.

## Backup and Restore

### Backup PostgreSQL

```bash
docker compose exec db pg_dump -U laptrackr -d laptrackr > laptrackr-$(date +%F).sql
```

### Restore PostgreSQL

```bash
cat laptrackr-2026-03-08.sql | docker compose exec -T db psql -U laptrackr -d laptrackr
```

### Backup session artifacts

PostgreSQL does not contain exported videos, depth files, or grading artifacts. Back up `/data/users`, `/data/calibration`, and `/data/models` alongside the database dump if you need a restorable system image.

## Useful Inspection Queries

```bash
# Current Alembic revision
docker compose exec db psql -U laptrackr -d laptrackr -c "SELECT version_num FROM alembic_version;"

# Session status enum values
docker compose exec db psql -U laptrackr -d laptrackr -c "
  SELECT e.enumlabel
  FROM pg_type t
  JOIN pg_enum e ON t.oid = e.enumtypid
  WHERE t.typname = 'sessionstatus'
  ORDER BY e.enumsortorder;
"

# Table sizes
docker compose exec db psql -U laptrackr -d laptrackr -c "
  SELECT relname AS table_name,
         pg_size_pretty(pg_total_relation_size(relid)) AS total_size
  FROM pg_catalog.pg_statio_user_tables
  ORDER BY pg_total_relation_size(relid) DESC;
"
```
