import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, select, text, update

from app.core.config import settings
from app.core.database import async_session, engine
from app.model_registry import MODEL_CATALOG
from app.models.models import Base, MLModel, ModelStatus, Session, SessionStatus
from app.routers import auth, calibration, camera_config, health, models, results, sessions, tip_init

logger = logging.getLogger("api.startup")

STALE_TIMEOUT_MINUTES = 30
SWEEP_INTERVAL_SECONDS = 300  # 5 minutes
CAMERA_CONFIG_APPLY_RETRY_INTERVAL_SECONDS = 5
CAMERA_CONFIG_APPLY_MAX_ATTEMPTS = 24
API_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = API_ROOT / "alembic.ini"
ALEMBIC_MIGRATIONS = API_ROOT / "migrations"

_INSECURE_JWT_SECRETS = {"change-me-in-production", "change-this-to-a-random-secret", "GENERATE_A_RANDOM_SECRET"}


async def _seed_model_registry() -> None:
    """Insert catalog models, update metadata, and prune stale built-ins."""
    catalog_by_slug = {entry["slug"]: entry for entry in MODEL_CATALOG}

    async with async_session() as db:
        active_result = await db.execute(
            select(MLModel).where(MLModel.is_active == True)  # noqa: E712
        )
        active_model_types = {
            model.model_type
            for model in active_result.scalars().all()
        }

        result = await db.execute(select(MLModel).where(MLModel.is_custom == False))  # noqa: E712
        existing_models = {model.slug: model for model in result.scalars()}

        removed = 0
        for slug, model in list(existing_models.items()):
            if slug in catalog_by_slug:
                continue
            _delete_catalog_model_files(model.file_path)
            await db.delete(model)
            existing_models.pop(slug, None)
            removed += 1

        for slug, entry in catalog_by_slug.items():
            local_file = _resolve_catalog_local_file(entry)
            managed_file_path = str(local_file) if local_file is not None else None
            managed_file_size = (
                int(local_file.stat().st_size)
                if local_file is not None
                else entry.get("file_size_bytes")
            )
            managed_status = (
                ModelStatus.ready
                if local_file is not None
                else ModelStatus.available
            )
            model = existing_models.get(slug)
            if model is None:
                model = MLModel(
                    slug=slug,
                    name=entry["name"],
                    model_type=entry["model_type"],
                    description=entry.get("description"),
                    version=entry.get("version"),
                    download_url=entry.get("download_url"),
                    file_size_bytes=managed_file_size,
                    file_path=managed_file_path,
                    status=managed_status,
                )
                db.add(model)
                existing_models[slug] = model
            else:
                model.name = entry["name"]
                model.model_type = entry["model_type"]
                model.description = entry.get("description")
                model.version = entry.get("version")
                model.download_url = entry.get("download_url")

                if local_file is not None:
                    model.file_path = managed_file_path
                    model.file_size_bytes = managed_file_size
                    if model.status != ModelStatus.active:
                        model.status = ModelStatus.ready
                elif entry.get("local_path") and model.status != ModelStatus.active:
                    model.file_path = None
                    model.file_size_bytes = managed_file_size
                    model.status = ModelStatus.available

            if (
                model.model_type not in active_model_types
                and entry.get("auto_activate")
                and model.status in {ModelStatus.ready, ModelStatus.custom}
            ):
                model.is_active = True
                model.status = ModelStatus.active if not model.is_custom else ModelStatus.custom
                active_model_types.add(model.model_type)

        await db.commit()
    logger.info(
        "Model registry reconciled with %d catalog entries (removed %d stale built-ins)",
        len(MODEL_CATALOG),
        removed,
    )


def _delete_catalog_model_files(file_path: str | None) -> None:
    if not file_path:
        return

    models_root = Path(settings.MODELS_DIR).resolve()
    path = Path(file_path)
    resolved = path.resolve(strict=False)
    if models_root not in resolved.parents:
        logger.warning("Skipping cleanup outside models dir: %s", resolved)
        return

    target = resolved.parent if resolved.suffix else resolved
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def _resolve_catalog_local_file(entry: dict) -> Path | None:
    local_path = entry.get("local_path")
    if not local_path:
        return None

    models_root = Path(settings.MODELS_DIR).resolve()
    resolved = (models_root / local_path).resolve(strict=False)
    if models_root not in resolved.parents:
        logger.warning("Ignoring catalog local path outside models dir: %s", resolved)
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


async def _run_alembic(command_name: str, revision: str = "head") -> None:
    """Run Alembic commands in a worker thread during startup."""

    def _invoke() -> None:
        cfg = Config(str(ALEMBIC_INI))
        cfg.set_main_option("script_location", str(ALEMBIC_MIGRATIONS))
        cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
        if command_name == "upgrade":
            command.upgrade(cfg, revision)
        elif command_name == "stamp":
            command.stamp(cfg, revision)
        else:
            raise ValueError(f"Unsupported Alembic command: {command_name}")

    await asyncio.to_thread(_invoke)


async def _bootstrap_legacy_schema() -> None:
    """Bring unmanaged databases up to the current schema, then stamp them."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        await conn.execute(text(
            "ALTER TYPE sessionstatus ADD VALUE IF NOT EXISTS 'exporting' AFTER 'completed'"
        ))
        await conn.execute(text(
            "ALTER TYPE sessionstatus ADD VALUE IF NOT EXISTS 'export_failed' AFTER 'exporting'"
        ))
        await conn.execute(text(
            "ALTER TYPE sessionstatus ADD VALUE IF NOT EXISTS 'awaiting_init' AFTER 'export_failed'"
        ))

        result = await conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'sessions' AND column_name = 'name'"
        ))
        if result.first() is None:
            await conn.execute(text(
                "ALTER TABLE sessions ADD COLUMN name VARCHAR(255) NOT NULL DEFAULT 'Untitled Session'"
            ))
            await conn.execute(text(
                "ALTER TABLE sessions ALTER COLUMN name DROP DEFAULT"
            ))

        result = await conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'grading_results' AND column_name = 'warnings'"
        ))
        if result.first() is None:
            await conn.execute(text(
                "ALTER TABLE grading_results ADD COLUMN warnings JSONB"
            ))


async def _initialize_schema() -> None:
    """Apply Alembic migrations automatically, bootstrapping if needed."""
    async with engine.begin() as conn:
        table_names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))

    if "alembic_version" in table_names:
        logger.info("Applying Alembic migrations")
        await _run_alembic("upgrade", "head")
        return

    if table_names:
        logger.warning(
            "Database has tables but no alembic_version; bootstrapping current schema and stamping head"
        )
    else:
        logger.info("Bootstrapping fresh database schema and stamping Alembic head")

    await _bootstrap_legacy_schema()
    await _run_alembic("stamp", "head")


async def _sweep_stale_sessions() -> None:
    """Reset sessions stuck in exporting/grading for too long."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_TIMEOUT_MINUTES)
    async with async_session() as db:
        # Stuck exporting -> export_failed
        result = await db.execute(
            update(Session)
            .where(
                Session.status == SessionStatus.exporting,
                Session.stopped_at.isnot(None),
                Session.stopped_at < cutoff,
            )
            .values(status=SessionStatus.export_failed)
            .returning(Session.id)
        )
        stuck_export = result.scalars().all()
        for sid in stuck_export:
            logger.warning("Stale session %s: exporting -> export_failed", sid)

        # Stuck grading -> failed
        result = await db.execute(
            update(Session)
            .where(
                Session.status == SessionStatus.grading,
                Session.stopped_at.isnot(None),
                Session.stopped_at < cutoff,
            )
            .values(status=SessionStatus.failed)
            .returning(Session.id)
        )
        stuck_grading = result.scalars().all()
        for sid in stuck_grading:
            logger.warning("Stale session %s: grading -> failed", sid)

        await db.commit()

    total = len(stuck_export) + len(stuck_grading)
    if total:
        logger.info("Stale session sweep: recovered %d sessions", total)


async def _periodic_sweep() -> None:
    """Run stale session sweep periodically."""
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            await _sweep_stale_sessions()
        except Exception:
            logger.exception("Error during stale session sweep")


async def _apply_saved_camera_config_on_startup() -> None:
    """Best-effort startup sync so the camera service uses persisted flips/rotation."""
    for attempt in range(1, CAMERA_CONFIG_APPLY_MAX_ATTEMPTS + 1):
        try:
            async with async_session() as db:
                await camera_config.push_saved_camera_config(db)
            logger.info("Applied saved camera config to camera service")
            return
        except Exception as exc:
            logger.warning(
                "Startup camera config apply attempt %d/%d failed: %s",
                attempt,
                CAMERA_CONFIG_APPLY_MAX_ATTEMPTS,
                exc,
            )
            if attempt == CAMERA_CONFIG_APPLY_MAX_ATTEMPTS:
                logger.error("Giving up on startup camera config apply")
                return
            await asyncio.sleep(CAMERA_CONFIG_APPLY_RETRY_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.JWT_SECRET in _INSECURE_JWT_SECRETS:
        logger.critical(
            "JWT_SECRET is set to an insecure default. "
            "Generate a secret with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
        raise SystemExit(1)
    await _initialize_schema()
    await _seed_model_registry()
    await _sweep_stale_sessions()
    sweep_task = asyncio.create_task(_periodic_sweep())
    camera_config_task = asyncio.create_task(_apply_saved_camera_config_on_startup())
    yield
    camera_config_task.cancel()
    try:
        await camera_config_task
    except asyncio.CancelledError:
        pass
    sweep_task.cancel()
    try:
        await sweep_task
    except asyncio.CancelledError:
        pass
    await engine.dispose()


app = FastAPI(title="lap-trackr API", version="1.0.0", lifespan=lifespan)

_cors_origins_raw = settings.CORS_ORIGINS.strip()
if _cors_origins_raw == "*":
    _cors_origins: list[str] = ["*"]
    _cors_credentials = False
else:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(results.router)
app.include_router(models.router)
app.include_router(calibration.router)
app.include_router(camera_config.router)
app.include_router(tip_init.router)
app.include_router(health.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
