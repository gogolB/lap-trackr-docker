import asyncio
import logging
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
API_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = API_ROOT / "alembic.ini"
ALEMBIC_MIGRATIONS = API_ROOT / "migrations"

_INSECURE_JWT_SECRETS = {"change-me-in-production", "change-this-to-a-random-secret", "GENERATE_A_RANDOM_SECRET"}


async def _seed_model_registry() -> None:
    """Insert catalog models that don't already exist in the database."""
    async with async_session() as db:
        for entry in MODEL_CATALOG:
            result = await db.execute(
                select(MLModel).where(MLModel.slug == entry["slug"])
            )
            if result.scalar_one_or_none() is None:
                db.add(MLModel(
                    slug=entry["slug"],
                    name=entry["name"],
                    model_type=entry["model_type"],
                    description=entry.get("description"),
                    version=entry.get("version"),
                    download_url=entry.get("download_url"),
                    file_size_bytes=entry.get("file_size_bytes"),
                    status=ModelStatus.available,
                ))
        await db.commit()
    logger.info("Model registry seeded with %d catalog entries", len(MODEL_CATALOG))


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
    yield
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
