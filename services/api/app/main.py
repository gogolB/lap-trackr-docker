import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text

from app.core.config import settings
from app.core.database import async_session, engine
from app.model_registry import MODEL_CATALOG
from app.models.models import Base, MLModel, ModelStatus
from app.routers import auth, calibration, camera_config, health, models, results, sessions, tip_init

logger = logging.getLogger("api.startup")

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.JWT_SECRET in _INSECURE_JWT_SECRETS:
        logger.critical(
            "JWT_SECRET is set to an insecure default. "
            "Generate a secret with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
        raise SystemExit(1)
    # Create all tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Lightweight migrations for new columns on existing tables
    async with engine.begin() as conn:
        # Add sessions.name if it doesn't exist yet
        result = await conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'sessions' AND column_name = 'name'"
        ))
        if result.first() is None:
            await conn.execute(text(
                "ALTER TABLE sessions ADD COLUMN name VARCHAR(255) NOT NULL DEFAULT 'Untitled Session'"
            ))
            logger.info("Migration: added 'name' column to sessions table")
    await _seed_model_registry()
    yield
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
