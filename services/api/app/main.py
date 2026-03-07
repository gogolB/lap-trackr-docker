import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.core.database import async_session, engine
from app.model_registry import MODEL_CATALOG
from app.models.models import Base, MLModel, ModelStatus
from app.routers import auth, calibration, health, models, results, sessions

logger = logging.getLogger("api.startup")


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
    # Create all tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _seed_model_registry()
    yield
    await engine.dispose()


app = FastAPI(title="lap-trackr API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(results.router)
app.include_router(models.router)
app.include_router(calibration.router)
app.include_router(health.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
