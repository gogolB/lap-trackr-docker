from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine
from app.models.models import Base
from app.routers import auth, results, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
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


@app.get("/health")
async def health():
    return {"status": "ok"}
