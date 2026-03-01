from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@db:5432/laptrackr"
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60
    REDIS_URL: str = "redis://redis:6379/0"
    DATA_DIR: str = "/data"
    CAMERA_SERVICE_URL: str = "http://camera:8001"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
