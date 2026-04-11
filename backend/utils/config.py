"""
backend/utils/config.py
Centralised settings — loaded once from .env on startup.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API keys
    openai_api_key: str
    elevenlabs_api_key: str
    hf_token: str = ""

    # Storage
    storage_backend: str = "local"          # "local" | "s3"
    local_storage_path: str = "./storage"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "mnc-dubbing-videos"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # App
    app_env: str = "development"
    secret_key: str = "change-me"
    allowed_origins: str = "http://localhost:5173"

    # Pipeline
    default_whisper_model: str = "whisper-1"
    max_upload_size_mb: int = 500
    worker_concurrency: int = 2

    class Config:
        env_file = ".env"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()