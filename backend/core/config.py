"""
backend/core/config.py

Centralized settings, loaded from environment variables / .env.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg2://btip:btip@localhost:5432/btip"
    postgres_user: str = "btip"
    postgres_password: str = "btip"
    postgres_db: str = "btip"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Mapbox
    mapbox_token: str = ""

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"

    # Auth
    secret_key: str = "dev-only-change-me"

    # Ingestion paths
    raw_violations_csv: str = "data/raw/violations.csv"
    processed_dir: str = "data/processed"

    # Bengaluru bounding box used to drop bad lat/lng during ingestion
    bbox_lat_min: float = 12.8
    bbox_lat_max: float = 13.2
    bbox_lng_min: float = 77.4
    bbox_lng_max: float = 77.8


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
