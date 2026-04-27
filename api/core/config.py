"""Application configuration loaded from environment variables.

All runtime configuration lives here — no hardcoded values anywhere else
in the codebase. Settings are loaded from environment variables, with
sensible defaults for local development.

In production (Docker / cloud): set these as real env vars.
In development: create a .env file in the project root (gitignored).

Why pydantic-settings:
  - Type-validated at startup — a misconfigured deployment fails fast
    with a clear error, not silently with wrong behaviour.
  - All config in one place — no scattered os.getenv() calls.
  - Automatically reads from .env files in development.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform runtime settings.

    All values can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application ----
    app_name: str = "Medical AI Clinical Platform"
    app_version: str = "0.1.0"
    debug: bool = False

    # ---- FHIR server ----
    fhir_base_url: str = "http://hapi.fhir.org/baseR4"
    fhir_timeout_seconds: float = 30.0

    # ---- Audit log ----
    audit_db_path: str = "audit_log.db"

    # ---- ML model ----
    model_weights_key: str = "densenet121-res224-all"
    confidence_threshold: float = 0.30

    # ---- Security (JWT) ----
    # In production: set SECRET_KEY to a long random string via env var.
    # Never commit a real secret key.
    secret_key: str = Field(
        default="dev-insecure-change-in-production-min-32-chars!!",
        description="JWT signing secret. Override via SECRET_KEY env var.",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ---- CORS ----
    allowed_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Using lru_cache means Settings is only instantiated once per process.
    FastAPI's Depends(get_settings) uses this cached instance for every
    request — no repeated env-var reads or .env file parsing.
    """
    return Settings()
