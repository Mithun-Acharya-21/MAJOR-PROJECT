"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Privacy-preserving defaults are intentional."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "EmotiSense AI"
    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"
    store_analysis_metadata: bool = False
    raw_input_retention_hours: int = Field(default=0, ge=0, le=24)
    max_upload_size_mb: int = Field(default=25, ge=1, le=100)
    model_version: str = "untrained-phase-1"
    device: str = "auto"

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

