from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.system import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness endpoint; it deliberately loads no models or user data."""

    return HealthResponse(status="healthy", service=settings.app_name, environment=settings.app_env)


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(settings: Settings = Depends(get_settings)) -> ModelInfoResponse:
    """Expose only honest model readiness information."""

    return ModelInfoResponse(
        model_version=settings.model_version,
        readiness="architecture_only",
        supported_modalities=["text", "voice", "image", "video"],
        raw_input_storage_default=settings.store_analysis_metadata,
    )

