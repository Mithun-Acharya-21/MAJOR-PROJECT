from typing import Literal

from pydantic import BaseModel, Field

from app.core.safety import RESEARCH_DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    service: str
    environment: str


class ModelInfoResponse(BaseModel):
    model_version: str
    readiness: Literal["architecture_only"]
    supported_modalities: list[str]
    raw_input_storage_default: bool = False
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)

