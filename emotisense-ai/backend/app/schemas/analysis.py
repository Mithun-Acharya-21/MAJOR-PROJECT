"""Pydantic schemas for all analysis routes (Phase 18)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.safety import RESEARCH_DISCLAIMER

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

MODALITY_DISCLAIMER = (
    "These are model-derived signals from a research system. "
    "They are not a clinical diagnosis and must not replace professional assessment."
)


class ModalityQuality(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    is_usable: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class EmotionProfile(BaseModel):
    probabilities: dict[str, float]
    predicted_emotion: str


class GateSummary(BaseModel):
    modality: str
    gate_value: float
    contribution_pct: float
    quality_score: Optional[float] = None
    active: bool


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

class TextAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    consent_given: bool = Field(description="User must explicitly consent before analysis")

    @field_validator("consent_given")
    @classmethod
    def consent_required(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Consent must be given before analysis")
        return v


class TextAnalysisResponse(BaseModel):
    request_id: str
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_tier: Optional[str] = None
    uncertainty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_tokens: list[str] = Field(default_factory=list)
    model_available: bool = False
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)
    note: str = "Text analysis is a research-only feature. No model is trained yet."


# ---------------------------------------------------------------------------
# Voice analysis
# ---------------------------------------------------------------------------

class VoiceAnalysisResponse(BaseModel):
    request_id: str
    transcript: Optional[str] = None
    language: Optional[str] = None
    asr_quality: Optional[ModalityQuality] = None
    audio_quality: Optional[ModalityQuality] = None
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_tier: Optional[str] = None
    emotion_profile: Optional[EmotionProfile] = None
    combined_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_available: bool = False
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

class ImageAnalysisResponse(BaseModel):
    request_id: str
    face_detected: bool = False
    n_faces: int = 0
    face_quality: Optional[ModalityQuality] = None
    emotion_profile: Optional[EmotionProfile] = None
    gradcam_heatmap_base64: Optional[str] = None
    model_available: bool = False
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)
    note: str = "Facial expression is an emotion signal, not a depression indicator."


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

class VideoAnalysisResponse(BaseModel):
    request_id: str
    n_frames_sampled: int = 0
    n_frames_with_face: int = 0
    mean_face_quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reliable_frame_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    emotion_profile: Optional[EmotionProfile] = None
    model_available: bool = False
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)


# ---------------------------------------------------------------------------
# Multimodal analysis
# ---------------------------------------------------------------------------

class MultimodalAnalysisResponse(BaseModel):
    request_id: str
    used_modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_tier: Optional[str] = None
    uncertainty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gate_contributions: list[GateSummary] = Field(default_factory=list)
    emotion_profile: Optional[EmotionProfile] = None
    suggestions: list[dict] = Field(default_factory=list)
    policy_disclaimer: Optional[str] = None
    model_available: bool = False
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

class ConsentRequest(BaseModel):
    consent_type: Literal["audio_recording", "image_upload", "video_upload", "text_analysis"]
    user_acknowledged: bool

    @field_validator("user_acknowledged")
    @classmethod
    def must_acknowledge(cls, v: bool) -> bool:
        if not v:
            raise ValueError("User must acknowledge consent")
        return v


class ConsentResponse(BaseModel):
    consent_type: str
    accepted: bool
    note: str = (
        "Consent is required before processing any biometric data. "
        "Data is processed in memory and not persisted by default."
    )
