"""
DAIC-WOZ API Data Schemas & Validation Models
NeuroScreen Major Project - Phase 10 Backend Schemas
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DaicWozTextRequest(BaseModel):
    transcript_text: str = Field(..., min_length=1, description="Participant interview utterance transcript text")
    participant_id: Optional[str] = Field(None, description="Optional participant identifier")


class DaicWozVisualFeatureRequest(BaseModel):
    features: List[float] = Field(..., description="40-dimensional OpenFace AU and kinematic summary vector")
    participant_id: Optional[str] = Field(None, description="Optional participant identifier")


class DaicWozMultimodalRequest(BaseModel):
    transcript_text: Optional[str] = Field(None, description="Participant speech transcript")
    visual_features: Optional[List[float]] = Field(None, description="OpenFace 40-dimensional feature vector")
    participant_id: Optional[str] = Field(None, description="Optional participant identifier")


class ModalityPrediction(BaseModel):
    modality: str
    available: bool
    model_name: str
    raw_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    status: str


class DaicWozPredictionResponse(BaseModel):
    screening_indicator: str
    calibrated_probability: Optional[float] = None
    risk_level: str
    confidence_interval: Optional[Dict[str, float]] = None
    modality_breakdown: Dict[str, ModalityPrediction]
    fusion_applied: bool
    model_provenance: Dict[str, str]
    research_disclaimer: str = (
        "Experimental research screening prototype. Not a diagnostic medical device. "
        "Results must not replace professional clinical evaluation."
    )


class DaicWozModelStatusResponse(BaseModel):
    text_model_loaded: bool
    audio_model_loaded: bool
    visual_model_loaded: bool
    fusion_model_loaded: bool
    device: str
    dataset_acquisition_status: str
    active_checkpoints: Dict[str, Optional[str]]


class DaicWozMetricsResponse(BaseModel):
    benchmark_name: str = "DAIC-WOZ / AVEC Depression Benchmark"
    split_protocol: str = "Stratified Participant-Level Partitioning"
    modalities_evaluated: List[str]
    target_accuracy_band: str = "70% - 80% Balanced Accuracy"
    metrics_summary: Dict[str, Any]
