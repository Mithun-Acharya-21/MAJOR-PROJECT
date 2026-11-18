"""
DAIC-WOZ API Endpoints Router
NeuroScreen Major Project - Phase 10 API Routes
"""

import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List

from backend.app.daic_woz.schemas import (
    DaicWozTextRequest,
    DaicWozVisualFeatureRequest,
    DaicWozMultimodalRequest,
    DaicWozPredictionResponse,
    DaicWozModelStatusResponse,
    DaicWozMetricsResponse
)
from backend.app.daic_woz.inference_service import DaicWozInferenceService
from backend.app.daic_woz.model_loader import DaicWozModelLoader

router = APIRouter(prefix="/api/daic-woz", tags=["DAIC-WOZ Research"])
inference_service = DaicWozInferenceService()


@router.get("/model/status", response_model=DaicWozModelStatusResponse)
def get_daic_woz_model_status():
    """Returns runtime checkpoint status and device information."""
    loader = DaicWozModelLoader.get_instance()
    status = loader.get_status()
    return DaicWozModelStatusResponse(**status)


@router.get("/model/metrics", response_model=DaicWozMetricsResponse)
def get_daic_woz_metrics():
    """Returns evaluation metrics summary from official test partitions."""
    metrics_path = Path("results/daic_woz/reports/test_metrics_summary.json")
    summary = {}
    if metrics_path.exists():
        try:
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return DaicWozMetricsResponse(
        benchmark_name="DAIC-WOZ (USC ICT) / AVEC Challenge",
        split_protocol="Stratified Participant-Level Partitioning",
        modalities_evaluated=["Text (Participant Utterances)", "Audio (openSMILE eGeMAPS)", "Visual (OpenFace Kinematics)"],
        target_accuracy_band="70% - 80% Balanced Accuracy",
        metrics_summary=summary
    )


@router.post("/analyze/text", response_model=DaicWozPredictionResponse)
def analyze_daic_woz_text(payload: DaicWozTextRequest):
    """Processes participant transcript text using DAIC-WOZ trained text model."""
    return inference_service.analyze_multimodal(transcript_text=payload.transcript_text)


@router.post("/analyze/visual", response_model=DaicWozPredictionResponse)
def analyze_daic_woz_visual(payload: DaicWozVisualFeatureRequest):
    """Processes OpenFace facial kinematic features using DAIC-WOZ visual model."""
    return inference_service.analyze_multimodal(visual_features=payload.features)


@router.post("/analyze", response_model=DaicWozPredictionResponse)
def analyze_daic_woz_multimodal(payload: DaicWozMultimodalRequest):
    """Executes multimodal screening analysis combining available modalities."""
    return inference_service.analyze_multimodal(
        transcript_text=payload.transcript_text,
        visual_features=payload.visual_features
    )


@router.post("/analyze/audio", response_model=DaicWozPredictionResponse)
def analyze_daic_woz_audio(file: UploadFile = File(...)):
    """Processes audio recording file using DAIC-WOZ acoustic model."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format. Please provide a WAV or MP3 file.")

    return inference_service.analyze_multimodal()
