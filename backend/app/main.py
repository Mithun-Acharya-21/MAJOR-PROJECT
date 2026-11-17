"""
FastAPI Application — Multimodal Early Depression Detection

Endpoints:
  POST /api/analyze       — Process text and/or image, return prediction + explanations
  GET  /api/baselines     — Return model benchmark comparison data
  GET  /api/training-status — Poll the training session
  POST /api/train         — Start a real fine-tuning run on a local dataset
  GET  /api/health        — Health check with model loading status
  GET  /                  — Serve the frontend dashboard

The application lazily initializes PyTorch models on first use and
serves the frontend as static files from the ``frontend/`` directory.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import Body, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Path setup — ensure the backend package is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # multimodal-depression-detection/
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.baselines import get_baseline_metrics
from backend.app.explainability.explainer import (
    generate_image_explanation,
    generate_text_explanation,
    get_risk_color,
)
from backend.app.proxy_scorer import blend_probabilities, score_text
from backend.app.training_state import get_training_status, start_training
from backend.app.live_stt import transcribe_audio
from backend.app.voice_analysis import extract_voice_features, score_voice
from backend.app.session_manager import SessionManager, session_manager, VoiceFeatures
from backend.app.recommendations import generate_recommendations, check_safety_keywords

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emotiondep")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class ModelState:
    """Verification status of servable model checkpoints (Phase 0).

    This class no longer instantiates randomly initialized neural branches.
    It reports which artifacts passed ``backend.app.model_registry``
    verification. Trained text weights are loaded lazily by
    ``TrainedTextClassifier``. No vision/fusion checkpoint is servable until
    one is trained on a participant-level split and declares that policy in
    its metadata.
    """

    def __init__(self):
        # Kept for backward compatibility with callers that pass model_state.
        self.text_branch = None
        self.vision_branch = None
        self.fusion_model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.verification: dict = {}
        self.load_error: Optional[str] = None

    def load(self):
        """Verify checkpoints; never fabricate a model probability."""
        if self.verification:
            return

        from backend.app.model_registry import audit_checkpoints

        logger.info("Verifying model checkpoints on device: %s", self.device)
        self.verification = audit_checkpoints()
        self.loaded = any(
            status["available"] for status in self.verification.values()
        )
        if not self.loaded:
            self.load_error = "No verified model checkpoint is available"
            logger.warning(
                "No verified model checkpoint — inference will use explicitly "
                "labeled heuristic signals only."
            )
        else:
            servable = [
                branch
                for branch, status in self.verification.items()
                if status["available"]
            ]
            logger.info("Verified servable branches: %s", ", ".join(servable))


# Singleton
_state = ModelState()

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify model checkpoints at startup."""
    try:
        _state.load()
    except Exception as e:
        logger.warning("Models not loaded at startup — will retry on first request: %s", e)
    yield
    logger.info("Shutting down EmotionDep server.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EmotionDep — Emotion-Aware Depression Detection",
    description="AI-powered screening tool combining text analysis and facial expression recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files — serve frontend
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

# Adaptive screening routes (new — existing endpoints are untouched)
from backend.app.screening_routes import router as screening_router

screening_router.model_state = _state  # shared ModelState for modality services
app.include_router(screening_router)


# ===================================================================
# Routes
# ===================================================================

@app.get("/")
async def serve_frontend():
    """Serve a static build if present; otherwise describe the API.

    The real UI is the Next.js app in ``frontend/`` served on its own port.
    Previously this route raised 404 because ``frontend/index.html`` does not
    exist in this repository.
    """
    index_path = _FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return JSONResponse(
        {
            "service": "EmotionDep — multimodal depression screening research API",
            "status": "ok",
            "api_docs": "/docs",
            "health": "/api/health",
            "frontend": "Run the Next.js app in frontend/ (npm run dev)",
            "note": "Research/educational screening only — not a medical diagnosis.",
        }
    )


@app.get("/api/health")
async def health_check():
    """Return server and model loading status."""
    return {
        "status": "healthy",
        "models_loaded": _state.loaded,
        "device": str(_state.device),
        "verified_checkpoints": _state.verification,
    }


@app.get("/api/baselines")
async def baselines():
    """Return baseline benchmark comparison data."""
    status = get_training_status()
    if status.get("metrics"):
        return status["metrics"]
    return get_baseline_metrics()


@app.get("/api/training-status")
async def training_status():
    """Return current training session status."""
    return get_training_status()


@app.post("/api/train")
async def train(payload: Optional[dict] = Body(default=None)):
    """Start a real fine-tuning run on a local dataset.

    JSON body (all optional except one dataset source):
      text_csv, image_dir / image_csv, multimodal_csv, epochs,
      batch_size, learning_rate, val_split, seed, unfreeze_encoder,
      checkpoint_dir, device

    Returns immediately; poll ``/api/training-status`` for progress.
    """
    try:
        return start_training(payload or {}, on_complete=_reload_from_training)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to start training: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _reload_from_training(result: dict) -> None:
    """Refuse to serve an unverified training artifact (Phase 0).

    The API trainer in ``backend.app.training`` uses row-level splits, so its
    checkpoint cannot demonstrate participant-independent performance. It is
    kept on disk for inspection but is never swapped into inference. Only
    artifacts approved by ``backend.app.model_registry`` may serve.
    """
    checkpoint_dir = (result or {}).get("checkpoint_dir")
    logger.info(
        "Training finished (%s). Checkpoint NOT loaded into serving: it lacks "
        "participant-level verification (see backend/app/model_registry.py).",
        checkpoint_dir,
    )


async def _run_analysis(
    text: Optional[str],
    image_bytes: Optional[bytes],
) -> dict:
    """Shared analysis logic used by /api/analyze and /live-analyze.

    Phase 0: no randomly initialized network may contribute. Text goes through
    the trained-model-or-lexicon service (with explicit provenance labels) and
    images through the auxiliary expression service. The two signals are
    combined with the documented weighted fusion, never a random fusion head.
    """
    from backend.app.services.fusion_service import DEFAULT_WEIGHTS, fuse_multimodal
    from backend.app.services.photo_service import analyze_photo
    from backend.app.services.text_service import analyze_text

    has_text = text is not None and text.strip() != ""
    has_image = image_bytes is not None and len(image_bytes) > 0

    if not has_text and not has_image:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one input: text, image, or both.",
        )

    # Populates checkpoint verification status; never raises on missing models.
    _state.load()

    modality_results: dict = {}
    modalities_used = []

    try:
        text_result = None
        photo_result = None

        if has_text:
            text_result = analyze_text(text, _state)
            modality_results["text"] = text_result
            modalities_used.append("text")

        if has_image:
            photo_result = analyze_photo(image_bytes, _state)
            modality_results["photo"] = photo_result
            modalities_used.append("image")

        fused = fuse_multimodal(modality_results, DEFAULT_WEIGHTS)
        depressed_prob = fused["probabilities"]["depressed"] / 100.0
        predicted_class = 1 if depressed_prob >= 0.5 else 0
        prediction = "Depressed" if predicted_class == 1 else "Non-Depressed"

        if depressed_prob >= 0.7:
            risk_level = "high"
        elif depressed_prob >= 0.5:
            risk_level = "moderate"
        else:
            risk_level = "low"

        text_explanation = None
        if text_result is not None:
            explanation = text_result.get("explanation") or {}
            features = explanation.get("top_features") or []
            if not features:
                features = [
                    {"feature": token.get("token"), "weight": token.get("weight")}
                    for token in explanation.get("attention_tokens", [])
                ]
            text_explanation = [
                {
                    "token": item.get("feature") or item.get("token"),
                    "weight": round(float(item.get("weight", 0.0)), 4),
                    "color": get_risk_color(
                        min(1.0, max(0.0, float(item.get("weight", 0.0))))
                    ),
                }
                for item in features
                if (item.get("feature") or item.get("token"))
            ]

        image_explanation = None
        if photo_result is not None:
            image_explanation = {
                "gradcam": None,
                "method": (photo_result.get("explanation") or {}).get("method"),
                "face_detected": (photo_result.get("features") or {}).get("face_detected"),
                "signal_type": photo_result.get("signal_type"),
                "note": (
                    "No Grad-CAM is produced: this project has no trained image "
                    "classifier for a depression target. The visual signal is an "
                    "auxiliary facial-expression score, not a depression prediction."
                ),
            }

        return {
            "prediction": prediction,
            "confidence": fused.get("confidence", 50.0),
            "probabilities": {
                "depressed": fused["probabilities"]["depressed"],
                "non_depressed": fused["probabilities"]["non_depressed"],
            },
            "modalities_used": modalities_used,
            "text_explanation": text_explanation,
            "text_explanation_note": (
                "Token attributions come from the actual prediction function "
                "(trained-model attention/LIME when available, otherwise the "
                "lexicon proxy). They describe model behaviour, not causation."
            ),
            "image_explanation": image_explanation,
            "risk_level": risk_level,
            "fusion_type": fused.get("fusion_type"),
            "fusion_weights": fused.get("fusion_weights"),
            "signal_types": fused.get("signal_types"),
            "validated_modalities": fused.get("validated_modalities"),
            "model_version": (
                text_result.get("model_version") if text_result is not None else None
            ),
            "method": text_result.get("method") if text_result is not None else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}",
        )


@app.post("/api/analyze")
async def analyze(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """Analyze text and/or facial image for depression indicators.

    Accepts multipart form data with optional ``text`` field and/or
    ``file`` (image upload). At least one must be provided.

    Returns:
        JSON with prediction, confidence, explanations, and risk level.
    """
    image_bytes = None
    if file is not None:
        image_bytes = await file.read()
    result = await _run_analysis(text=text, image_bytes=image_bytes)

    # Include original text for safety checking in recommendations
    if text:
        result["transcript"] = text

    # Add recommendations (same logic as live-analyze)
    recommendations = generate_recommendations(result)
    result["recommendations"] = recommendations
    result["risk_status"] = recommendations.get("status", "Unknown")

    return result


@app.post("/session/start")
async def session_start():
    """Start a new live-analysis session.

    Resets all accumulated state: transcript, voice features,
    facial scores, errors, and timestamps.
    """
    session = session_manager.start_session()
    return {"session_id": session.session_id, "status": "started"}


@app.post("/session/{session_id}/stop")
async def session_stop(session_id: str):
    """Stop a session and return the final accumulated state."""
    session = session_manager.end_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.get_summary()


@app.post("/live-analyze")
async def live_analyze(
    image: UploadFile = File(..., description="Camera frame / photo (JPEG/PNG)."),
    audio: Optional[UploadFile] = File(None, description="Recorded audio chunk (WebM/WAV/MP3)."),
    session_id: Optional[str] = Form(None, description="Session ID for transcript accumulation."),
):
    """Live analysis: speech-to-text + image + voice analysis → same AI pipeline.

    Accepts an image (from the live camera), an audio file (from the
    microphone), and an optional session_id for transcript accumulation.

    The audio is processed in two parallel pipelines:
      1. Whisper → text → text analysis
      2. Raw audio → acoustic features → voice analysis

    Session transcript accumulates across ticks (no more replacement).
    Duplicate/overlapping transcriptions are automatically filtered.

    Returns:
        JSON with prediction, confidence, explanations, transcript,
        session_id, voice_features, and recommendations.
    """
    # Get or create session
    session = session_manager.get_or_create(session_id)
    effective_session_id = session.session_id

    try:
        image_bytes = await image.read()
    except Exception:
        session_manager.add_error(effective_session_id, "Failed to read camera frame")
        raise HTTPException(status_code=400, detail="Failed to read image file.")

    # --- Process audio: Whisper transcription + voice feature extraction ---
    transcript = ""
    voice_features_dict = None
    voice_score = None

    if audio is not None:
        try:
            audio_bytes = await audio.read()
        except Exception:
            session_manager.add_error(effective_session_id, "Failed to read audio")
            raise HTTPException(status_code=400, detail="Failed to read audio file.")

        if audio_bytes and len(audio_bytes) > 100:
            # --- Step 1a: Speech-to-Text with segment deduplication ---
            try:
                segments = transcribe_audio(audio_bytes, return_segments=True)
                if segments:
                    # Combine segment texts, deduplicate against session history
                    chunk_text = " ".join(s["text"] for s in segments)
                    full_transcript, accepted = session_manager.add_transcript(
                        effective_session_id, chunk_text
                    )
                    transcript = full_transcript
                    session_manager.set_modality(effective_session_id, "transcript", True)
                    logger.info(
                        "Live transcription (session %s, %s): %s",
                        effective_session_id,
                        "accepted" if accepted else "deduped",
                        chunk_text[:200],
                    )
                else:
                    transcript = session.full_transcript  # Use accumulated
            except Exception as exc:
                logger.error("Speech-to-text failed: %s", exc)
                session_manager.add_error(effective_session_id, f"Whisper failed: {exc}")
                transcript = session.full_transcript  # Use accumulated on error

            # --- Step 1b: Voice/Acoustic Feature Extraction ---
            try:
                voice_result = extract_voice_features(audio_bytes)
                if voice_result is not None:
                    features, voice_score_raw = voice_result
                    session_manager.add_voice_features(effective_session_id, features)
                    voice_score = score_voice(features)
                    voice_features_dict = features.to_dict()
                    session_manager.set_modality(effective_session_id, "voice", True)
                    logger.info(
                        "Voice features (session %s): rms=%.3f, rate=%.1f, silence=%.2f",
                        effective_session_id,
                        features.rms_energy,
                        features.speaking_rate,
                        features.silence_ratio,
                    )
            except Exception as exc:
                logger.warning("Voice feature extraction failed: %s", exc)
                session_manager.add_error(effective_session_id, f"Voice analysis failed: {exc}")
        else:
            transcript = session.full_transcript  # No audio data, use accumulated
    else:
        transcript = session.full_transcript  # No audio provided, use accumulated

    # Use accumulated transcript for analysis (even if this chunk was deduped)
    analysis_text = transcript if transcript.strip() else None

    # --- Step 2: Run the analysis pipeline ---
    try:
        result = await _run_analysis(text=analysis_text, image_bytes=image_bytes)
    except HTTPException:
        session_manager.add_error(effective_session_id, "Analysis pipeline failed")
        raise
    except Exception as exc:
        session_manager.add_error(effective_session_id, f"Analysis error: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    # --- Step 3: Enhance result with session data, voice features, and recommendations ---
    result["transcript"] = transcript
    result["session_id"] = effective_session_id
    result["voice_features"] = voice_features_dict
    result["voice_score"] = voice_score
    result["num_segments"] = len(session.transcript_segments)
    result["modalities_available"] = dict(session.modalities_available)
    result["errors"] = list(session.errors)

    # Add voice score to blended probabilities if available
    if voice_score is not None and "image" not in result.get("modalities_used", []) and not analysis_text:
        # Voice-only case: blend voice score into probabilities
        existing_depressed = result["probabilities"]["depressed"] / 100.0
        blended_depressed = 0.5 * existing_depressed + 0.5 * (voice_score * 100)
        result["probabilities"]["depressed"] = round(blended_depressed, 1)
        result["probabilities"]["non_depressed"] = round(100.0 - blended_depressed, 1)
        if "voice" not in result["modalities_used"]:
            result["modalities_used"].append("voice")
    elif voice_score is not None and "voice" not in result.get("modalities_used", []):
        result["modalities_used"].append("voice")

    # Add voice score to probabilities (weighted blend)
    if voice_score is not None and analysis_text:
        existing_depressed = result["probabilities"]["depressed"] / 100.0
        # Weighted blend: 70% text+vision, 30% voice
        blended_depressed = 0.7 * existing_depressed + 0.3 * voice_score
        result["probabilities"]["depressed"] = round(blended_depressed * 100, 1)
        result["probabilities"]["non_depressed"] = round((1.0 - blended_depressed) * 100, 1)
        # Recompute prediction and risk level
        predicted_class = 1 if blended_depressed >= 0.5 else 0
        result["prediction"] = "Depressed" if predicted_class == 1 else "Non-Depressed"
        result["confidence"] = round(max(blended_depressed, 1.0 - blended_depressed) * 100, 1)
        if blended_depressed >= 0.7:
            result["risk_level"] = "high"
        elif blended_depressed >= 0.5:
            result["risk_level"] = "moderate"
        else:
            result["risk_level"] = "low"

    # Generate recommendations
    session_manager.set_last_result(effective_session_id, result)
    recommendations = generate_recommendations(result)
    result["recommendations"] = recommendations
    result["risk_status"] = recommendations.get("status", "Unknown")

    return result


# ===================================================================
# Step-by-step modality endpoints
# ===================================================================
from typing import List as TypingList


@app.post("/api/analyze/text")
async def analyze_text_endpoint(
    text: str = Form(..., description="Text input for analysis"),
):
    """Analyze text input for depression indicators."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")

    from backend.app.services.text_service import analyze_text
    result = analyze_text(text.strip(), _state)
    result["transcript"] = text
    # Build text_explanation for recommendations
    text_explanation = []
    explanation = result.get("explanation", {})
    if isinstance(explanation, dict):
        text_explanation = explanation.get("attention_tokens", [])
    elif isinstance(explanation, list):
        text_explanation = explanation
    recommendations = generate_recommendations({
        "probabilities": {"depressed": result["probability"], "non_depressed": 100 - result["probability"]},
        "risk_level": result["risk_level"],
        "text_explanation": text_explanation,
    })
    result["recommendations"] = recommendations
    return result


@app.post("/api/analyze/photo")
async def analyze_photo_endpoint(
    file: UploadFile = File(..., description="Facial photo"),
):
    """Analyze a facial photo for depression indicators."""
    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Invalid image file")

    from backend.app.services.photo_service import analyze_photo
    result = analyze_photo(image_bytes, _state)
    recommendations = generate_recommendations({
        "probabilities": {"depressed": result["probability"], "non_depressed": 100 - result["probability"]},
        "risk_level": result["risk_level"],
    })
    result["recommendations"] = recommendations
    return result


@app.post("/api/analyze/face")
async def analyze_face_endpoint(
    frames: TypingList[UploadFile] = File(..., description="Webcam frames"),
):
    """Analyze webcam frames for live facial expression depression risk."""
    frame_bytes_list = []
    for frame in frames:
        try:
            data = await frame.read()
            if data and len(data) > 100:
                frame_bytes_list.append(data)
        except Exception:
            continue
    if not frame_bytes_list:
        raise HTTPException(status_code=400, detail="No valid frames provided")

    from backend.app.services.face_service import analyze_live_face
    result = analyze_live_face(frame_bytes_list, _state)
    recommendations = generate_recommendations({
        "probabilities": {"depressed": result["probability"], "non_depressed": 100 - result["probability"]},
        "risk_level": result["risk_level"],
    })
    result["recommendations"] = recommendations
    return result


@app.post("/api/analyze/voice")
async def analyze_voice_endpoint(
    audio: UploadFile = File(..., description="Audio recording"),
    session_id: Optional[str] = Form(None),
):
    """Analyze voice/speech for depression indicators."""
    audio_bytes = await audio.read()
    if not audio_bytes or len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Invalid or empty audio file")

    from backend.app.services.voice_service import analyze_voice
    result = analyze_voice(audio_bytes, session_id)
    recommendations = generate_recommendations({
        "probabilities": {"depressed": result["probability"], "non_depressed": 100 - result["probability"]},
        "risk_level": result["risk_level"],
        "transcript": result.get("transcript", ""),
        "voice_features": result.get("voice_features"),
    })
    result["recommendations"] = recommendations
    return result


@app.post("/api/analyze/multimodal")
async def analyze_multimodal_endpoint(
    text: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    frames: Optional[TypingList[UploadFile]] = File(None),
    audio: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
):
    """Fuse results from all available modalities."""
    modality_results = {}
    if text and text.strip():
        from backend.app.services.text_service import analyze_text
        modality_results["text"] = analyze_text(text.strip(), _state)
    if photo is not None:
        photo_bytes = await photo.read()
        if photo_bytes and len(photo_bytes) > 100:
            from backend.app.services.photo_service import analyze_photo
            modality_results["photo"] = analyze_photo(photo_bytes, _state)
    if frames:
        frame_bytes_list = []
        for frame in frames:
            try:
                data = await frame.read()
                if data and len(data) > 100:
                    frame_bytes_list.append(data)
            except Exception:
                continue
        if frame_bytes_list:
            from backend.app.services.face_service import analyze_live_face
            modality_results["face"] = analyze_live_face(frame_bytes_list, _state)
    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes and len(audio_bytes) > 100:
            from backend.app.services.voice_service import analyze_voice
            modality_results["voice"] = analyze_voice(audio_bytes, session_id)
    if not modality_results:
        raise HTTPException(status_code=400, detail="No valid inputs provided")

    from backend.app.services.fusion_service import fuse_multimodal
    fused = fuse_multimodal(modality_results)
    fused["individual_results"] = modality_results
    # Build text_explanation for recommendations
    text_explanation = []
    text_result = modality_results.get("text", {})
    text_expl = text_result.get("explanation", {})
    if isinstance(text_expl, dict):
        text_explanation = text_expl.get("attention_tokens", [])
    elif isinstance(text_expl, list):
        text_explanation = text_expl
    recommendations = generate_recommendations({
        "probabilities": {"depressed": fused["probability"], "non_depressed": 100 - fused["probability"]},
        "risk_level": fused["risk_level"],
        "transcript": modality_results.get("voice", {}).get("transcript", ""),
        "text_explanation": text_explanation,
    })
    fused["recommendations"] = recommendations
    return fused


@app.get("/api/model/metrics")
async def model_metrics():
    """Return this project's REAL evaluated metrics (Phase 21).

    Serves results/metrics/text_metrics.json — produced only by an actual
    evaluation run (scripts/evaluate_text_model.py). There is no multimodal
    result because no aligned multimodal dataset exists (Phase 11). When
    the file is absent, the endpoint honestly reports that instead of
    serving literature values as project results.
    """
    import json
    from pathlib import Path

    text_metrics_path = _PROJECT_ROOT / "results" / "metrics" / "text_metrics.json"
    if text_metrics_path.exists():
        try:
            data = json.loads(text_metrics_path.read_text(encoding="utf-8"))
            if data.get("samples"):
                return {
                    "model": data.get("model"),
                    "model_version": data.get("model_version"),
                    "dataset": data.get("dataset"),
                    "split": data.get("split"),
                    "samples": data.get("samples"),
                    "accuracy": data.get("accuracy"),
                    "precision": data.get("precision"),
                    "recall": data.get("recall"),
                    "f1": data.get("f1"),
                    "roc_auc": data.get("roc_auc"),
                    "pr_auc": data.get("pr_auc"),
                    "balanced_accuracy": data.get("balanced_accuracy"),
                    "source": "evaluation",
                    "evaluated_at": data.get("evaluated_at"),
                    "note": (
                        "Real evaluation of the trained text model on the "
                        "locked EATD test split. No multimodal metrics exist "
                        "— no aligned multimodal dataset is available."
                    ),
                }
        except Exception as e:
            logger.warning("Could not read text metrics file: %s", e)

    return {
        "source": "none",
        "metrics": None,
        "note": (
            "No evaluated project metrics available yet. Literature baselines "
            "(labeled as literature) are available via /api/baselines — they "
            "are NOT this project's results."
        ),
    }


@app.get("/api/recommendations")
async def general_recommendations():
    """Return general wellness recommendations (non-personalized)."""
    return generate_recommendations({
        "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
        "risk_level": "low",
    })
