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
from backend.app.training import DEFAULT_CHECKPOINT_DIR, apply_checkpoint
from backend.app.training_state import get_training_status, start_training
from backend.app.live_stt import transcribe_audio
from backend.app.voice_analysis import extract_voice_features, score_voice
from backend.app.session_manager import SessionManager, session_manager, VoiceFeatures
from backend.app.recommendations import generate_recommendations, check_safety_keywords

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("neuroscreen")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class ModelState:
    """Holds lazily-initialized model instances."""

    def __init__(self):
        self.text_branch = None
        self.vision_branch = None
        self.fusion_model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load(self):
        """Initialize all model components."""
        if self.loaded:
            return

        logger.info("Loading models onto device: %s", self.device)

        try:
            from backend.app.models.text_branch import TextBranch, get_tokenizer
            from backend.app.models.vision_branch import VisionBranch
            from backend.app.models.fusion_layer import MultimodalFusionModel

            # Text branch
            logger.info("  → Initializing TextBranch (DistilBERT + BiLSTM + Attention)...")
            self.text_branch = TextBranch().to(self.device)
            self.text_branch.eval()
            self.tokenizer = get_tokenizer()

            # Vision branch
            logger.info("  → Initializing VisionBranch (ResNet-18)...")
            self.vision_branch = VisionBranch().to(self.device)
            self.vision_branch.eval()

            # Fusion model
            logger.info("  → Initializing MultimodalFusionModel (Cross-Attention)...")
            self.fusion_model = MultimodalFusionModel().to(self.device)
            self.fusion_model.eval()

            # Load a previously trained checkpoint if one exists
            checkpoint_dir = os.getenv(
                "NEUROSCREEN_CHECKPOINT_DIR", str(_PROJECT_ROOT / DEFAULT_CHECKPOINT_DIR)
            )
            loaded = apply_checkpoint(
                {
                    "text_branch": self.text_branch,
                    "vision_branch": self.vision_branch,
                    "fusion_model": self.fusion_model,
                },
                checkpoint_dir,
            )
            if loaded:
                logger.info("✓ Loaded trained checkpoint weights: %s", ", ".join(loaded))

            self.loaded = True
            logger.info("✓ All models loaded successfully.")

        except Exception as e:
            logger.error("Failed to load models: %s", e)
            logger.error(traceback.format_exc())
            raise


# Singleton
_state = ModelState()

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    try:
        _state.load()
    except Exception as e:
        logger.warning("Models not loaded at startup — will retry on first request: %s", e)
    yield
    logger.info("Shutting down NeuroScreen server.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuroScreen — Multimodal Depression Detection",
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


# ===================================================================
# Routes
# ===================================================================

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the NeuroScreen dashboard HTML with no-cache headers."""
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(
        str(index_path),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health")
async def health_check():
    """Return server and model loading status."""
    return {
        "status": "healthy",
        "models_loaded": _state.loaded,
        "device": str(_state.device),
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
    """Swap inference models for the freshly trained checkpoint."""
    try:
        from backend.app.training import load_models_from_checkpoint

        checkpoint_dir = result.get("checkpoint_dir")
        if not checkpoint_dir:
            return
        models = load_models_from_checkpoint(checkpoint_dir, _state.device)
        if models.get("text_branch") is not None:
            _state.text_branch = models["text_branch"]
        if models.get("vision_branch") is not None:
            _state.vision_branch = models["vision_branch"]
        _state.fusion_model = models["fusion_model"]
        _state.loaded = True
        logger.info("✓ Inference models reloaded from trained checkpoint: %s", checkpoint_dir)
    except Exception as e:
        logger.warning("Could not reload inference models from checkpoint: %s", e)


async def _run_analysis(
    text: Optional[str],
    image_bytes: Optional[bytes],
) -> dict:
    """Shared analysis logic used by both /api/analyze and /live-analyze.

    Runs the full AI pipeline: TextBranch → VisionBranch → Fusion → Classification
    → Explainability. This is the single source of truth for all analysis.

    Args:
        text: Raw text input (may be empty/None for image-only).
        image_bytes: Raw image file bytes (may be None for text-only).

    Returns:
        JSON-serializable dict with prediction, confidence, explanations, etc.
    """
    from backend.app.models.text_branch import preprocess_text
    from backend.app.models.vision_branch import preprocess_image, generate_gradcam

    has_text = text is not None and text.strip() != ""
    has_image = image_bytes is not None and len(image_bytes) > 0

    if not has_text and not has_image:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one input: text, image, or both.",
        )

    # Ensure models are loaded
    if not _state.loaded:
        try:
            _state.load()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Models failed to initialize: {str(e)}",
            )

    modalities_used = []
    text_features = None
    vision_features = None
    text_explanation = None
    image_explanation = None

    try:
        # --- Text Processing ---
        if has_text:
            modalities_used.append("text")
            cleaned = preprocess_text(text)

            # Tokenize
            encoding = _state.tokenizer(
                cleaned,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            input_ids = encoding["input_ids"].to(_state.device)
            attention_mask = encoding["attention_mask"].to(_state.device)

            # Get tokens for explainability
            tokens_list = _state.tokenizer.convert_ids_to_tokens(
                input_ids.squeeze().tolist()
            )

            # Forward pass
            with torch.no_grad():
                text_features, attn_weights = _state.text_branch(input_ids, attention_mask)

            attention_weights_list = attn_weights.squeeze().cpu().tolist()

            # Generate text explanation
            text_explanation = generate_text_explanation(
                text=cleaned,
                attention_weights=attention_weights_list,
                tokens=tokens_list,
            )

            # Enrich saliency with lexicon hints when attention is flat
            _, word_hints = score_text(cleaned)
            if word_hints and text_explanation:
                for item in text_explanation:
                    hint = word_hints.get(item["token"].lower(), 0.0)
                    if hint > 0:
                        boosted = min(1.0, item["weight"] + hint * 0.35)
                        item["weight"] = round(boosted, 4)
                        item["color"] = get_risk_color(boosted)

        # --- Image Processing ---
        if has_image:
            modalities_used.append("image")

            # Preprocess
            image_tensor = preprocess_image(image_bytes).to(_state.device)

            # Forward pass
            with torch.no_grad():
                vision_features = _state.vision_branch(image_tensor)

            # Generate Grad-CAM heatmap
            gradcam_heatmap = generate_gradcam(
                _state.vision_branch, image_tensor
            )

            # Generate image explanation (base64 data URLs)
            image_explanation = generate_image_explanation(
                original_image_bytes=image_bytes,
                gradcam_heatmap=gradcam_heatmap,
            )

        # --- Fusion & Classification ---
        with torch.no_grad():
            logits, probabilities = _state.fusion_model(
                text_features=text_features,
                vision_features=vision_features,
            )

        model_probs = probabilities.squeeze().cpu().tolist()

        # Blend with proxy heuristics for meaningful demo outputs
        cleaned_text = preprocess_text(text) if has_text else None

        probs, _ = blend_probabilities(
            model_probs=model_probs,
            text=cleaned_text,
            image_bytes=image_bytes if has_image else None,
            modalities=modalities_used,
        )

        predicted_class = 1 if probs[1] >= probs[0] else 0

        # Class mapping: 0 = Non-Depressed, 1 = Depressed
        prediction = "Depressed" if predicted_class == 1 else "Non-Depressed"
        confidence = round(probs[predicted_class] * 100, 1)

        # Determine risk level
        depressed_prob = probs[1]
        if depressed_prob >= 0.7:
            risk_level = "high"
        elif depressed_prob >= 0.4:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "depressed": round(probs[1] * 100, 1),
                "non_depressed": round(probs[0] * 100, 1),
            },
            "modalities_used": modalities_used,
            "text_explanation": text_explanation,
            "image_explanation": image_explanation,
            "risk_level": risk_level,
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
        elif blended_depressed >= 0.4:
            result["risk_level"] = "moderate"
        else:
            result["risk_level"] = "low"

    # Generate recommendations
    session_manager.set_last_result(effective_session_id, result)
    recommendations = generate_recommendations(result)
    result["recommendations"] = recommendations
    result["risk_status"] = recommendations.get("status", "Unknown")

    return result
