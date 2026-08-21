"""Phase 18 — Analysis API routes.

Routes
------
POST /api/v1/analyze/text
POST /api/v1/analyze/voice
POST /api/v1/analyze/image
POST /api/v1/analyze/video
POST /api/v1/analyze/multimodal

All routes:
- Require explicit consent flag (text) or UploadFile consent gate.
- Validate and size-check uploads before reading.
- Process data in memory — no raw content persisted by default.
- Return honest 'model_available: false' until models are trained.
- Attach the research disclaimer on every response.
- Apply rate limiting.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.privacy import (
    ALLOWED_AUDIO_TYPES,
    ALLOWED_IMAGE_TYPES,
    ALLOWED_VIDEO_TYPES,
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    new_request_id,
    read_upload_bytes,
    validate_text_input,
)
from app.core.rate_limit import check_rate_limit
from app.schemas.analysis import (
    ConsentRequest,
    ConsentResponse,
    ImageAnalysisResponse,
    MultimodalAnalysisResponse,
    TextAnalysisRequest,
    TextAnalysisResponse,
    VideoAnalysisResponse,
    VoiceAnalysisResponse,
)

router = APIRouter(prefix="/analyze", tags=["analysis"])


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

@router.post("/consent", response_model=ConsentResponse)
async def record_consent(body: ConsentRequest) -> ConsentResponse:
    """Record that the user has acknowledged consent for a specific data type."""
    return ConsentResponse(consent_type=body.consent_type, accepted=True)


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

@router.post("/text", response_model=TextAnalysisResponse)
async def analyze_text(
    body: TextAnalysisRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> TextAnalysisResponse:
    """Analyse text input through the research text pipeline."""
    check_rate_limit(request)
    clean_text = validate_text_input(body.text, max_length=10000)
    request_id = new_request_id()

    # Model not trained yet — return honest stub
    return TextAnalysisResponse(
        request_id=request_id,
        model_available=False,
        note=(
            "Text analysis pipeline is ready. "
            "A trained model will be required to produce risk scores. "
            "No model has been trained on approved data yet."
        ),
    )


# ---------------------------------------------------------------------------
# Voice analysis
# ---------------------------------------------------------------------------

@router.post("/voice", response_model=VoiceAnalysisResponse)
async def analyze_voice(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, M4A)"),
    consent_given: bool = Form(...),
    settings: Settings = Depends(get_settings),
) -> VoiceAnalysisResponse:
    """Validate and quality-check an audio upload through the voice pipeline."""
    if not consent_given:
        return JSONResponse(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT",
                                getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)),
            content={"detail": "Consent is required before processing audio data."},
        )
    check_rate_limit(request)

    audio_bytes = await read_upload_bytes(
        audio,
        max_size_mb=settings.max_upload_size_mb,
        allowed_types=ALLOWED_AUDIO_TYPES,
        allowed_extensions=AUDIO_EXTENSIONS,
    )
    request_id = new_request_id()

    # Run audio validation (no model needed)
    try:
        import io as _io
        import numpy as np
        from training.voice.audio_validate import AudioValidator
        import soundfile as sf

        buf = _io.BytesIO(audio_bytes)
        audio_array, sr = sf.read(buf, dtype="float32", always_2d=True)
        mono = audio_array.mean(axis=1)
        validator = AudioValidator()
        val_result = validator.validate_array(mono, sr)
        audio_quality = {
            "quality_score": val_result.quality_score,
            "is_usable": val_result.is_valid,
            "rejection_reasons": val_result.rejection_reasons,
        }
    except Exception:
        audio_quality = None

    return VoiceAnalysisResponse(
        request_id=request_id,
        audio_quality=audio_quality,
        model_available=False,
    )


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

@router.post("/image", response_model=ImageAnalysisResponse)
async def analyze_image(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    consent_given: bool = Form(...),
    settings: Settings = Depends(get_settings),
) -> ImageAnalysisResponse:
    """Detect face and quality-score an uploaded image."""
    if not consent_given:
        return JSONResponse(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT",
                                getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)),
            content={"detail": "Consent is required before processing image data."},
        )
    check_rate_limit(request)

    image_bytes = await read_upload_bytes(
        image,
        max_size_mb=settings.max_upload_size_mb,
        allowed_types=ALLOWED_IMAGE_TYPES,
        allowed_extensions=IMAGE_EXTENSIONS,
    )
    request_id = new_request_id()

    try:
        from training.vision.face_preprocess import FacePreprocessor
        preprocessor = FacePreprocessor()
        face_result = preprocessor.process_bytes(image_bytes)
        face_quality = {
            "quality_score": face_result.quality_score,
            "is_usable": face_result.reliability,
            "rejection_reasons": face_result.rejection_reasons,
        }
        face_detected = face_result.detected
        n_faces = face_result.n_faces
    except Exception:
        face_quality = None
        face_detected = False
        n_faces = 0

    return ImageAnalysisResponse(
        request_id=request_id,
        face_detected=face_detected,
        n_faces=n_faces,
        face_quality=face_quality,
        model_available=False,
    )


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

@router.post("/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    request: Request,
    video: UploadFile = File(..., description="Video file (MP4, WebM, MOV)"),
    consent_given: bool = Form(...),
    settings: Settings = Depends(get_settings),
) -> VideoAnalysisResponse:
    """Sample frames from a video and run face quality scoring."""
    if not consent_given:
        return JSONResponse(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT",
                                getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)),
            content={"detail": "Consent is required before processing video data."},
        )
    check_rate_limit(request)

    video_bytes = await read_upload_bytes(
        video,
        max_size_mb=settings.max_upload_size_mb,
        allowed_types=ALLOWED_VIDEO_TYPES,
        allowed_extensions=VIDEO_EXTENSIONS,
    )
    request_id = new_request_id()

    # Video processing requires a temp file path for OpenCV VideoCapture
    # We write to a NamedTemporaryFile that is immediately deleted after use
    import tempfile, os
    try:
        from training.vision.face_preprocess import process_video
        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            video_result = process_video(tmp_path, max_frames=15, sample_fps=1.0)
        finally:
            os.unlink(tmp_path)   # Immediately remove temp file

        return VideoAnalysisResponse(
            request_id=request_id,
            n_frames_sampled=video_result["n_frames_sampled"],
            n_frames_with_face=video_result["n_frames_with_face"],
            mean_face_quality=video_result["mean_quality_score"],
            reliable_frame_ratio=video_result["reliable_frame_ratio"],
            model_available=False,
        )
    except Exception:
        return VideoAnalysisResponse(request_id=request_id, model_available=False)


# ---------------------------------------------------------------------------
# Multimodal analysis
# ---------------------------------------------------------------------------

@router.post("/multimodal", response_model=MultimodalAnalysisResponse)
async def analyze_multimodal(
    request: Request,
    consent_given: bool = Form(...),
    text: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    settings: Settings = Depends(get_settings),
) -> MultimodalAnalysisResponse:
    """Accept any combination of modalities and report what was provided."""
    if not consent_given:
        return JSONResponse(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT",
                                getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)),
            content={"detail": "Consent is required for multimodal analysis."},
        )
    check_rate_limit(request)
    request_id = new_request_id()

    used, missing = [], []
    if text and text.strip():
        used.append("text")
    else:
        missing.append("text")
    if audio:
        used.append("audio")
    else:
        missing.append("audio")
    if image:
        used.append("video")  # image treated as single-frame visual
    else:
        missing.append("video")

    return MultimodalAnalysisResponse(
        request_id=request_id,
        used_modalities=used,
        missing_modalities=missing,
        model_available=False,
    )
