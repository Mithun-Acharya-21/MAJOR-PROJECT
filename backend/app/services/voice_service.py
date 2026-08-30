"""
Voice/Speech Analysis Service — Audio-based depression screening.

Combines Whisper transcription with acoustic feature analysis.
Used by POST /api/analyze/voice endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("neuroscreen.voice_service")


def analyze_voice(
    audio_bytes: bytes,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run voice/speech analysis on audio data.

    Pipeline:
      1. Decode audio
      2. Transcribe with Whisper
      3. Extract acoustic features
      4. Score depression risk
      5. Generate LIME explanation for transcript

    Args:
        audio_bytes: Raw audio file bytes.
        session_id: Optional session for transcript accumulation.

    Returns:
        Structured result dict with prediction, transcript, features, explanation.
    """
    from backend.app.voice_analysis import extract_voice_features, score_voice
    from backend.app.session_manager import session_manager

    if not audio_bytes or len(audio_bytes) < 100:
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 50.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "features": {},
            "explanation": {"method": "none"},
            "risk_level": "low",
            "transcript": "",
            "error": "No audio data provided",
        }

    transcript = ""
    voice_features_dict = None
    voice_score = None

    # --- Step 1: Transcribe with Whisper ---
    try:
        from backend.app.live_stt import transcribe_audio
        segments = transcribe_audio(audio_bytes, return_segments=True)
        if segments:
            chunk_text = " ".join(s["text"] for s in segments)
            if session_id:
                session = session_manager.get_or_create(session_id)
                full_transcript, accepted = session_manager.add_transcript(session_id, chunk_text)
                transcript = full_transcript
            else:
                transcript = chunk_text
        else:
            transcript = ""
    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)

    # --- Step 2: Extract acoustic features ---
    try:
        voice_result = extract_voice_features(audio_bytes)
        if voice_result is not None:
            features, voice_score = voice_result
            voice_features_dict = features.to_dict()
    except Exception as e:
        logger.warning("Voice feature extraction failed: %s", e)

    # --- Step 3: Analyze transcript with lexicon ---
    transcript_score = 0.5
    if transcript.strip():
        from backend.app.proxy_scorer import score_text
        transcript_score, word_scores = score_text(transcript)
    else:
        word_scores = {}

    # --- Step 4: Combine voice + transcript scores ---
    if voice_score is not None and transcript.strip():
        # Weighted blend: 60% transcript, 40% voice
        depressed_prob = 0.6 * transcript_score + 0.4 * voice_score
    elif voice_score is not None:
        depressed_prob = voice_score
    elif transcript.strip():
        depressed_prob = transcript_score
    else:
        depressed_prob = 0.5

    depressed_prob = max(0.01, min(0.99, depressed_prob))
    confidence = abs(depressed_prob - 0.5) * 200

    prediction = "Depressed" if depressed_prob >= 0.5 else "Non-Depressed"

    if depressed_prob >= 0.7:
        risk_level = "high"
    elif depressed_prob >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # --- Step 5: Generate explanation ---
    top_features = []
    for token, weight in sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        direction = "risk_increasing" if weight > 0 else "risk_decreasing"
        top_features.append({
            "feature": token,
            "weight": round(abs(weight), 4),
            "direction": direction,
        })

    # Add voice features as explanation
    if voice_features_dict:
        for k, v in voice_features_dict.items():
            if k in ("silence_ratio", "speaking_rate", "rms_energy", "pitch_variation"):
                # Map voice features to contribution direction
                if k == "silence_ratio" and v > 0.25:
                    top_features.append({"feature": f"high_{k}", "weight": 0.3, "direction": "risk_increasing"})
                elif k == "speaking_rate" and v < 2.0:
                    top_features.append({"feature": f"low_{k}", "weight": 0.25, "direction": "risk_increasing"})
                elif k == "rms_energy" and v < 0.03:
                    top_features.append({"feature": f"low_{k}", "weight": 0.2, "direction": "risk_increasing"})
                elif k == "pitch_variation" and v < 5.0:
                    top_features.append({"feature": f"low_{k}", "weight": 0.2, "direction": "risk_increasing"})

    explanation = {
        "method": "voice_acoustic" if voice_score else "transcript_lexicon",
        "top_features": top_features,
        "voice_contribution": round(voice_score * 100, 1) if voice_score else None,
        "transcript_contribution": round(transcript_score * 100, 1) if transcript.strip() else None,
    }

    features = {
        "transcript_length": len(transcript),
        "has_voice_features": voice_features_dict is not None,
        **(voice_features_dict or {}),
    }

    return {
        "prediction": prediction,
        "probability": round(depressed_prob * 100, 1),
        "confidence": round(confidence, 1),
        "probabilities": {
            "depressed": round(depressed_prob * 100, 1),
            "non_depressed": round((1 - depressed_prob) * 100, 1),
        },
        "features": features,
        "explanation": explanation,
        "risk_level": risk_level,
        "transcript": transcript,
        "voice_features": voice_features_dict,
    }
