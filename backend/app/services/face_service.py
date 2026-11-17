"""
Live Face Analysis Service — Webcam-based facial expression analysis.

Uses pre-trained FER library for actual facial expression recognition
instead of brightness-based heuristics.

Aggregates predictions over multiple frames for stable results.
Used by POST /api/analyze/face endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("emotiondep.face_service")


def analyze_live_face(
    frames: List[bytes],
    model_state: Any,
) -> Dict[str, Any]:
    """Analyze multiple webcam frames for facial expression depression risk.

    Uses pre-trained FER library for 7-class emotion detection:
    angry, disgust, fear, happy, sad, surprise, neutral.
    Maps emotions to depression probability using research-backed weights.

    Args:
        frames: List of raw image bytes from webcam.
        model_state: ModelState instance with loaded models.

    Returns:
        Structured result dict with aggregated prediction.
    """
    if not frames:
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 50.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "features": {"frame_count": 0},
            "explanation": {"method": "none"},
            "risk_level": "low",
            "error": "No frames provided",
        }

    # --- Use pre-trained FER expression analyzer ---
    try:
        from backend.app.face_expression_analyzer import analyze_multiple_frames
        result = analyze_multiple_frames(frames)
    except Exception as e:
        logger.error("FER analyzer failed: %s — falling back to proxy", e)
        return _fallback_analysis(frames)

    # --- Map to standard response format ---
    depression_prob = result["depression_prob"]
    prediction = result["prediction"]
    confidence = result["confidence"]
    risk_level = result["risk_level"]

    # Expression distribution (percentage)
    emotion_dist = result.get("emotion_distribution", {})
    expression_probs = {
        k: round(v * 100, 1) for k, v in emotion_dist.items()
    }

    # HONESTY LABEL (Phase 9): this is an auxiliary facial-EXPRESSION score
    # (FER emotion weighting), NOT a validated depression probability. No
    # image model in this project has been trained to predict depression.
    expression_note = (
        "Auxiliary facial-expression score from pretrained FER emotion "
        "weighting — NOT a validated depression probability."
    )

    features = {
        "frame_count": result["total_frames"],
        "analyzed_count": result["face_detected_count"],
        "avg_depression_prob": result["depression_prob"],
        "std_depression_prob": result.get("std_across_frames", 0.0),
        "expression_distribution": expression_probs,
        "dominant_emotion": result.get("dominant_emotion_overall", "unknown"),
    }

    explanation = {
        "method": "fer_mtcnn_cnn",
        "frames_analyzed": result["face_detected_count"],
        "consistency": max(0.0, 1.0 - result.get("std_across_frames", 0) * 3),
        "expression_distribution": expression_probs,
        "emotion_weights_used": {
            "sad": 0.35, "fear": 0.15, "angry": 0.15,
            "disgust": 0.10, "happy": -0.25, "surprise": -0.10, "neutral": 0.05,
        },
        "score_interpretation": expression_note,
        "note": expression_note,
    }

    dominant_emotion = result.get("dominant_emotion_overall", "unknown")
    dominant_expression = result.get("dominant_expression") or (
        dominant_emotion.capitalize() if dominant_emotion not in ("no_face", "unknown") else None
    )
    prediction_label = (
        f"Expression: {dominant_expression}"
        if dominant_expression is not None
        else "No Face Detected"
    )

    return {
        "prediction": prediction_label,
        "dominant_expression": dominant_expression,
        "dominant_emotion": dominant_emotion,
        "uncertainty_label": result.get("uncertainty_label", "Multi-frame tracking"),
        "expression_prominence": result.get("avg_expression_prominence", 0.0),
        "observed_movement": result.get("observed_movement", {}),
        "is_depression_prediction": False,
        "is_depression_model": False,
        "model_name": "MediaPipe FaceLandmarker (Multi-Frame Aggregator)",
        "model_type": "Multi-Frame Expression Tracking (Auxiliary)",
        "processing_status": f"{result['face_detected_count']} of {result['total_frames']} frames analyzed with face detected",
        "confidence_source": "temporal_frame_consistency",
        "probability": round(depression_prob * 100, 1),
        "confidence": round(confidence, 1),
        "probabilities": {
            "depressed": round(depression_prob * 100, 1),
            "non_depressed": round((1 - depression_prob) * 100, 1),
        },
        "visual_expression_score": round(depression_prob, 4),
        "signal_type": "auxiliary_expression_score",
        "model_version": None,
        "features": features,
        "explanation": explanation,
        "risk_level": risk_level,
    }


def _fallback_analysis(frames: List[bytes]) -> Dict[str, Any]:
    """Fallback to proxy scoring if FER library is unavailable."""
    from backend.app.proxy_scorer import score_image

    frame_results = []
    for frame_bytes in frames:
        try:
            prob = score_image(frame_bytes)
            frame_results.append(prob)
        except Exception:
            continue

    if not frame_results:
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 0.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "features": {"frame_count": len(frames), "analyzed_count": 0},
            "explanation": {"method": "proxy_fallback"},
            "risk_level": "low",
            "error": "Could not analyze any frames",
        }

    avg_prob = np.mean(frame_results)
    depressed_prob = max(0.01, min(0.99, avg_prob))
    prediction = "Depressed" if depressed_prob >= 0.5 else "Non-Depressed"

    if depressed_prob >= 0.7:
        risk_level = "high"
    elif depressed_prob >= 0.5:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "prediction": prediction,
        "probability": round(depressed_prob * 100, 1),
        "confidence": round(abs(avg_prob - 0.5) * 200, 1),
        "probabilities": {
            "depressed": round(depressed_prob * 100, 1),
            "non_depressed": round((1 - depressed_prob) * 100, 1),
        },
        "signal_type": "proxy_heuristic",
        "model_version": None,
        "features": {"frame_count": len(frames), "analyzed_count": len(frame_results)},
        "explanation": {
            "method": "proxy_fallback",
            "note": (
                "FER library unavailable — brightness/contrast proxy used. "
                "Weak image-statistics heuristic, not a depression measurement."
            ),
        },
        "risk_level": risk_level,
    }
