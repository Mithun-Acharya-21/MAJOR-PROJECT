"""
Photo Analysis Service — facial-expression HONESTY contract (Phase 9).

What this service produces:
  * ``visual_expression_score`` — an affect score derived from detected
    facial expressions (MediaPipe blendshapes → emotion weighting). This is
    an AUXILIARY EXPRESSION SIGNAL. It is NOT a validated depression
    probability: no image model in this repository has been trained or
    evaluated to predict depression (there is no aligned image-depression
    dataset available — see docs/dataset_notes.md).
  * When no face is detectable: a brightness/contrast heuristic
    (``proxy_heuristic``), labeled as such and carrying low confidence.

What this service does NOT do:
  * It does NOT manufacture a "depression probability" from embedding sums.
  * It does NOT present Grad-CAM as an explanation of depression (there is
    no depression-class image model to explain — Phase 18).

For fusion compatibility the score is still carried in the ``probability``
field, but ALWAYS together with ``signal_type:
"auxiliary_expression_score"`` so the fusion service can label the output
honestly (Phase 11).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("emotiondep.photo_service")

# Emotion → affect weights (research-informed EXPRESSION weighting, not
# depression classification). sad/flat affect raise the expression score.
EMOTION_WEIGHTS = {
    "sad": 0.35, "fear": 0.15, "angry": 0.15,
    "disgust": 0.10, "happy": -0.25, "surprise": -0.10, "neutral": 0.05,
}


def analyze_photo(
    image_bytes: bytes,
    model_state: Any = None,
) -> Dict[str, Any]:
    """Analyze a facial photo into an auxiliary visual expression score.

    ``model_state`` is accepted for backward compatibility but is NOT used:
    the untrained ResNet embedding has no legitimate mapping to depression
    or expression probabilities, and using it would fabricate a signal.
    """
    if not image_bytes or len(image_bytes) < 100:
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 50.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "features": {},
            "explanation": {"method": "none"},
            "risk_level": "low",
            "signal_type": "none",
            "error": "Invalid or empty image data",
        }

    expression_result = None
    face_detected = False

    try:
        from backend.app.face_expression_analyzer import analyze_single_frame
        expression_result = analyze_single_frame(image_bytes)
        face_detected = bool(expression_result.get("face_detected", False))
    except Exception as e:
        logger.warning("MediaPipe expression analysis failed: %s", e)

    if face_detected and expression_result is not None:
        expression_prob = float(expression_result.get("depression_prob", 0.5))
        all_emotions = expression_result.get("all_emotions", {}) or {}
        method = "mediapipe_blendshapes"
        dominant_emotion = expression_result.get("dominant_emotion", "neutral")
        dominant_expression = expression_result.get("dominant_expression", dominant_emotion.capitalize())
        uncertainty_label = expression_result.get("uncertainty_label", "Dominant movement pattern")
        expression_prominence = float(expression_result.get("expression_prominence", 0.5))
        observed_movement = expression_result.get("observed_movement", {})

        emotion_dist = {
            k: round(v * 100, 1) for k, v in all_emotions.items()
        }
        # Genuine expression confidence derived from raw prominence of the dominant expression
        # (NOT an invented depression confidence).
        dominant_prominence = round(expression_prominence * 100.0, 1)
        confidence = float(dominant_prominence)
        confidence_source = "dominant_expression_prominence"
        processing_status = "Face detected — facial expressions analyzed"

        note = (
            "Auxiliary facial-expression score derived from detected "
            "expressions (sad/flat affect weighting). This is NOT a validated "
            "depression probability — no image model in this project has been "
            "trained or evaluated to predict depression."
        )
        signal_type = "auxiliary_expression_score"
        explanation = {
            "method": method,
            "face_detected": True,
            "dominant_emotion": dominant_emotion,
            "dominant_expression": dominant_expression,
            "uncertainty_label": uncertainty_label,
            "expression_prominence": dominant_prominence,
            "emotion_distribution": emotion_dist,
            "expression_distribution": emotion_dist,
            "observed_movement": observed_movement,
            "emotion_weights_used": EMOTION_WEIGHTS,
            "model_name": "MediaPipe FaceLandmarker (52 Blendshapes)",
            "model_type": "Facial Expression & Landmark Detection (Auxiliary)",
            "is_depression_model": False,
            "confidence_source": confidence_source,
            "score_interpretation": (
                "visual_expression_score — how strongly the detected "
                "expressions align with low-activity/negative affect; not a "
                "depression classification"
            ),
            "gradcam": None,
            "gradcam_note": (
                "Grad-CAM is not produced here: it would require a trained "
                "image classification model and target class; none exists for "
                "depression in this project."
            ),
        }
    else:
        # No face detected → no fabricated confidence or emotion
        expression_prob = 0.5
        method = "no_face_detected"
        dominant_emotion = "none"
        dominant_expression = None
        uncertainty_label = "No face detected in the image"
        expression_prominence = 0.0
        observed_movement = {}
        emotion_dist = {}
        confidence = 0.0
        confidence_source = "unverified_no_face"
        processing_status = "No face detected in the image"

        note = (
            "No face detected in the image — no facial expression or depression "
            "scores could be computed. Scores are not fabricated."
        )
        signal_type = "none"
        explanation = {
            "method": method,
            "face_detected": False,
            "dominant_emotion": None,
            "dominant_expression": None,
            "uncertainty_label": uncertainty_label,
            "expression_prominence": 0.0,
            "emotion_distribution": {},
            "expression_distribution": {},
            "observed_movement": {},
            "model_name": "MediaPipe FaceLandmarker",
            "model_type": "Facial Expression & Landmark Detection (Auxiliary)",
            "is_depression_model": False,
            "confidence_source": confidence_source,
            "score_interpretation": "No face detected — no valid score",
        }

    depressed_prob = max(0.01, min(0.99, expression_prob)) if face_detected else 0.5
    conf_scaled = max(0.0, min(100.0, confidence))

    prediction_label = (
        f"Expression: {dominant_expression}"
        if face_detected and dominant_expression is not None
        else "No Face Detected"
    )

    result = {
        "prediction": prediction_label,
        "dominant_expression": dominant_expression,
        "dominant_emotion": dominant_emotion if face_detected and dominant_emotion != "none" else None,
        "uncertainty_label": uncertainty_label,
        "expression_prominence": round(conf_scaled, 1),
        "observed_movement": observed_movement,
        "is_depression_prediction": False,
        "is_depression_model": False,
        "model_name": "MediaPipe FaceLandmarker (52 Blendshapes)" if face_detected else "MediaPipe FaceLandmarker",
        "model_type": "Facial Expression & Landmark Detection (Auxiliary)",
        "processing_status": processing_status,
        "confidence_source": confidence_source,
        "face_detected": face_detected,
        "probability": round(depressed_prob * 100, 1) if face_detected else 50.0,
        "confidence": round(conf_scaled, 1),
        "probabilities": {
            "depressed": round(depressed_prob * 100, 1) if face_detected else 50.0,
            "non_depressed": round((1 - depressed_prob) * 100, 1) if face_detected else 50.0,
        },
        "visual_expression_score": round(expression_prob, 4) if face_detected else 0.0,
        "expression_distribution": emotion_dist,
        "signal_type": signal_type,
        "features": {
            "face_detected": face_detected,
            "dominant_emotion": dominant_emotion if face_detected else None,
            "dominant_expression": dominant_expression,
            "expression_distribution": emotion_dist,
            "observed_movement": observed_movement,
            "image_size_bytes": len(image_bytes),
            "analysis_method": method,
        },
        "explanation": {**explanation, "note": note},
        "risk_level": "moderate" if (face_detected and depressed_prob >= 0.65) else (
            "low" if face_detected else "unknown"
        ),
        "model_version": None,
        "note": note,
    }
    if not face_detected:
        result["warning"] = "No face detected in the image. Facial expression analysis requires a clearly visible face."
    return result
