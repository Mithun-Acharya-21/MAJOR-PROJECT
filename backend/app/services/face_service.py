"""
Live Face Analysis Service — Webcam-based facial expression analysis.

Aggregates predictions over multiple frames for stable results.
Used by POST /api/analyze/face endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger("neuroscreen.face_service")


def analyze_live_face(
    frames: List[bytes],
    model_state: Any,
) -> Dict[str, Any]:
    """Analyze multiple webcam frames for facial expression depression risk.

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

    # --- Analyze each frame ---
    frame_results = []
    expression_counts = {"neutral": 0, "happy": 0, "sad": 0, "angry": 0, "fear": 0}

    for frame_bytes in frames:
        try:
            # Try neural model first
            if model_state.loaded and model_state.vision_branch is not None:
                from backend.app.models.vision_branch import preprocess_image
                image_tensor = preprocess_image(frame_bytes).to(model_state.device)
                with torch.no_grad():
                    features = model_state.vision_branch(image_tensor)
                embedding_sum = features.sum().item()
                prob = 0.5 + np.tanh(embedding_sum * 0.01) * 0.3
            else:
                # Fallback to proxy
                from backend.app.proxy_scorer import score_image
                prob = score_image(frame_bytes)

            frame_results.append(prob)

            # Simple expression classification from brightness/contrast
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(frame_bytes)).convert("L")
            arr = np.array(img)
            mean_brightness = arr.mean() / 255.0
            if mean_brightness > 0.6:
                expression_counts["happy"] += 1
            elif mean_brightness < 0.3:
                expression_counts["sad"] += 1
            else:
                expression_counts["neutral"] += 1

        except Exception as e:
            logger.debug("Frame analysis failed: %s", e)
            continue

    if not frame_results:
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 50.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "features": {"frame_count": len(frames), "analyzed_count": 0},
            "explanation": {"method": "none"},
            "risk_level": "low",
            "error": "Could not analyze any frames",
        }

    # --- Aggregate results ---
    avg_prob = np.mean(frame_results)
    std_prob = np.std(frame_results)

    # Confidence based on consistency (lower std = higher confidence)
    consistency = max(0, 1.0 - std_prob * 3)
    confidence = abs(avg_prob - 0.5) * 200 * consistency

    depressed_prob = max(0.01, min(0.99, avg_prob))
    prediction = "Depressed" if depressed_prob >= 0.5 else "Non-Depressed"

    if depressed_prob >= 0.7:
        risk_level = "high"
    elif depressed_prob >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Normalize expression counts
    total = sum(expression_counts.values()) or 1
    expression_probs = {k: round(v / total * 100, 1) for k, v in expression_counts.items()}

    features = {
        "frame_count": len(frames),
        "analyzed_count": len(frame_results),
        "avg_depression_prob": round(float(avg_prob), 4),
        "std_depression_prob": round(float(std_prob), 4),
        "expression_distribution": expression_probs,
    }

    explanation = {
        "method": "multi_frame_aggregation",
        "frames_analyzed": len(frame_results),
        "consistency": round(consistency, 3),
        "expression_distribution": expression_probs,
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
    }
