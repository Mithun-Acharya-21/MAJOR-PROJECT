"""
Photo Analysis Service — Static photo depression screening.

Provides photo-only analysis with Grad-CAM explanations.
Used by POST /api/analyze/photo endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("neuroscreen.photo_service")


def analyze_photo(
    image_bytes: bytes,
    model_state: Any,
) -> Dict[str, Any]:
    """Run photo-only analysis through the ResNet-18 vision pipeline.

    Args:
        image_bytes: Raw image file bytes.
        model_state: ModelState instance with loaded models.

    Returns:
        Structured result dict with prediction, probabilities, explanation.
    """
    from backend.app.proxy_scorer import score_image

    # --- Validate image ---
    if not image_bytes or len(image_bytes) < 100:
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 50.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "features": {},
            "explanation": {"method": "none"},
            "risk_level": "low",
            "error": "Invalid or empty image data",
        }

    # --- Check for face detection ---
    face_detected = False
    try:
        import cv2
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        face_detected = len(faces) > 0
    except Exception:
        face_detected = True  # Assume face if detection fails

    # --- Proxy scoring (always available) ---
    proxy_prob = score_image(image_bytes)

    # --- Neural model scoring (if loaded) ---
    neural_prob = None
    image_explanation = None

    if model_state.loaded and model_state.vision_branch is not None:
        try:
            from backend.app.models.vision_branch import preprocess_image, generate_gradcam
            from backend.app.explainability.explainer import generate_image_explanation

            image_tensor = preprocess_image(image_bytes).to(model_state.device)

            with torch.no_grad():
                vision_features = model_state.vision_branch(image_tensor)

            # Generate Grad-CAM heatmap
            gradcam_heatmap = generate_gradcam(model_state.vision_branch, image_tensor)

            # Generate explanation images
            image_explanation = generate_image_explanation(
                original_image_bytes=image_bytes,
                gradcam_heatmap=gradcam_heatmap,
            )

            # Use embedding sum as rough probability proxy
            embedding_sum = vision_features.sum().item()
            neural_prob = 0.5 + np.tanh(embedding_sum * 0.01) * 0.3

        except Exception as e:
            logger.warning("Neural photo analysis failed, using proxy only: %s", e)

    # --- Blend probabilities ---
    if neural_prob is not None:
        depressed_prob = 0.4 * neural_prob + 0.6 * proxy_prob
    else:
        depressed_prob = proxy_prob

    depressed_prob = max(0.01, min(0.99, depressed_prob))
    confidence = abs(depressed_prob - 0.5) * 200

    prediction = "Depressed" if depressed_prob >= 0.5 else "Non-Depressed"

    if depressed_prob >= 0.7:
        risk_level = "high"
    elif depressed_prob >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # --- Build features ---
    features = {
        "face_detected": face_detected,
        "image_size_bytes": len(image_bytes),
    }

    # --- Build explanation ---
    explanation = {
        "method": "grad_cam" if image_explanation else "proxy_heuristic",
        "face_detected": face_detected,
        "gradcam": image_explanation.get("overlay") if image_explanation else None,
        "heatmap": image_explanation.get("heatmap") if image_explanation else None,
    }

    result = {
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

    if not face_detected:
        result["warning"] = "No face detected in the image. Results may be unreliable."

    return result
