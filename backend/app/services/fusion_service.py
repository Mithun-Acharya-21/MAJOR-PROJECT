"""
Multimodal Fusion Service — Combines results from all 4 modalities.

Implements weighted probability fusion with configurable weights.
Used by POST /api/analyze/multimodal endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("neuroscreen.fusion_service")

# Default fusion weights (experimentally justified as baseline)
# Text is weighted highest because it provides the most direct depression signals.
# Voice is second because acoustic markers are well-documented in depression research.
# Photo and live face are weighted lower because facial expression alone is a weaker predictor.
DEFAULT_WEIGHTS = {
    "text": 0.35,
    "photo": 0.15,
    "face": 0.20,
    "voice": 0.30,
}


def fuse_multimodal(
    modality_results: Dict[str, Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Fuse results from multiple modalities into a final prediction.

    Uses weighted probability fusion where each modality contributes
    based on its configured weight. Missing modalities are excluded
    and weights are renormalized.

    Args:
        modality_results: Dict mapping modality name to its analysis result.
            Keys can be: "text", "photo", "face", "voice"
        weights: Optional custom weights. If None, uses DEFAULT_WEIGHTS.

    Returns:
        Final fused result with prediction, probability, individual contributions.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    available = {}
    for modality, result in modality_results.items():
        if result is not None and "probability" in result:
            available[modality] = result

    if not available:
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 50.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "individual_results": {},
            "fusion_weights": {},
            "risk_level": "low",
            "error": "No modality results available for fusion",
        }

    # --- Renormalize weights for available modalities ---
    total_weight = sum(weights.get(m, 0.1) for m in available)
    if total_weight == 0:
        total_weight = 1.0

    normalized_weights = {
        m: weights.get(m, 0.1) / total_weight for m in available
    }

    # --- Weighted probability fusion ---
    fused_depressed = sum(
        available[m]["probability"] / 100.0 * normalized_weights[m]
        for m in available
    )

    fused_depressed = max(0.01, min(0.99, fused_depressed))

    # --- Confidence from agreement and individual confidence ---
    probs = [available[m]["probability"] / 100.0 for m in available]
    agreement = 1.0 - np.std(probs) * 2  # Higher agreement = higher confidence
    avg_confidence = np.mean([
        available[m].get("confidence", 50) / 100.0 for m in available
    ])
    confidence = avg_confidence * max(0.3, agreement) * 100
    confidence = max(5.0, min(95.0, confidence))

    prediction = "Depressed" if fused_depressed >= 0.5 else "Non-Depressed"

    if fused_depressed >= 0.7:
        risk_level = "high"
    elif fused_depressed >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # --- Build individual results summary ---
    individual_results = {}
    for modality, result in available.items():
        individual_results[modality] = {
            "prediction": result.get("prediction", "Unknown"),
            "probability": result.get("probability", 50.0),
            "confidence": result.get("confidence", 50.0),
            "weight": round(normalized_weights[modality], 3),
            "contribution": round(result["probability"] / 100.0 * normalized_weights[modality], 4),
        }

    # --- Build multimodal explanation ---
    contributions = []
    for modality, result in available.items():
        prob = result["probability"]
        weight = normalized_weights[modality]
        contribution = prob / 100.0 * weight
        if prob >= 60:
            level = "Higher risk contribution"
        elif prob >= 45:
            level = "Moderate contribution"
        else:
            level = "Lower risk contribution"
        contributions.append({
            "modality": modality,
            "probability": round(prob, 1),
            "weight": round(weight, 3),
            "contribution": round(contribution, 4),
            "level": level,
        })

    explanation = {
        "method": "weighted_probability_fusion",
        "contributions": contributions,
        "available_modalities": list(available.keys()),
        "weights_used": normalized_weights,
        "agreement_score": round(float(agreement), 3),
    }

    return {
        "prediction": prediction,
        "probability": round(fused_depressed * 100, 1),
        "confidence": round(confidence, 1),
        "probabilities": {
            "depressed": round(fused_depressed * 100, 1),
            "non_depressed": round((1 - fused_depressed) * 100, 1),
        },
        "individual_results": individual_results,
        "fusion_weights": normalized_weights,
        "explanation": explanation,
        "risk_level": risk_level,
    }
