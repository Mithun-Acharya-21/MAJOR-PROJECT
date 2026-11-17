"""
Multimodal Fusion Service — Combines results from all 4 modalities.

Implements weighted probability fusion with configurable weights.
Used by POST /api/analyze/multimodal endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("emotiondep.fusion_service")

# Default fusion weights (experimentally justified as baseline)
# Text is weighted highest because it provides the most direct depression signals.
# Voice is second because acoustic markers are well-documented in depression research.
# Photo and live face are weighted lower because facial expression alone is a weaker predictor.
# The questionnaire is a structured-interview signal with moderate weight.
DEFAULT_WEIGHTS = {
    "text": 0.35,
    "photo": 0.15,
    "face": 0.20,
    "voice": 0.30,
    "questionnaire": 0.40,
}

# Weights used when the questionnaire is part of the screening flow —
# the structured interview is the most direct symptom-level signal.
SCREENING_WEIGHTS = {
    "questionnaire": 0.40,
    "text": 0.25,
    "voice": 0.20,
    "face": 0.10,
    "photo": 0.05,
}


def fuse_multimodal(
    modality_results: Dict[str, Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Fuse results from multiple modalities into a final prediction.

    Uses weighted probability fusion where each modality contributes
    based on its configured weight. Missing modalities are EXCLUDED
    (no fake predictions are inserted) and weights are renormalized over
    only the available modalities, e.g.:

        text=0.35, voice=0.30, face=0.20, photo=0.15
        face & photo unavailable  →  text: 0.35/0.65, voice: 0.30/0.65

    Args:
        modality_results: Dict mapping modality name to its analysis result.
            Keys can be: "text", "photo", "face", "voice", "questionnaire".
            The "questionnaire" key carries the structured screening-interview
            signal (probability + confidence) produced by the screening service.
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

    # --- Phase 11: signal-type transparency ---
    # Only signals explicitly marked 'trained_model_probability' come from a
    # model trained AND evaluated for depression classification. Everything
    # else (expression scores, acoustic heuristics, questionnaire mapping) is
    # auxiliary. The fusion output labels itself accordingly — it is an
    # EXPERIMENTAL HEURISTIC FUSION, never a clinically validated model.
    signal_types = {
        m: available[m].get("signal_type", "rule_based_screening_score")
        for m in available
    }
    validated_modalities = [
        m for m, st in signal_types.items() if st == "trained_model_probability"
    ]

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
    elif fused_depressed >= 0.5:
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
            "signal_type": signal_types.get(modality, "rule_based_screening_score"),
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
        "signal_types": signal_types,
        "validated_modalities": validated_modalities,
        "interpretation_note": (
            "Experimental heuristic fusion of available modality signals. "
            "Only signals with signal_type 'trained_model_probability' come "
            "from models trained and evaluated for depression classification "
            "(see results/metrics/text_metrics.json). All other signals are "
            "auxiliary expression/acoustic/questionnaire scores. This fusion "
            "is NOT clinically validated and must not be reported as "
            "validated accuracy."
        ),
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
        "fusion_type": "experimental_heuristic_fusion",
        "is_trained_model": False,
        "is_clinical_diagnosis": False,
        "signal_types": signal_types,
        "validated_modalities": validated_modalities,
        "explanation": explanation,
        "risk_level": risk_level,
        # Exposed for the uncertainty service: per-modality probabilities + agreement
        "_fusion_meta": {
            "agreement_score": float(agreement),
            "modality_probabilities": {m: available[m]["probability"] / 100.0 for m in available},
            "n_modalities": len(available),
        },
    }
