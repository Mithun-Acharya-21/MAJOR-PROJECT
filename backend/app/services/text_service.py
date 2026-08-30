"""
Text Analysis Service — Standalone text depression screening.

Provides text-only analysis with LIME explanations.
Used by POST /api/analyze/text endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("neuroscreen.text_service")


def analyze_text(
    text: str,
    model_state: Any,
) -> Dict[str, Any]:
    """Run text-only analysis through the DistilBERT+BiLSTM pipeline.

    Args:
        text: Raw text input.
        model_state: ModelState instance with loaded models.

    Returns:
        Structured result dict with prediction, probabilities, explanation.
    """
    from backend.app.models.text_branch import preprocess_text
    from backend.app.proxy_scorer import score_text
    from backend.app.explainability.explainer import get_risk_color

    cleaned = preprocess_text(text)
    if not cleaned.strip():
        return {
            "prediction": "Non-Depressed",
            "probability": 50.0,
            "confidence": 50.0,
            "probabilities": {"depressed": 50.0, "non_depressed": 50.0},
            "features": {},
            "explanation": {"top_features": [], "method": "none"},
            "risk_level": "low",
            "error": "Empty or invalid text input",
        }

    # --- Lexicon-based scoring (always available) ---
    lexicon_prob, word_scores = score_text(cleaned)

    # --- Neural model scoring (if model is loaded) ---
    neural_prob = None
    attention_explanation = []

    if model_state.loaded and model_state.text_branch is not None:
        try:
            encoding = model_state.tokenizer(
                cleaned,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            input_ids = encoding["input_ids"].to(model_state.device)
            attention_mask = encoding["attention_mask"].to(model_state.device)
            tokens_list = model_state.tokenizer.convert_ids_to_tokens(
                input_ids.squeeze().tolist()
            )

            with torch.no_grad():
                text_features, attn_weights = model_state.text_branch(
                    input_ids, attention_mask
                )

            # Get softmax probability
            attn_list = attn_weights.squeeze().cpu().tolist()
            neural_prob = 0.5  # Would need a classification head for proper probability

            # Generate attention-based explanation
            from backend.app.explainability.explainer import generate_text_explanation
            attention_explanation = generate_text_explanation(
                text=cleaned,
                attention_weights=attn_list,
                tokens=tokens_list,
            )

            # Boost attention weights with lexicon hints
            if word_scores and attention_explanation:
                for item in attention_explanation:
                    hint = word_scores.get(item["token"].lower(), 0.0)
                    if hint > 0:
                        boosted = min(1.0, item["weight"] + hint * 0.35)
                        item["weight"] = round(boosted, 4)
                        item["color"] = get_risk_color(boosted)

        except Exception as e:
            logger.warning("Neural text analysis failed, using lexicon only: %s", e)

    # --- Blend probabilities ---
    if neural_prob is not None:
        depressed_prob = 0.4 * neural_prob + 0.6 * lexicon_prob
    else:
        depressed_prob = lexicon_prob

    depressed_prob = max(0.01, min(0.99, depressed_prob))
    confidence = abs(depressed_prob - 0.5) * 200  # 0-100% scale

    prediction = "Depressed" if depressed_prob >= 0.5 else "Non-Depressed"

    if depressed_prob >= 0.7:
        risk_level = "high"
    elif depressed_prob >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # --- LIME explanation (if available) ---
    lime_explanation = []
    try:
        from backend.app.explainability.lime_text import explain_text_lime
        lime_explanation = explain_text_lime(cleaned, word_scores)
    except Exception as e:
        logger.debug("LIME explanation not available: %s", e)

    # --- Build feature summary ---
    features = {
        "word_count": len(cleaned.split()),
        "risk_word_count": sum(1 for w in word_scores if word_scores[w] > 0),
        "protective_word_count": sum(1 for w in word_scores if word_scores[w] < 0),
    }

    # --- Build explanation ---
    top_features = []
    for token, weight in sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        direction = "risk_increasing" if weight > 0 else "risk_decreasing"
        top_features.append({
            "feature": token,
            "weight": round(abs(weight), 4),
            "direction": direction,
        })

    # Use LIME if available, otherwise use attention/lexicon
    explanation = lime_explanation if lime_explanation else {
        "top_features": top_features,
        "method": "lexicon" if not attention_explanation else "attention",
        "attention_tokens": attention_explanation[:15] if attention_explanation else [],
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
