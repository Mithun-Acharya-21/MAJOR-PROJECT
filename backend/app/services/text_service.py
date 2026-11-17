"""
Text Analysis Service — honest priority chain (Phase 4 / 17).

Priority:
  1. TRAINED text model (models/final/text_model) — for CHINESE text, i.e.
     the language distribution the model was actually trained and evaluated
     on (EATD). probability = sigmoid(logit) of the classification head.
     Calibrated with models/calibration.json when fitted (valid: calibration
     was fitted for this model's outputs on validation data).
  2. FALLBACK for other languages (e.g. the English screening interview):
     lexicon proxy score, explicitly labeled ``method: "lexicon_proxy"``.
     This is a RULE-BASED heuristic — it is never presented as a trained
     model probability, and no model_version of the trained model is
     claimed for it.

LIME (Phase 17) always explains the ACTUAL prediction function that produced
the probability — the trained model when used, the lexicon proxy otherwise.

Exports:
    analyze_text — text → structured result dict
    detect_script — script detection used for language gating
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("emotiondep.text_service")

# Characters in the CJK unified ideograph blocks (Chinese-distribution gate)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def detect_script(text: str) -> str:
    """Return 'chinese' when CJK ideographs dominate, else 'non-chinese'."""
    if not text:
        return "non-chinese"
    cjk = len(_CJK_RE.findall(text))
    total = len(re.findall(r"\S", text))
    if total == 0:
        return "non-chinese"
    return "chinese" if cjk / total >= 0.15 else "non-chinese"


# ---------------------------------------------------------------------------
# Lexicon proxy (rule-based) — clearly labeled, never presented as a model
# ---------------------------------------------------------------------------

from backend.app.proxy_scorer import score_text  # noqa: E402


def _finalize(
    depressed_prob: float,
    method: str,
    explanation: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    depressed_prob = max(0.01, min(0.99, float(depressed_prob)))
    confidence = abs(depressed_prob - 0.5) * 200
    prediction = "Depressed" if depressed_prob >= 0.5 else "Non-Depressed"
    if depressed_prob >= 0.7:
        risk_level = "high"
    elif depressed_prob >= 0.5:
        risk_level = "moderate"
    else:
        risk_level = "low"
    result = {
        "prediction": prediction,
        "probability": round(depressed_prob * 100, 1),
        "confidence": round(confidence, 1),
        "probabilities": {
            "depressed": round(depressed_prob * 100, 1),
            "non_depressed": round((1 - depressed_prob) * 100, 1),
        },
        "method": method,
        "explanation": explanation,
        "risk_level": risk_level,
    }
    if extra:
        result.update(extra)
    return result


def _lexicon_result(cleaned: str) -> Dict[str, Any]:
    lexicon_prob, word_scores = score_text(cleaned)
    explanation = {
        "method": "lexicon_proxy",
        "note": (
            "Rule-based keyword proxy score — NOT a trained model probability. "
            "Used because no trained model covers this input language."
        ),
        "top_features": [
            {
                "feature": tok,
                "weight": round(abs(w), 4),
                "direction": "risk_increasing" if w > 0 else "risk_decreasing",
            }
            for tok, w in sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        ],
    }
    return _finalize(lexicon_prob, "lexicon_proxy", explanation, {
        "model_version": None,
        "score_type": "rule_based_proxy",
        # Explicit provenance label consumed by fusion_service: this signal
        # must never be presented as a trained-model probability.
        "signal_type": "lexicon_proxy",
    })


def _trained_result(clf, cleaned: str) -> Dict[str, Any]:
    out = clf.predict(cleaned)
    raw_prob = out["probability"]

    # Calibrated probability when a fitted calibration artifact exists.
    calibration_method = "identity"
    prob = raw_prob
    try:
        from backend.app.services.calibration_service import load_calibration_model
        cal = load_calibration_model()
        if cal is not None:
            prob = cal.transform(raw_prob)
            calibration_method = cal.method
    except Exception as e:
        logger.warning("Calibration application failed: %s", e)

    attention = clf.attention_explanation(cleaned)
    explanation: Dict[str, Any] = {
        "method": "trained_model",
        "model_version": clf.model_version,
        "model_feature_attribution": {
            "label": (
                "Text features that influenced the model's prediction "
                "(attention weights — model feature attribution, not causal explanation)"
            ),
            "tokens": attention[:30],
        },
        "calibration_method": calibration_method,
    }

    # LIME on the ACTUAL trained prediction function (Phase 17)
    lime_explanation: List[Dict[str, Any]] = []
    try:
        from backend.app.explainability.lime_text import explain_text_with_model
        lime_explanation = explain_text_with_model(cleaned, clf.predict_proba_batch)
        explanation["lime"] = {
            "label": (
                "Model feature attribution (LIME) computed on the trained "
                "classification model — not a causal explanation"
            ),
            "features": lime_explanation,
        }
    except Exception as e:
        logger.debug("LIME on trained model unavailable: %s", e)

    extra = {
        "model_version": clf.model_version,
        "score_type": "trained_model_probability",
        # Fusion must be able to see that this signal came from a verified,
        # trained checkpoint (and therefore may be calibrated).
        "signal_type": "trained_model_probability",
        "raw_probability": round(raw_prob * 100, 1),
        "calibrated_probability": round(prob * 100, 1),
        "calibration_method": calibration_method,
    }
    result = _finalize(prob, "trained_model", explanation, extra)

    # Standard explanation fields consumed by the existing frontend mapper:
    # they are populated from the ACTUAL trained-model attribution output.
    if lime_explanation:
        result["explanation"]["top_features"] = [
            {
                "feature": f["feature"],
                "weight": f["weight"],
                "direction": f["direction"],
            }
            for f in lime_explanation
        ]
    elif attention:
        result["explanation"]["attention_tokens"] = [
            {"token": t["token"], "weight": t["weight"]} for t in attention[:15]
        ]
    return result


def analyze_text(
    text: str,
    model_state: Any = None,
) -> Dict[str, Any]:
    """Run text analysis through the honest priority chain.

    Note on ``model_state``: kept for backward compatibility with existing
    callers/endpoints, but the trained classifier is loaded via
    ``get_text_classifier()`` — a randomly initialized neural branch would
    be a meaningless probability source, so it is intentionally NOT used.
    """
    from backend.app.models.text_branch import preprocess_text

    cleaned = preprocess_text(text or "")
    if not cleaned.strip():
        return _finalize(
            0.5,
            "none",
            {"method": "none", "note": "Empty or invalid text input"},
            {"error": "Empty or invalid text input"},
        )

    script = detect_script(cleaned)

    # Priority 1: the trained model — only for its training language.
    if script == "chinese":
        from backend.app.models.text_classifier import get_text_classifier
        clf = get_text_classifier()
        if clf is not None and clf.available:
            result = _trained_result(clf, cleaned)
            result["input_language_gate"] = "chinese: trained EATD model applied"
            return result
        # No trained model at all → labeled proxy fallback
        result = _lexicon_result(cleaned)
        result["input_language_gate"] = (
            "chinese: no trained model checkpoint available — proxy score used"
        )
        return result

    # Priority 2: non-Chinese input → the trained EATD model would be out of
    # distribution; honestly fall back to the labeled lexicon proxy.
    result = _lexicon_result(cleaned)
    result["input_language_gate"] = (
        "non-chinese: trained EATD model is Chinese-distribution — proxy score used"
    )
    return result
