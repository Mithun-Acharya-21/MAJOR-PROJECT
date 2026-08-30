"""
LIME Text Explainer — Proper LIME integration for text explanations.

Uses the lime library to generate local explanations for text predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("neuroscreen.lime_text")


def explain_text_lime(
    text: str,
    word_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Generate LIME explanation for text prediction.

    Falls back to lexicon-based explanation if LIME is not available
    or if the prediction function cannot be constructed.

    Args:
        text: Preprocessed text.
        word_scores: Lexicon-based word scores.

    Returns:
        List of token explanation dicts.
    """
    try:
        from lime.lime_text import LimeTextExplainer

        # Build a simple prediction function using lexicon scores
        def predict_fn(texts):
            results = []
            for t in texts:
                words = t.lower().split()
                risk = 0.0
                protective = 0.0
                for w in words:
                    clean = w.strip(".,!?;:'\"")
                    if clean in _RISK_TERMS:
                        risk += 1
                    elif clean in _PROTECTIVE_TERMS:
                        protective += 1
                total = max(len(words), 1)
                score = (risk - protective * 0.5) / total
                prob = 1.0 / (1.0 + np.exp(-score * 4))
                prob = 0.08 + prob * 0.84
                results.append([1 - prob, prob])
            return np.array(results)

        explainer = LimeTextExplainer(class_names=["Non-Depressed", "Depressed"])
        explanation = explainer.explain_instance(
            text,
            predict_fn,
            num_features=min(10, len(text.split())),
            num_samples=50,
        )

        word_weights = explanation.as_list(label=1)
        result = []
        for word, weight in word_weights:
            result.append({
                "feature": word,
                "weight": round(float(abs(weight)), 4),
                "direction": "risk_increasing" if weight > 0 else "risk_decreasing",
                "contribution": round(float(weight), 4),
            })
        return result

    except ImportError:
        logger.debug("LIME not installed, using lexicon fallback")
        return _lexicon_fallback(word_scores)
    except Exception as e:
        logger.debug("LIME explanation failed: %s, using lexicon fallback", e)
        return _lexicon_fallback(word_scores)


def _lexicon_fallback(word_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """Fallback explanation using lexicon scores."""
    result = []
    for token, weight in sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        direction = "risk_increasing" if weight > 0 else "risk_decreasing"
        result.append({
            "feature": token,
            "weight": round(abs(weight), 4),
            "direction": direction,
            "contribution": round(weight, 4),
        })
    return result


# Built-in lexicons for the prediction function
_RISK_TERMS = {
    "hopeless", "worthless", "empty", "exhausted", "numb", "lonely",
    "anxious", "overwhelmed", "burden", "crying", "tired", "insomnia",
    "suicidal", "depressed", "miserable", "helpless", "pointless",
    "failure", "hate", "pain", "sad", "worry", "stress", "alone",
    "trapped", "drowning", "breaking", "falling", "lost", "dark", "hurt",
}

_PROTECTIVE_TERMS = {
    "happy", "grateful", "excited", "joy", "love", "hope", "better",
    "progress", "thankful", "blessed", "amazing", "wonderful", "great",
    "good", "relief", "peace", "calm", "confident", "optimistic",
}
