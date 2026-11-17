"""
LIME Explainer — Local Interpretable Model-agnostic Explanations for text.

Provides token-level attribution explanations for text predictions,
highlighting which words push the model toward or away from depression.

Exports:
  explain_text_lime — generate LIME explanation for a text prediction
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("emotiondep.explainability.lime")


def explain_text_lime(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    num_features: int = 10,
    num_samples: int = 100,
) -> List[Dict[str, Any]]:
    """Generate a LIME explanation for a text prediction.

    Args:
        text: Input text to explain.
        predict_fn: Function that takes a list of strings and returns
                    an (N, 2) array of [non_depressed_prob, depressed_prob].
        num_features: Number of top features to return.
        num_samples: Number of perturbed samples to generate.

    Returns:
        List of dicts with 'token', 'weight', 'color' keys.
    """
    try:
        import lime
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=["Non-Depressed", "Depressed"])

        explanation = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

        # Extract word attributions
        word_weights = explanation.as_list(label=1)  # Class 1 = Depressed

        result = []
        for word, weight in word_weights:
            result.append({
                "token": word,
                "weight": round(float(weight), 4),
                "color": _weight_to_color(weight),
            })

        return result

    except ImportError:
        logger.warning("LIME not installed. Using attention-based fallback.")
        return _fallback_explanation(text)


def _weight_to_color(weight: float) -> str:
    """Map a LIME weight to a color string."""
    # Positive weight = pushes toward depression (red)
    # Negative weight = pushes away (green)
    normalized = max(0.0, min(1.0, abs(weight) * 2))
    if weight > 0:
        r = int(239 * normalized + 245 * (1 - normalized))
        g = int(68 * normalized + 158 * (1 - normalized))
        b = int(68 * normalized + 11 * (1 - normalized))
    else:
        r = int(34 * normalized + 245 * (1 - normalized))
        g = int(197 * normalized + 158 * (1 - normalized))
        b = int(94 * normalized + 11 * (1 - normalized))
    return f"#{r:02x}{g:02x}{b:02x}"


def _fallback_explanation(text: str) -> List[Dict[str, Any]]:
    """Simple word-level explanation when LIME is unavailable."""
    risk_words = {
        "hopeless", "worthless", "empty", "exhausted", "numb", "lonely",
        "anxious", "overwhelmed", "crying", "tired", "depressed", "pain",
        "sad", "lost", "dark", "hurt", "pointless", "failure", "miserable",
    }
    protective_words = {
        "happy", "grateful", "excited", "joy", "love", "hope", "better",
        "progress", "good", "calm", "friends", "support",
    }

    words = text.lower().split()
    result = []
    for word in words:
        clean = word.strip(".,!?;:'\"")
        if clean in risk_words:
            result.append({"token": word, "weight": 0.8, "color": "#ef4444"})
        elif clean in protective_words:
            result.append({"token": word, "weight": -0.6, "color": "#22c55e"})
        else:
            result.append({"token": word, "weight": 0.0, "color": "#f59e0b"})

    return result
