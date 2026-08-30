"""
SHAP Explainer — SHapley Additive exPlanations for model interpretability.

Provides global and local explanations using SHAP values.

Exports:
  explain_text_shap — generate SHAP explanation for text predictions
  compute_feature_importance — aggregate SHAP values across samples
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("neuroscreen.explainability.shap")


def explain_text_shap(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    background_texts: Optional[List[str]] = None,
    num_features: int = 10,
) -> List[Dict[str, Any]]:
    """Generate a SHAP explanation for a text prediction.

    Args:
        text: Input text to explain.
        predict_fn: Function that takes a list of strings and returns
                    an (N, 2) array of [non_depressed_prob, depressed_prob].
        background_texts: Background dataset for SHAP computation.
                          If None, uses a small default set.
        num_features: Number of top features to display.

    Returns:
        List of dicts with 'token', 'weight', 'color' keys.
    """
    try:
        import shap

        if background_texts is None:
            background_texts = [
                "I feel great today and everything is wonderful",
                "I am so sad and nothing matters anymore",
                "The weather is nice and I went for a walk",
                "I cannot stop crying and feel hopeless",
            ]

        explainer = shap.Explainer(
            lambda x: predict_fn(list(x)),
            shap.maskers.Text(),
        )

        shap_values = explainer([text])

        # Extract token-level SHAP values for the "depressed" class
        tokens = shap_values[0].data
        values = shap_values[0].values

        if isinstance(values, list) and len(values) > 1:
            # Use class 1 (depressed) SHAP values
            values = values[1] if len(values) > 1 else values[0]

        # Sort by absolute value
        indices = np.argsort(np.abs(values))[::-1][:num_features]

        result = []
        for idx in indices:
            if idx < len(tokens):
                token = tokens[idx] if isinstance(tokens[idx], str) else str(tokens[idx])
                weight = float(values[idx])
                result.append({
                    "token": token,
                    "weight": round(weight, 4),
                    "color": _shap_weight_to_color(weight),
                })

        return result

    except ImportError:
        logger.warning("SHAP not installed. Using attention-based fallback.")
        return []


def _shap_weight_to_color(weight: float) -> str:
    """Map a SHAP weight to a color string."""
    normalized = max(0.0, min(1.0, abs(weight) * 3))
    if weight > 0:
        r = int(239 * normalized + 245 * (1 - normalized))
        g = int(68 * normalized + 158 * (1 - normalized))
        b = int(68 * normalized + 11 * (1 - normalized))
    else:
        r = int(34 * normalized + 245 * (1 - normalized))
        g = int(197 * normalized + 158 * (1 - normalized))
        b = int(94 * normalized + 11 * (1 - normalized))
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_feature_importance(
    texts: List[str],
    predict_fn: Callable[[List[str]], np.ndarray],
    top_n: int = 20,
) -> Dict[str, float]:
    """Compute global feature importance across a dataset.

    Args:
        texts: List of input texts.
        predict_fn: Model prediction function.
        top_n: Number of top features to return.

    Returns:
        Dict mapping tokens to their mean absolute SHAP importance.
    """
    try:
        import shap

        explainer = shap.Explainer(
            lambda x: predict_fn(list(x)),
            shap.maskers.Text(),
        )

        shap_values = explainer(texts[:50])  # Limit for performance

        # Aggregate importance across all samples
        importance: Dict[str, float] = {}
        for sv in shap_values:
            tokens = sv.data
            values = sv.values
            if isinstance(values, list) and len(values) > 1:
                values = values[1]
            for token, val in zip(tokens, values):
                if isinstance(token, str):
                    token = token.strip()
                    if token:
                        importance[token] = importance.get(token, 0) + abs(float(val))

        # Sort and return top N
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:top_n])

    except ImportError:
        logger.warning("SHAP not installed. Cannot compute feature importance.")
        return {}
