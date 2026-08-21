"""Phase 17 — Text branch explainability: LIME and SHAP.

Both methods treat the deployed text model as a black-box classifier and
produce word/phrase attributions for display in the UI.

Safety constraints
------------------
- Attributions explain model behaviour, not clinical causality.
- Every result carries a mandatory disclaimer.
- LIME and SHAP attributions are supplementary diagnostic views.
- Attention weights (from Phase 5/6) are shown separately as a secondary
  signal and must NEVER be described as an explanation of the prediction.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

DISCLAIMER = (
    "These word attributions show which tokens most influenced this model prediction. "
    "They explain model behaviour, not clinical causality, "
    "and must not be interpreted as evidence of a mental-health condition."
)

ATTENTION_DISCLAIMER = (
    "Attention weights indicate which tokens the model attended to. "
    "They are a supplementary diagnostic signal, not a causal explanation."
)


# ---------------------------------------------------------------------------
# LIME text explainer
# ---------------------------------------------------------------------------

def explain_text_lime(
    text: str,
    predict_fn: Callable[[list[str]], np.ndarray],
    class_names: Optional[list[str]] = None,
    num_features: int = 10,
    num_samples: int = 500,
    positive_class_idx: int = 1,
) -> dict:
    """Compute LIME word attributions for a single text sample.

    Parameters
    ----------
    text             : Raw input text (normalisation applied inside predict_fn)
    predict_fn       : Callable(list[str]) → (N, n_classes) probability array
    class_names      : e.g. ["control", "depressed"]
    num_features     : Max tokens to include in explanation
    num_samples      : LIME perturbation samples
    positive_class_idx: Class index to explain (default: 1 = positive class)

    Returns
    -------
    dict with token attributions, top tokens, and disclaimer
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:
        raise ImportError("lime is required. pip install lime") from exc

    class_names = class_names or ["class_0", "class_1"]
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=[positive_class_idx],
    )
    raw = explanation.as_list(label=positive_class_idx)
    # Sort by absolute attribution
    sorted_attrs = sorted(raw, key=lambda x: abs(x[1]), reverse=True)
    return {
        "method": "lime",
        "text": text,
        "attributions": [{"token": token, "score": round(float(score), 4)}
                         for token, score in sorted_attrs],
        "top_tokens": [token for token, _ in sorted_attrs[:5]],
        "positive_class": class_names[positive_class_idx],
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# SHAP text explainer (KernelSHAP via masker)
# ---------------------------------------------------------------------------

def explain_text_shap(
    text: str,
    predict_fn: Callable[[list[str]], np.ndarray],
    max_evals: int = 500,
    positive_class_idx: int = 1,
) -> dict:
    """Compute KernelSHAP word attributions for a single text sample.

    Uses the SHAP Text masker (token-level perturbation).

    Parameters
    ----------
    text              : Raw input text
    predict_fn        : Callable(list[str]) → (N, n_classes) array
    max_evals         : Budget for SHAP evaluations
    positive_class_idx: Class column to explain

    Returns
    -------
    dict with per-token SHAP values and disclaimer
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required. pip install shap") from exc

    masker = shap.maskers.Text(r"\s+")
    explainer = shap.Explainer(
        lambda texts: predict_fn(list(texts))[:, positive_class_idx],
        masker,
    )
    shap_values = explainer([text], max_evals=max_evals, silent=True)

    # shap_values.values shape: (1, n_tokens)
    values = np.array(shap_values.values[0], dtype=float).flatten()
    tokens = list(shap_values.data[0])

    pairs = sorted(
        zip(tokens, values.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return {
        "method": "shap",
        "text": text,
        "attributions": [{"token": str(t), "score": round(float(v), 4)} for t, v in pairs],
        "top_tokens": [str(t) for t, _ in pairs[:5]],
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Attention weight formatter (supplementary only)
# ---------------------------------------------------------------------------

def format_attention_weights(
    tokens: list[str],
    weights: "np.ndarray",
    top_k: int = 10,
) -> dict:
    """Format DeBERTa attention weights for UI display.

    Attention weights are shown as a supplementary view only.
    They must never be presented as the primary explanation.

    Parameters
    ----------
    tokens  : Token strings (length T)
    weights : (T,) numpy array — already normalised (sums to 1)
    top_k   : Number of top-attended tokens to highlight

    Returns
    -------
    dict with per-token weights and mandatory disclaimer
    """
    weights = np.array(weights, dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    pairs = sorted(
        zip(tokens, weights.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "type": "attention_weights",
        "all_tokens": [{"token": str(t), "weight": round(float(w), 4)} for t, w in
                       zip(tokens, weights.tolist())],
        "top_k_tokens": [{"token": str(t), "weight": round(float(w), 4)}
                         for t, w in pairs[:top_k]],
        "disclaimer": ATTENTION_DISCLAIMER,
    }
