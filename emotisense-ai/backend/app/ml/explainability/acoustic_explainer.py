"""Phase 17 — Voice branch explainability: acoustic feature attribution.

Provides two complementary views:
1. SHAP KernelExplainer over acoustic summary statistics
   (tabular, one row per sample)
2. Permutation importance (already in Phase 10 for MFCC rows;
   this module wraps it for the API layer with UI-ready formatting)

All results carry a mandatory disclaimer.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

DISCLAIMER = (
    "Acoustic feature attributions show which signal characteristics most "
    "influenced this model prediction. "
    "They are model-observed patterns, not clinical evidence, "
    "and must not be interpreted as proof of a mental-health condition."
)


# ---------------------------------------------------------------------------
# SHAP over acoustic summary features
# ---------------------------------------------------------------------------

def explain_acoustic_shap(
    feature_vector: np.ndarray,
    feature_names: list[str],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: Optional[np.ndarray] = None,
    max_evals: int = 200,
    positive_class_idx: int = 1,
) -> dict:
    """KernelSHAP attribution over a 1-D acoustic feature vector.

    Parameters
    ----------
    feature_vector    : (n_features,) float array of acoustic summary stats
    feature_names     : One name per feature
    predict_fn        : Callable((N, n_features)) → (N, n_classes) array
    background        : (M, n_features) background dataset for SHAP;
                        if None, uses zero baseline
    max_evals         : SHAP evaluation budget
    positive_class_idx: Column of predict_fn output to explain

    Returns
    -------
    dict with per-feature attributions sorted by absolute SHAP value
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required. pip install shap") from exc

    x = feature_vector.reshape(1, -1).astype(np.float32)
    bg = background if background is not None else np.zeros_like(x)

    explainer = shap.KernelExplainer(
        lambda arr: predict_fn(arr.astype(np.float32))[:, positive_class_idx],
        bg,
    )
    shap_values = explainer.shap_values(x, nsamples=max_evals, silent=True)
    values = np.array(shap_values).squeeze()

    pairs = sorted(
        zip(feature_names, values.tolist()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    return {
        "method": "shap_kernel",
        "attributions": [
            {"feature": name, "shap_value": round(float(v), 4)}
            for name, v in pairs
        ],
        "top_features": [name for name, _ in pairs[:5]],
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Format permutation importance for UI (from Phase 10 output)
# ---------------------------------------------------------------------------

def format_permutation_importance(
    importance_dict: dict[str, float],
    top_k: int = 10,
) -> dict:
    """Format Phase 10 permutation importance output for API response.

    Parameters
    ----------
    importance_dict : {feature_name: mean_accuracy_drop}
    top_k           : Number of top features to highlight

    Returns
    -------
    dict with sorted attributions and disclaimer
    """
    sorted_items = sorted(
        importance_dict.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    return {
        "method": "permutation_importance",
        "attributions": [
            {"feature": name, "importance": round(float(val), 4)}
            for name, val in sorted_items
        ],
        "top_features": [name for name, _ in sorted_items[:top_k]],
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# ASR quality metadata view
# ---------------------------------------------------------------------------

def format_asr_quality_view(asr_metadata: dict) -> dict:
    """Format ASR quality metadata for the explainability view.

    Shows the user what quality factors influenced the voice branch weight.
    """
    return {
        "type": "asr_quality",
        "language": asr_metadata.get("language", "unknown"),
        "language_confidence": asr_metadata.get("language_prob", 0.0),
        "no_speech_probability": asr_metadata.get("no_speech_prob", 0.0),
        "word_count": asr_metadata.get("word_count", 0),
        "quality_score": asr_metadata.get("quality_score", 0.0),
        "rejection_reasons": asr_metadata.get("rejection_reasons", []),
        "note": (
            "A lower quality score reduces this modality's contribution "
            "to the fusion prediction via the reliability gate."
        ),
        "disclaimer": DISCLAIMER,
    }
