"""Phase 17 — Fusion branch explainability: gate weights and cross-modal attention.

Provides a unified explanation view that combines:
  - Per-modality gate weights (reliability-weighted contribution)
  - Cross-modal attention matrix (inter-modality alignment)
  - Risk score + uncertainty
  - Emotion profile per modality

All values are derived from the model's own outputs (no additional
black-box sampling required). The explanation is fast and deterministic.

Safety constraints
------------------
- Gate and attention weights are DIAGNOSTIC signals.
- They must never be described as proving which modality caused the prediction.
- Contribution percentages are derived from learned gates, not ground-truth.
- Every response carries a mandatory disclaimer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from training.multimodal.fusion_baselines import MODALITY_ORDER

DISCLAIMER = (
    "Fusion weights and attention maps show how the model weighted each modality. "
    "They are diagnostic signals, not causal proof. "
    "This is a research screening tool, not a clinical diagnostic system."
)

GATE_DISCLAIMER = (
    "Gate values reflect the model's learned reliability weighting for each modality "
    "given quality scores and input features. "
    "Higher gate values indicate greater contribution — not causal proof of clinical significance."
)


# ---------------------------------------------------------------------------
# Gate-weight contribution view
# ---------------------------------------------------------------------------

def format_gate_contribution(
    gate_weights: "np.ndarray | list",
    used_modalities: list[str],
    quality_scores: Optional[dict[str, float]] = None,
) -> dict:
    """Format per-modality gate weights for the UI explanation view.

    Parameters
    ----------
    gate_weights    : (3,) or (B, 3) array — per-modality gate values
    used_modalities : List of modality names that contributed
    quality_scores  : Optional dict of quality scores per modality

    Returns
    -------
    dict with per-modality contribution and formatted percentages
    """
    weights = np.array(gate_weights, dtype=float)
    if weights.ndim == 2:
        weights = weights.mean(axis=0)  # Average across batch
    weights = weights.flatten()

    total = weights.sum()
    if total > 1e-8:
        normalised = weights / total
    else:
        normalised = np.zeros_like(weights)

    modality_contributions = []
    for i, modality in enumerate(MODALITY_ORDER):
        w = float(weights[i])
        pct = round(float(normalised[i]) * 100, 1)
        qs = quality_scores.get(modality) if quality_scores else None
        modality_contributions.append({
            "modality": modality,
            "gate_value": round(w, 4),
            "contribution_pct": pct,
            "quality_score": round(float(qs), 3) if qs is not None else None,
            "active": modality in used_modalities,
        })

    return {
        "type": "gate_contribution",
        "modalities": modality_contributions,
        "used_modalities": used_modalities,
        "disclaimer": GATE_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Cross-modal attention view
# ---------------------------------------------------------------------------

def format_attention_matrix(
    attention_weights: "np.ndarray",
    modality_names: Optional[list[str]] = None,
) -> dict:
    """Format cross-modal attention matrix for UI heatmap display.

    Parameters
    ----------
    attention_weights : (3, 3) or (B, 3, 3) attention weight matrix
    modality_names    : Ordered list of modality names (default: MODALITY_ORDER)

    Returns
    -------
    dict with matrix rows, column labels, and disclaimer
    """
    modality_names = modality_names or MODALITY_ORDER
    attn = np.array(attention_weights, dtype=float)
    if attn.ndim == 3:
        attn = attn.mean(axis=0)  # Average over batch

    matrix_rows = []
    for i, query_mod in enumerate(modality_names):
        row_values = [round(float(attn[i, j]), 4) for j in range(len(modality_names))]
        matrix_rows.append({
            "query_modality": query_mod,
            "attention_to": dict(zip(modality_names, row_values)),
        })

    return {
        "type": "cross_modal_attention",
        "matrix": matrix_rows,
        "column_labels": modality_names,
        "interpretation": (
            "Each row shows how much one modality attended to every other modality. "
            "Higher values indicate stronger alignment — not causal influence."
        ),
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Unified fusion explanation bundle
# ---------------------------------------------------------------------------

def build_fusion_explanation(
    model_output: dict,
    quality_scores: Optional[dict[str, float]] = None,
    emotion_labels: Optional[list[str]] = None,
) -> dict:
    """Assemble a complete fusion explanation bundle from model output.

    Parameters
    ----------
    model_output   : Dict returned by ReliabilityAwareAttentionFusion.forward()
    quality_scores : Per-modality quality scores from validation pipelines
    emotion_labels : Ordered list of emotion class names (7 classes)

    Returns
    -------
    Complete explanation dict suitable for API serialisation
    """
    import torch

    def _to_numpy(v):
        if v is None:
            return None
        if hasattr(v, "detach"):
            return v.detach().cpu().numpy()
        return np.array(v)

    gate_w = _to_numpy(model_output.get("gate_weights"))
    attn_w = _to_numpy(model_output.get("attention_weights"))
    risk = float(_to_numpy(model_output.get("risk_score", np.array([0.0]))).flatten()[0])
    uncertainty = float(_to_numpy(model_output.get("uncertainty", np.array([0.0]))).flatten()[0])
    used = model_output.get("used_modalities", [])

    # Emotion profile
    emotion_profile = None
    ep = _to_numpy(model_output.get("emotion_profile"))
    if ep is not None:
        ep = ep.flatten()
        labels = emotion_labels or [f"emotion_{i}" for i in range(len(ep))]
        emotion_profile = {
            "probabilities": {
                lab: round(float(ep[i]), 4)
                for i, lab in enumerate(labels)
            },
            "predicted_emotion": labels[int(ep.argmax())],
        }

    gate_view = (
        format_gate_contribution(gate_w, used, quality_scores)
        if gate_w is not None else None
    )
    attn_view = (
        format_attention_matrix(attn_w)
        if attn_w is not None else None
    )

    return {
        "risk_score": round(risk, 4),
        "uncertainty": round(uncertainty, 4),
        "risk_tier": _score_to_tier_label(risk),
        "used_modalities": used,
        "gate_contribution": gate_view,
        "cross_modal_attention": attn_view,
        "emotion_profile": emotion_profile,
        "disclaimer": DISCLAIMER,
    }


def _score_to_tier_label(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.40:
        return "moderate"
    return "low"
