"""Phase 13 — Multimodal fusion baselines and missing-modality handling.

Three fusion strategies are implemented here so they can be benchmarked
on the same participant-aligned split before the full reliability-aware
attention fusion is built in Phase 14.

Fusion variants
---------------
1. ConcatFusion          — project each present embedding, concatenate, classify.
2. MeanPoolFusion        — average present embeddings after projection, classify.
3. WeightedSumFusion     — fixed uniform weights over present modalities, classify.

Missing-modality contract
--------------------------
- A missing modality (mask value 0) contributes a ZERO vector to all
  baselines — it is never fabricated or imputed.
- The fusion model receives:
    embeddings  : dict[str, Tensor]  — {modality: (B, D)} or absent key
    masks       : dict[str, Tensor]  — {modality: (B,) float in {0,1}}
    quality     : dict[str, Tensor]  — {modality: (B,) float in [0,1]}
                                       (used only in Phase 14)

Modality order is deterministic: ["text", "audio", "video"].
A model trained with all three modalities can be evaluated with any
subset by zeroing the corresponding mask.

Outputs
-------
dict with:
  logits          : (B, 2)
  probs           : (B, 2) softmax
  risk_score      : (B,)   probability of positive class
  used_modalities : list[str] — modalities with non-zero mask for this batch
  fusion_weights  : (B, n_modalities) — contribution weight per modality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MODALITY_ORDER = ["text", "audio", "video"]

DISCLAIMER = (
    "Risk scores are model-derived research signals. "
    "They are not clinical diagnoses and must not be used as such."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    text_dim: int = 256          # Input embedding dims (must match encoder outputs)
    audio_dim: int = 256
    video_dim: int = 256
    projected_dim: int = 256     # Common projected dimension
    n_classes: int = 2
    dropout: float = 0.3
    fusion_type: str = "concat"  # "concat" | "mean_pool" | "weighted_sum"


# ---------------------------------------------------------------------------
# Shared projection layer
# ---------------------------------------------------------------------------

class ModalityProjection(nn.Module):
    """Project each modality embedding to a common dimension."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        dims = {
            "text": config.text_dim,
            "audio": config.audio_dim,
            "video": config.video_dim,
        }
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], config.projected_dim),
                nn.LayerNorm(config.projected_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            )
            for m in MODALITY_ORDER
        })
        self.projected_dim = config.projected_dim

    def forward(
        self,
        embeddings: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Project present embeddings; return zeros for absent modalities.

        Parameters
        ----------
        embeddings : {modality: (B, D_m)} — may be missing keys
        masks      : {modality: (B,)} — float 0/1

        Returns
        -------
        projected : {modality: (B, projected_dim)} — zeros where mask=0
        masks     : same dict, unchanged
        """
        B = next(iter(embeddings.values())).shape[0]
        projected: dict[str, Tensor] = {}
        for m in MODALITY_ORDER:
            mask = masks.get(m)
            if mask is None:
                mask = torch.zeros(B, device=next(iter(embeddings.values())).device)
                masks = {**masks, m: mask}
            if m in embeddings:
                p = self.projections[m](embeddings[m])
            else:
                # Modality absent → zero vector
                device = next(iter(embeddings.values())).device
                p = torch.zeros(B, self.projected_dim, device=device)
            # Zero out rows where mask == 0
            p = p * mask.unsqueeze(1)
            projected[m] = p
        return projected, masks


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseFusion(nn.Module):
    """Shared logic for all fusion baselines."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = ModalityProjection(config)

    def _classify(self, fused: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    def forward(
        self,
        embeddings: dict[str, Tensor],
        masks: dict[str, Tensor],
        quality: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, Tensor]:
        projected, masks = self.projection(embeddings, masks)
        out = self._fuse_and_classify(projected, masks)
        # Annotate used modalities (batch-level: any sample has non-zero mask)
        used = [
            m for m in MODALITY_ORDER
            if masks.get(m, torch.zeros(1)).sum() > 0
        ]
        out["used_modalities"] = used
        out["disclaimer"] = DISCLAIMER
        return out

    def _fuse_and_classify(
        self,
        projected: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Concatenation baseline
# ---------------------------------------------------------------------------

class ConcatFusion(_BaseFusion):
    """Concatenate projected embeddings → classify.

    Missing modalities contribute zero vectors, so the concatenated
    input is always ``projected_dim × 3`` wide regardless of missing data.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__(config)
        concat_dim = config.projected_dim * len(MODALITY_ORDER)
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, config.projected_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.projected_dim, config.n_classes),
        )

    def _fuse_and_classify(
        self,
        projected: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        # Deterministic modality order
        vecs = [projected[m] for m in MODALITY_ORDER]
        fused = torch.cat(vecs, dim=1)                    # (B, D×3)
        n_present = sum(
            masks.get(m, torch.zeros(1)).unsqueeze(1)
            for m in MODALITY_ORDER
        )                                                  # (B, 1)
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=1)
        # Fusion weights: equal over present modalities
        weights = torch.stack(
            [masks.get(m, torch.zeros(logits.shape[0], device=logits.device))
             for m in MODALITY_ORDER], dim=1
        )  # (B, 3)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "logits": logits,
            "probs": probs,
            "risk_score": probs[:, 1],
            "fusion_weights": weights / denom,
        }


# ---------------------------------------------------------------------------
# 2. Mean-pool baseline
# ---------------------------------------------------------------------------

class MeanPoolFusion(_BaseFusion):
    """Average projected embeddings over present modalities → classify."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.projected_dim, config.projected_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.projected_dim // 2, config.n_classes),
        )

    def _fuse_and_classify(
        self,
        projected: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        B = next(iter(projected.values())).shape[0]
        device = next(iter(projected.values())).device

        stacked = torch.stack(
            [projected[m] for m in MODALITY_ORDER], dim=1
        )  # (B, 3, D)
        mask_vec = torch.stack(
            [masks.get(m, torch.zeros(B, device=device)) for m in MODALITY_ORDER],
            dim=1,
        ).unsqueeze(2)  # (B, 3, 1)

        summed = (stacked * mask_vec).sum(dim=1)           # (B, D)
        count = mask_vec.sum(dim=1).clamp_min(1.0)        # (B, 1)
        fused = summed / count

        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=1)
        weights = mask_vec.squeeze(2)                       # (B, 3)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "logits": logits,
            "probs": probs,
            "risk_score": probs[:, 1],
            "fusion_weights": weights / denom,
        }


# ---------------------------------------------------------------------------
# 3. Weighted-sum baseline (fixed uniform weights)
# ---------------------------------------------------------------------------

class WeightedSumFusion(_BaseFusion):
    """Weighted sum with fixed uniform weights over present modalities."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.projected_dim, config.projected_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.projected_dim // 2, config.n_classes),
        )

    def _fuse_and_classify(
        self,
        projected: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        # Identical to MeanPool for fixed uniform weights —
        # but kept as a separate variant so learned weights can be plugged in later
        B = next(iter(projected.values())).shape[0]
        device = next(iter(projected.values())).device

        stacked = torch.stack(
            [projected[m] for m in MODALITY_ORDER], dim=1
        )
        mask_vec = torch.stack(
            [masks.get(m, torch.zeros(B, device=device)) for m in MODALITY_ORDER],
            dim=1,
        ).unsqueeze(2)

        count = mask_vec.sum(dim=1).clamp_min(1.0)
        uniform_weights = mask_vec / count.unsqueeze(1)     # (B, 3, 1)
        fused = (stacked * uniform_weights).sum(dim=1)

        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=1)
        weights = uniform_weights.squeeze(2)
        return {
            "logits": logits,
            "probs": probs,
            "risk_score": probs[:, 1],
            "fusion_weights": weights,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

FUSION_VARIANTS: dict[str, type] = {
    "concat": ConcatFusion,
    "mean_pool": MeanPoolFusion,
    "weighted_sum": WeightedSumFusion,
}


def build_fusion_model(config: FusionConfig) -> _BaseFusion:
    """Instantiate a fusion model by config.fusion_type."""
    cls = FUSION_VARIANTS.get(config.fusion_type)
    if cls is None:
        raise ValueError(
            f"Unknown fusion_type '{config.fusion_type}'. "
            f"Choose one of {sorted(FUSION_VARIANTS)}."
        )
    return cls(config)


# ---------------------------------------------------------------------------
# Missing-modality dropout (training augmentation)
# ---------------------------------------------------------------------------

def apply_modality_dropout(
    masks: dict[str, Tensor],
    drop_prob: float = 0.15,
    training: bool = True,
    rng: Optional[torch.Generator] = None,
) -> dict[str, Tensor]:
    """Randomly zero out entire modalities during training.

    Ensures the model learns to be robust to missing inputs at inference.
    At least one modality is always preserved per sample.

    Parameters
    ----------
    masks    : {modality: (B,) float tensor} — 1=present, 0=absent
    drop_prob: probability of dropping an individual present modality
    training : no-op when False
    """
    if not training or drop_prob <= 0.0:
        return masks

    B = next(iter(masks.values())).shape[0]
    device = next(iter(masks.values())).device
    new_masks = {m: v.clone() for m, v in masks.items()}

    for b in range(B):
        present = [m for m in MODALITY_ORDER if new_masks.get(m, torch.zeros(1))[b] > 0]
        if len(present) <= 1:
            continue  # Can't drop the only modality
        for m in present:
            if torch.rand(1, generator=rng).item() < drop_prob:
                new_masks[m][b] = 0.0
        # Safety: ensure at least one remains
        still_present = [m for m in present if new_masks[m][b] > 0]
        if not still_present:
            keep = present[torch.randint(len(present), (1,)).item()]
            new_masks[keep][b] = 1.0

    return new_masks


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_fusion_ablation(
    model: _BaseFusion,
    embeddings: dict[str, Tensor],
    full_masks: dict[str, Tensor],
) -> dict[str, dict]:
    """Evaluate a fusion model under all single and combined modality subsets.

    Returns a dict mapping subset_name → model output dict.
    Useful for missing-modality robustness evaluation (Phase 15).
    """
    B = next(iter(embeddings.values())).shape[0]
    device = next(iter(embeddings.values())).device

    subsets: dict[str, dict[str, Tensor]] = {}

    # All present
    subsets["all"] = full_masks

    # Single modalities
    for m in MODALITY_ORDER:
        if full_masks.get(m, torch.zeros(B))[0] > 0:
            single = {k: torch.zeros(B, device=device) for k in MODALITY_ORDER}
            single[m] = full_masks[m].clone()
            subsets[f"only_{m}"] = single

    # Pairwise
    for i, m1 in enumerate(MODALITY_ORDER):
        for m2 in MODALITY_ORDER[i + 1:]:
            if (full_masks.get(m1, torch.zeros(B))[0] > 0 and
                    full_masks.get(m2, torch.zeros(B))[0] > 0):
                pair = {k: torch.zeros(B, device=device) for k in MODALITY_ORDER}
                pair[m1] = full_masks[m1].clone()
                pair[m2] = full_masks[m2].clone()
                subsets[f"{m1}+{m2}"] = pair

    model.eval()
    results: dict[str, dict] = {}
    with torch.no_grad():
        for subset_name, subset_masks in subsets.items():
            out = model(embeddings, subset_masks)
            results[subset_name] = {
                "risk_score_mean": float(out["risk_score"].mean()),
                "used_modalities": out["used_modalities"],
                "fusion_weights_mean": out["fusion_weights"].mean(dim=0).tolist(),
            }
    return results
