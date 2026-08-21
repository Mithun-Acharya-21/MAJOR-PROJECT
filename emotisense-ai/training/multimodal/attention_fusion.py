"""Phase 14 — Reliability-aware gated attention fusion and calibration.

Architecture (production candidate)
-------------------------------------
projected embeddings  (text, audio, video)
        ↓
learned modality gates  (sigmoid, conditioned on quality scores + mask)
        ↓
cross-modal attention encoder  (multi-head, query = pooled context)
        ↓
fused representation
        ↓
risk head      → risk probability + uncertainty estimate
emotion profile→ aggregated emotion signals

Calibration
-----------
Temperature scaling on the validation set only, after the model is locked.
Produces a calibrated probability alongside the raw model probability.

Missing-modality contract (identical to Phase 13)
--------------------------------------------------
- Missing modality → mask = 0, gate output = 0, zero contribution.
- Never imputed or fabricated.
- Quality scores from validation/ASR/face pipeline directly modulate gates.

Outputs
-------
{
  logits             : (B, 2)
  probs              : (B, 2)   — uncalibrated
  calibrated_probs   : (B, 2)   — after temperature scaling (None if not calibrated)
  risk_score         : (B,)
  uncertainty        : (B,)     — entropy of probability distribution
  gate_weights       : (B, 3)   — per-modality gate values (inspectable)
  attention_weights  : (B, 3, 3)— cross-modal attention (inspectable)
  emotion_profile    : (B, n_emotion_classes) or None
  used_modalities    : list[str]
  disclaimer         : str
}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from training.multimodal.fusion_baselines import MODALITY_ORDER, ModalityProjection

DISCLAIMER = (
    "Risk scores are model-derived research signals. "
    "They are not clinical diagnoses and must not be used as such. "
    "Gate and attention weights are diagnostic signals, not causal proof."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AttentionFusionConfig:
    text_dim: int = 256
    audio_dim: int = 256
    video_dim: int = 256
    projected_dim: int = 256
    n_classes: int = 2
    n_emotion_classes: int = 7       # Set to 0 to disable emotion profile head
    n_attention_heads: int = 4
    dropout: float = 0.3
    gate_hidden_dim: int = 64        # Hidden dim for gate MLP
    uncertainty_method: str = "entropy"   # "entropy" only for now
    temperature: float = 1.0         # Updated by calibrate(); 1.0 = uncalibrated


# ---------------------------------------------------------------------------
# Modality gate
# ---------------------------------------------------------------------------

class ModalityGate(nn.Module):
    """Learned gate: quality score + mask → scalar weight ∈ (0, 1).

    Input per modality:
      - quality ∈ [0, 1]  (from validation pipeline)
      - mask    ∈ {0, 1}  (presence flag)
      - projected embedding (B, D)  [contextual signal]

    Output:
      gate ∈ [0, 1] per modality — hard-zeroed where mask = 0.
    """

    def __init__(self, projected_dim: int, gate_hidden_dim: int) -> None:
        super().__init__()
        # Input: embedding mean (1) + quality scalar (1) = 2 features
        self.gate_net = nn.Sequential(
            nn.Linear(projected_dim + 2, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        projected: Tensor,   # (B, D)
        quality: Tensor,     # (B,)
        mask: Tensor,        # (B,)  0 or 1
    ) -> Tensor:             # (B,)
        B = projected.shape[0]
        # Contextual summary: mean of projected embedding per sample
        emb_summary = projected.mean(dim=1, keepdim=True)  # (B, 1)
        q = quality.unsqueeze(1).clamp(0.0, 1.0)           # (B, 1)
        m = mask.unsqueeze(1).float()                       # (B, 1)
        gate_input = torch.cat([projected, q, m], dim=1)   # (B, D+2)
        gate = self.gate_net(gate_input).squeeze(1)         # (B,)
        # Hard-zero absent modalities
        return gate * mask.float()


# ---------------------------------------------------------------------------
# Cross-modal attention encoder
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """Multi-head attention over the three modality tokens.

    tokens   : (B, 3, D) — one token per modality
    The output is a weighted combination fed to the risk head.
    Attention weights are retained for explainability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: Tensor,         # (B, 3, D)
        key_padding_mask: Optional[Tensor] = None,  # (B, 3) True=ignore
    ) -> tuple[Tensor, Tensor]:
        """Returns (fused_tokens, attention_weights).
        fused_tokens     : (B, 3, D)
        attention_weights: (B, 3, 3)
        """
        attn_out, attn_w = self.attn(
            tokens, tokens, tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        tokens = self.norm(tokens + attn_out)
        tokens = self.norm2(tokens + self.ff(tokens))
        return tokens, attn_w


# ---------------------------------------------------------------------------
# Full reliability-aware fusion model
# ---------------------------------------------------------------------------

class ReliabilityAwareAttentionFusion(nn.Module):
    """Production fusion model: gates + cross-modal attention + risk head."""

    def __init__(self, config: AttentionFusionConfig) -> None:
        super().__init__()
        self.config = config
        D = config.projected_dim

        # Shared projection (reuse Phase 13 component)
        from training.multimodal.fusion_baselines import FusionConfig
        proj_cfg = FusionConfig(
            text_dim=config.text_dim,
            audio_dim=config.audio_dim,
            video_dim=config.video_dim,
            projected_dim=D,
            dropout=config.dropout,
        )
        self.projection = ModalityProjection(proj_cfg)

        # Per-modality gates
        self.gates = nn.ModuleDict({
            m: ModalityGate(D, config.gate_hidden_dim)
            for m in MODALITY_ORDER
        })

        # Cross-modal attention encoder
        self.cross_attn = CrossModalAttention(D, config.n_attention_heads, config.dropout)

        # Risk head (mean-pool fused tokens → classify)
        self.risk_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D // 2, config.n_classes),
        )

        # Optional emotion profile head
        if config.n_emotion_classes > 0:
            self.emotion_head: Optional[nn.Module] = nn.Linear(D, config.n_emotion_classes)
        else:
            self.emotion_head = None

        # Temperature parameter (updated by calibrate())
        self.register_buffer("temperature", torch.tensor(config.temperature))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: dict[str, Tensor],
        masks: dict[str, Tensor],
        quality: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, Tensor]:
        B = next(iter(embeddings.values())).shape[0]
        device = next(iter(embeddings.values())).device

        # Default quality = 1.0 for all present modalities if not supplied
        if quality is None:
            quality = {
                m: masks.get(m, torch.zeros(B, device=device)).clone()
                for m in MODALITY_ORDER
            }

        # 1. Project
        projected, masks = self.projection(embeddings, masks)

        # 2. Compute gates
        gate_vals: dict[str, Tensor] = {}
        for m in MODALITY_ORDER:
            q = quality.get(m, torch.ones(B, device=device))
            mk = masks.get(m, torch.zeros(B, device=device))
            gate_vals[m] = self.gates[m](projected[m], q, mk)  # (B,)

        # 3. Apply gates to projected embeddings
        gated = {m: projected[m] * gate_vals[m].unsqueeze(1) for m in MODALITY_ORDER}

        # 4. Stack into token sequence (B, 3, D)
        tokens = torch.stack([gated[m] for m in MODALITY_ORDER], dim=1)

        # 5. Build key_padding_mask: True where gate ≈ 0 (suppress in attention)
        gate_stack = torch.stack([gate_vals[m] for m in MODALITY_ORDER], dim=1)  # (B, 3)
        key_padding_mask = (gate_stack < 1e-6)  # True = masked out

        # 6. Cross-modal attention
        fused_tokens, attn_weights = self.cross_attn(tokens, key_padding_mask)  # (B,3,D), (B,3,3)

        # 7. Pool fused tokens (mean over non-masked positions)
        valid = (~key_padding_mask).float().unsqueeze(2)   # (B, 3, 1)
        denom = valid.sum(dim=1).clamp_min(1.0)            # (B, 1)
        pooled = (fused_tokens * valid).sum(dim=1) / denom # (B, D)

        # 8. Risk head
        logits = self.risk_head(pooled)
        raw_probs = F.softmax(logits, dim=1)

        # 9. Temperature-scaled calibrated probs
        cal_logits = logits / self.temperature.clamp_min(1e-3)
        cal_probs = F.softmax(cal_logits, dim=1)

        # 10. Uncertainty (entropy)
        uncertainty = -(cal_probs * (cal_probs + 1e-9).log()).sum(dim=1)  # (B,)
        max_entropy = math.log(self.config.n_classes)
        uncertainty = (uncertainty / max_entropy).clamp(0.0, 1.0)

        # 11. Emotion profile
        emotion_profile: Optional[Tensor] = None
        if self.emotion_head is not None:
            emotion_profile = F.softmax(self.emotion_head(pooled), dim=1)

        used = [m for m in MODALITY_ORDER if gate_vals.get(m, torch.zeros(1)).sum() > 0]

        out: dict[str, object] = {
            "logits": logits,
            "probs": raw_probs,
            "calibrated_probs": cal_probs,
            "risk_score": cal_probs[:, 1],
            "uncertainty": uncertainty,
            "gate_weights": gate_stack,         # (B, 3) — diagnostic
            "attention_weights": attn_weights,  # (B, 3, 3) — diagnostic
            "emotion_profile": emotion_profile,
            "used_modalities": used,
            "disclaimer": DISCLAIMER,
        }
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Temperature scaling calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_temperature(
    model: ReliabilityAwareAttentionFusion,
    val_embeddings: list[dict[str, Tensor]],
    val_masks: list[dict[str, Tensor]],
    val_labels: Tensor,
    lr: float = 0.01,
    max_steps: int = 500,
) -> float:
    """Find the temperature T that minimises NLL on the validation set.

    Follows the standard post-hoc temperature scaling protocol:
      - Model weights are FROZEN.
      - Only the scalar temperature parameter is optimised.
      - Validation set is the only data used.

    Parameters
    ----------
    model            : Trained, locked ReliabilityAwareAttentionFusion
    val_embeddings   : List of per-sample embedding dicts
    val_masks        : List of per-sample mask dicts
    val_labels       : (N,) integer class labels
    lr               : Learning rate for temperature optimisation
    max_steps        : Max gradient steps

    Returns
    -------
    Optimal temperature (also written back to model.temperature buffer).
    """
    # Collect all logits in one pass
    model.eval()
    all_logits: list[Tensor] = []
    for emb, msk in zip(val_embeddings, val_masks):
        # Add batch dimension if missing
        emb_b = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in emb.items()}
        msk_b = {k: v.unsqueeze(0) if v.dim() == 0 else v for k, v in msk.items()}
        out = model(emb_b, msk_b)
        all_logits.append(out["logits"])

    logits = torch.cat(all_logits, dim=0)  # (N, 2)
    labels = val_labels.to(logits.device)

    # Optimise temperature
    temperature = nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_steps)

    def _eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(_eval)
    optimal_t = float(temperature.item())

    # Write back to model buffer
    model.temperature.fill_(optimal_t)
    return optimal_t


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error (ECE) and per-bin statistics.

    Parameters
    ----------
    probs  : (N,) positive-class probability (risk_score)
    labels : (N,) integer labels (0 or 1)
    n_bins : number of confidence bins

    Returns
    -------
    dict with ece, max_calibration_error, bin_stats list
    """
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()

    bins = torch.linspace(0.0, 1.0, n_bins + 1).numpy()
    ece = 0.0
    mce = 0.0
    N = len(probs_np)
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (probs_np >= lo) & (probs_np < hi)
        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            bin_stats.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": 0})
            continue
        conf = float(probs_np[in_bin].mean())
        acc = float(labels_np[in_bin].mean())
        gap = abs(conf - acc)
        ece += (n_in_bin / N) * gap
        mce = max(mce, gap)
        bin_stats.append({
            "bin": f"[{lo:.2f},{hi:.2f})",
            "n": n_in_bin,
            "mean_confidence": round(conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(gap, 4),
        })

    return {
        "ece": round(float(ece), 4),
        "max_calibration_error": round(float(mce), 4),
        "n_bins": n_bins,
        "n_samples": N,
        "bin_stats": bin_stats,
    }
