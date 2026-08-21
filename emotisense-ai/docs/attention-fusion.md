# Phase 14 — Reliability-Aware Attention Fusion and Calibration

## Architecture

```
text embedding (B, 256)  ─────────────────────────────────────────┐
audio embedding (B, 256) ──── ModalityProjection ────────────────├──► (B, 3, 256) tokens
video embedding (B, 256) ─────────────────────────────────────────┘
                                      ↓
                           ModalityGate × 3
                    [quality score + mask → sigmoid gate ∈ (0,1)]
                    [zero-masked where modality absent]
                                      ↓
                      CrossModalAttention (4 heads)
                    [query=key=value=gated token sequence]
                    [key_padding_mask suppresses absent tokens]
                                      ↓
                      Mean-pool over valid tokens (B, 256)
                                      ↓
                         ┌────────────┴─────────────┐
                      Risk head               Emotion profile head
                    Linear → GELU           Linear → softmax
                    → (B, 2) logits         → (B, 7) emotion probs
                                      ↓
                   Temperature scaling (post-hoc calibration)
                         → calibrated_probs (B, 2)
                         → uncertainty (normalised entropy)
```

## Modality gate

The gate MLP takes `[projected_embedding, quality_score, mask_flag]` as input
and outputs a scalar in (0, 1), then hard-multiplied by the presence mask.

- **quality_score** comes from the validation/ASR/face pipelines (Phase 9–11).
- A failed ASR, low audio SNR, or blurry face reduces that modality's gate.
- An absent modality (mask=0) always produces gate=0.

Gate values are retained in `gate_weights` for inspection in Phase 17.
They are a **diagnostic signal, not causal proof**.

## Cross-modal attention

Standard multi-head self-attention over the three modality tokens.
Tokens with gate≈0 are suppressed via `key_padding_mask`.
The `attention_weights` (B×3×3) matrix shows inter-modality alignment and
is exposed to the explainability view (Phase 17).

## Temperature scaling calibration

Post-hoc protocol — model weights are **frozen**:
1. Collect logits on the validation set.
2. Minimise NLL by optimising a single scalar temperature T with L-BFGS.
3. Write T back to the model buffer.
4. All subsequent `calibrated_probs` = softmax(logits / T).

```python
from training.multimodal.attention_fusion import calibrate_temperature
t = calibrate_temperature(model, val_embeddings, val_masks, val_labels)
```

## Calibration quality — ECE

```python
from training.multimodal.attention_fusion import expected_calibration_error
result = expected_calibration_error(model_risk_scores, labels, n_bins=10)
print(result["ece"])
```

Outputs: `ece`, `max_calibration_error`, per-bin confidence vs accuracy.

## Output fields

| Field | Shape | Description |
|---|---|---|
| `logits` | (B, 2) | Raw classification logits |
| `probs` | (B, 2) | Uncalibrated softmax probs |
| `calibrated_probs` | (B, 2) | Temperature-scaled probs |
| `risk_score` | (B,) | Calibrated positive-class probability |
| `uncertainty` | (B,) | Normalised entropy ∈ [0, 1] |
| `gate_weights` | (B, 3) | Per-modality gate values |
| `attention_weights` | (B, 3, 3) | Cross-modal attention |
| `emotion_profile` | (B, 7) | Aggregated emotion signals |
| `used_modalities` | list | Modalities with non-zero gate |
| `disclaimer` | str | Mandatory safety note |

## Mandatory disclaimer

> *"Risk scores are model-derived research signals. They are not clinical
> diagnoses and must not be used as such. Gate and attention weights are
> diagnostic signals, not causal proof."*

Every prediction screen must display this text (enforced in Phase 19).
