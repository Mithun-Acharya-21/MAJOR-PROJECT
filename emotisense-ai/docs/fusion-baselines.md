# Phase 13 — Fusion Baselines and Missing-Modality Handling

## Three baseline fusion strategies

All three receive the same inputs and are evaluated on the same frozen split.

| Variant | Fusion operation | Classifier input size |
|---|---|---|
| `concat` | Concatenate all projected embeddings | `projected_dim × 3` |
| `mean_pool` | Mean over present modality embeddings | `projected_dim` |
| `weighted_sum` | Uniform weighted sum over present | `projected_dim` |

Each modality embedding is first projected to a common `projected_dim`
through `Linear → LayerNorm → ReLU → Dropout`.

## Missing-modality contract

```
embeddings  : {"text": (B, D), "audio": (B, D), "video": (B, D)}
              — absent keys treated as zero vectors
masks       : {"text": (B,), "audio": (B,), "video": (B,)}
              — float 0 = absent, 1 = present
quality     : {"text": (B,), ...}  [not used in baselines; used in Phase 14]
```

- A missing modality contributes a **zero vector** after projection.
- Its mask value zeros its contribution in `fusion_weights`.
- The model is never given fabricated or imputed features.
- `used_modalities` in the output lists only branches with non-zero mask.

## Modality dropout (training augmentation)

`apply_modality_dropout(masks, drop_prob=0.15, training=True)` randomly
zeros entire modality masks during training so the model learns robustness
to missing inputs at inference time.

**Safety guarantee:** at least one modality is always preserved per sample.

## Ablation runner

`run_fusion_ablation(model, embeddings, masks)` evaluates a trained fusion
model under all single-modality and pairwise subsets:

```
"all"           — all present modalities
"only_text"     — text branch only
"only_audio"    — audio branch only
"only_video"    — video branch only
"text+audio"    — both text and audio
"text+video"    — both text and video
"audio+video"   — both audio and video
```

Results inform Phase 15 missing-modality robustness analysis.

## Running a baseline comparison

```python
from training.multimodal.fusion_baselines import build_fusion_model, FusionConfig, run_fusion_ablation
import torch

for ft in ["concat", "mean_pool", "weighted_sum"]:
    model = build_fusion_model(FusionConfig(fusion_type=ft))
    # ... train on aligned split ...
    ablation = run_fusion_ablation(model, embeddings, masks)
```

## Output fields

| Field | Shape | Description |
|---|---|---|
| `logits` | (B, 2) | Raw classification logits |
| `probs` | (B, 2) | Softmax probabilities |
| `risk_score` | (B,) | Probability of positive class |
| `fusion_weights` | (B, 3) | Per-modality contribution weight |
| `used_modalities` | list | Modalities with non-zero mask |
| `disclaimer` | str | Mandatory safety note |

## Mandatory disclaimer

> *"Risk scores are model-derived research signals. They are not clinical
> diagnoses and must not be used as such."*

This string is included in every model output and must appear on every
prediction display in the UI (Phase 19).
