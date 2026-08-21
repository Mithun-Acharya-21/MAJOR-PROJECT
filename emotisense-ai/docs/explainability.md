# Phase 17 — LIME, SHAP, Grad-CAM, and Fusion Explanation Views

## Explainability modules

| Module | Method | Branch |
|---|---|---|
| `text_explainer.py` | LIME token attributions | Text |
| `text_explainer.py` | SHAP (Text masker) token attributions | Text |
| `text_explainer.py` | Attention weight formatter (supplementary) | Text |
| `acoustic_explainer.py` | SHAP KernelExplainer over summary features | Voice |
| `acoustic_explainer.py` | Permutation importance formatter | Voice |
| `acoustic_explainer.py` | ASR quality metadata view | Voice |
| `vision_explainer.py` | Grad-CAM heatmap → base64 PNG | Face |
| `fusion_explainer.py` | Gate contribution + attention matrix | Fusion |
| `fusion_explainer.py` | Unified fusion explanation bundle | Fusion |

## Mandatory framing rules

Every explainability output must state:

| Output | Framing |
|---|---|
| LIME / SHAP tokens | "Explains model behaviour, not clinical causality" |
| Attention weights | "Supplementary diagnostic signal, not a causal explanation" |
| Acoustic attributions | "Model-observed patterns, not clinical evidence" |
| Grad-CAM | "Emotion-model visualisation, not a depression-risk indicator" |
| Gate weights | "Diagnostic signals, not causal proof" |
| Cross-modal attention | "Alignment signal, not causal influence" |

These disclaimers are enforced in code and tested.

## Text: LIME

```python
from app.ml.explainability.text_explainer import explain_text_lime
result = explain_text_lime(text, predict_fn, class_names=["control", "depressed"])
# result["attributions"]  → [{token, score}, ...]
# result["top_tokens"]    → top 5 influential tokens
# result["disclaimer"]    → mandatory text
```

## Text: SHAP

```python
from app.ml.explainability.text_explainer import explain_text_shap
result = explain_text_shap(text, predict_fn, max_evals=500)
```

## Voice: Acoustic SHAP

```python
from app.ml.explainability.acoustic_explainer import explain_acoustic_shap
result = explain_acoustic_shap(feature_vector, feature_names, predict_fn)
# result["top_features"] → most influential acoustic features
```

## Vision: Grad-CAM

```python
from app.ml.explainability.vision_explainer import build_vision_explanation
result = build_vision_explanation(gradcam_result, face_quality_metadata)
# result["heatmap_base64_png"] → base64 PNG for <img src="data:image/png;base64,..." />
```

## Fusion: Gate + attention view

```python
from app.ml.explainability.fusion_explainer import build_fusion_explanation
result = build_fusion_explanation(model_output, quality_scores, emotion_labels)
# result["gate_contribution"]    → {modalities: [{modality, gate_value, contribution_pct}]}
# result["cross_modal_attention"]→ {matrix: [{query_modality, attention_to: {}}]}
# result["emotion_profile"]      → {probabilities: {}, predicted_emotion: str}
```

## Performance considerations

- LIME (500 samples) and SHAP (KernelExplainer) are slow: cache results
  and compute asynchronously. Grad-CAM and fusion explanation are fast
  (single forward/backward pass).
- LIME/SHAP are not run during real-time inference; they are available
  as an on-demand "explain this result" action in the UI.
