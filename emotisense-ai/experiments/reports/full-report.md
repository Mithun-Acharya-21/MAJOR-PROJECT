# EmotiSense AI — Consolidated Experiment Report

> **Research disclaimer:** All metrics are computed on research datasets under
> participant-disjoint evaluation conditions. Performance on clinical populations
> has not been established. These results are not evidence of clinical validity
> and must not be interpreted as diagnostic accuracy.

## System overview

EmotiSense AI is a multimodal early-risk research screening tool. It accepts
text, voice, and facial input and produces a calibrated risk score with
modality-level explanations.

Architecture: DeBERTa-v3 (text) + CNN-BiLSTM (voice) + MobileNetV2/ResNet-18
(face) → reliability-aware gated cross-modal attention fusion → risk head +
emotion profile + support policy engine.


## Benchmark results

### Run: 2026-08-21T09:05:07.844357+00:00
**Hardware:** Development machine  **Status:** no_checkpoint

> No trained model checkpoint found. Train the model on approved aligned data before running the final benchmark. This is expected during the architecture-only phase.


## Key limitations

- Depression-risk labels from research datasets are not clinical diagnoses.
- All model outputs are research signals; none constitute medical advice.
- Emotion recognition is noisy and culturally dependent.
- Acted emotion datasets (RAVDESS, CREMA-D, FER-2013) may not generalise to
  naturalistic expressions.
- Performance on DAIC-WOZ benchmark interviews may not generalise to casual
  recordings.
- Explainability outputs (LIME, SHAP, Grad-CAM) explain model behaviour, not
  clinical causality.
- All reported metrics require access to approved clinical datasets. No metrics
  are fabricated for this report.



## Reproducibility checklist

| Item | Status |
|---|---|
| Participant-disjoint splits with fixed random seed | Required |
| Split manifests SHA-256 recorded | Required |
| Model checkpoint SHA-256 recorded | Required |
| Temperature calibration on validation set only | Required |
| Final test evaluation after model lock | Required |
| Training hyperparameters logged to configs/model.yaml | Done |
| Dataset registry and data cards in configs/datasets.yaml | Done |
| All code versioned in git with commit hash | Recommended |


---
*Generated: 2026-08-21T09:05:20.098900+00:00*
