# Model Card — EmotiSense AI

> **Mandatory disclaimer:** This is a research prototype. It is not a medical
> device, clinical diagnostic system, or approved treatment tool. All outputs
> are research signals and must not replace professional mental-health
> assessment or emergency care.

---

## Model details

| Field | Value |
|---|---|
| Model name | EmotiSense AI |
| Version | untrained-phase-20 (architecture complete; training requires approved data) |
| Type | Multimodal early-risk research screening |
| Architecture | DeBERTa-v3 + BiLSTM + attention (text) · CNN-BiLSTM (voice) · MobileNetV2 (face) → Reliability-aware gated cross-modal attention fusion |
| Primary use | Research and development of multimodal mental-health screening methods |
| Intended users | Researchers, ML engineers, and academic projects studying depression-risk indicators |
| Out-of-scope uses | Clinical diagnosis, treatment recommendation, medication guidance, consumer mental-health apps without appropriate clinical oversight |
| Developers | EmotiSense AI Research Project |
| Phase | 20 / 20 — architecture and pipeline complete; pending trained weights on approved dataset |

---

## Architecture summary

```
Text ──► DeBERTa-v3 ─► BiLSTM ─► Trainable attention ─┐
Voice ─► ASR (Whisper) ─────────────► text branch      ├─► Reliability-aware
      └► CNN-BiLSTM speech encoder ─────────────────────┤   gated cross-modal
Face  ─► MobileNetV2 / ResNet-18 ──────────────────────┘   attention fusion
                                                              │
                                                    ┌─────────┴──────────┐
                                                 Risk head        Emotion profile
                                               (calibrated)       (7 emotions)
                                                    │
                                            Support policy engine
                                            (crisis-first, rule-based)
```

Each modality produces:
- A fixed-size embedding (256-d)
- A quality score ∈ [0, 1]
- A modality mask (0 = absent)

The fusion gate multiplies quality × learned gate to zero out unreliable branches.

---

## Training data requirements

| Dataset | Purpose | Access |
|---|---|---|
| DAIC-WOZ | Multimodal depression-risk training and evaluation | Requires approved academic/research agreement |
| E-DAIC | Extended candidate — same access process | Same as DAIC-WOZ |
| eRisk | Text-only depression baseline pretraining | CLEF eRisk data sharing agreement |
| RAVDESS | Speech emotion branch pretraining | CC BY-NC-SA 4.0 (non-commercial) |
| CREMA-D | Audio-visual emotion pretraining | Open (permissive) |
| FER-2013 / RAF-DB | Facial emotion branch pretraining | Open research use |

**Critical constraint:** DAIC-WOZ/E-DAIC labels must remain tied to the same
participant session as the audio, video, and transcript. Cross-dataset synthetic
pairings are invalid and are rejected by the dataset validator.

---

## Evaluation protocol

- Participant-disjoint train / validation / test splits (no participant appears
  in more than one split).
- Model and hyperparameter selection on the **validation** split only.
- One final evaluation on the **locked test** split after no further changes.
- Metrics reported: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier
  score, sensitivity, specificity, ECE (calibration), 95 % bootstrap CIs.
- Modality ablation over all single and pairwise subsets.
- Subgroup fairness analysis if demographic labels are available.

---

## Metrics (placeholder — no model trained yet)

No model has been trained on approved aligned data at this stage. The
architecture is complete and the evaluation protocol is fully implemented.
All metric fields below will be populated after training.

| Metric | Value |
|---|---|
| Test F1 | — (no trained model) |
| Test ROC-AUC | — |
| Test Brier score | — |
| ECE (post-calibration) | — |
| Sensitivity | — |
| Specificity | — |
| Fusion vs. text-only F1 delta | — |
| Inference latency (CPU) | — |

---

## Ethical considerations

### Benefits
- Provides a structured, validated pipeline for researching multimodal
  depression-risk indicators.
- Forces participant-disjoint evaluation to prevent inflated results.
- All outputs labelled as research signals, never diagnoses.

### Risks
- **Misinterpretation:** Risk scores could be mistaken for diagnoses if
  displayed without adequate framing. The mandatory disclaimer and
  `FORBIDDEN_DIAGNOSTIC_PHRASES` guard mitigate this.
- **Dataset bias:** DAIC-WOZ is predominantly English-language clinical
  interviews; generalisation to other languages, cultures, or informal
  settings is unknown.
- **Emotion model limitations:** Acted emotional expressions in pretraining
  data may not represent natural clinical affect.
- **Privacy:** Biometric data (voice, face) is sensitive. The system processes
  data in memory by default and does not persist raw inputs.
- **Accessibility:** ASR quality degrades with accented or non-native speech.

### Mitigations implemented
- Crisis-first safety policy: self-harm/danger signals bypass all suggestions.
- Mandatory disclaimers on every model output.
- Consent gates before any biometric data is processed.
- In-memory processing; no raw data stored by default.
- Rate limiting and upload size validation.
- Non-diagnostic, non-prescriptive suggestion library only.

---

## Caveats and known limitations

1. Depression-risk labels are not equivalent to clinical diagnoses.
2. Emotion recognition is noisy and influenced by cultural context.
3. Performance on DAIC-WOZ interview recordings may not generalise to
   casual user recordings or other populations.
4. A single modality cannot establish mental state.
5. Explanations (LIME, SHAP, Grad-CAM, gates) show model behaviour,
   not clinical causality.
6. The system cannot replace qualified professional assessment or
   emergency care.
7. Demographic representation in DAIC-WOZ is limited; subgroup fairness
   analysis is required before any deployment.

---

## How to reproduce

```powershell
# 1. Set up environment
cd emotisense-ai
python -m venv backend\.venv
backend\.venv\Scripts\pip install -r backend\requirements.txt

# 2. Prepare aligned dataset (requires DAIC-WOZ access)
python training\prepare_dataset.py  approved-index.csv --output-dir data\processed\manifests\v1

# 3. Pretrain emotion encoders
python training\vision\vision_transfer.py  --train fer_train.csv --validation fer_val.csv --output-dir models\checkpoints\face-v1
python training\voice\speech_emotion_model.py --train rav_train.csv --validation rav_val.csv --output-dir models\checkpoints\voice-v1

# 4. Train text model (ablation)
python training\text\run_ablation.py  --train train.csv --validation val.csv --output-dir models\checkpoints\text-ablation-v1

# 5. Train fusion model
python training\multimodal\train_fusion.py --train train.csv --validation val.csv --output-dir models\checkpoints\fusion-v1

# 6. Calibrate (validation only)
python training\multimodal\calibrate.py --checkpoint models\checkpoints\fusion-v1\best.pt --validation val.csv

# 7. Final locked benchmark (once only)
python experiments\scripts\run_benchmark.py --checkpoint models\checkpoints\fusion-v1\best.pt --test-manifest data\processed\manifests\v1\test.csv --output-dir experiments\reports\benchmark-v1

# 8. Generate consolidated report
python experiments\scripts\generate_report.py --reports-dir experiments\reports --output experiments\reports\full-report.md

# 9. Run test suite
backend\.venv\Scripts\python.exe -m pytest
```

---

## References

- [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu/)
- [eRisk at CLEF](https://erisk.irlab.org/)
- [RAVDESS](https://zenodo.org/record/1188976)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-base)
- [WavLM](https://huggingface.co/microsoft/wavlm-base)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [LIME](https://github.com/marcotcr/lime)
- [SHAP](https://github.com/shap/shap)

*Content was paraphrased for compliance with licensing restrictions.*
