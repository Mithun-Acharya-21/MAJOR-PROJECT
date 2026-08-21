# System Summary — EmotiSense AI (Viva One-Pager)

## What the system does

EmotiSense AI is a **multimodal early-risk research screening tool**. It accepts
text, voice recordings, and facial images, processes each through a dedicated
encoder, and fuses the results with a reliability-aware attention mechanism to
produce a calibrated risk signal and supportive suggestions.

> It is a **research screening tool, not a diagnostic system.**

---

## Architecture at a glance

```
Text input   ──► DeBERTa-v3 + BiLSTM + attention ──────────────┐
Voice input  ──► Whisper ASR ─────────► text branch            │
             └─► CNN-BiLSTM acoustic encoder ───────────────────┼──►  Reliability-
Image/video  ──► Face detection + MobileNetV2 ──────────────────┘     aware gated
                                                                       cross-modal
                                                                       attention
                                                                         │
                                                              ┌──────────┴──────────┐
                                                           Risk head          Emotion profile
                                                       (temperature-             (7 classes)
                                                        calibrated)
                                                              │
                                                     Support policy engine
                                                     (crisis-first, rule-based,
                                                      non-prescriptive)
```

---

## 20-phase delivery summary

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Repo, FastAPI skeleton, safety config, Docker | Done |
| 2 | Dataset strategy, compatibility audit | Done |
| 3 | Leakage-safe preprocessing, participant splits | Done |
| 4 | Text preprocessing, TF-IDF + LR baseline | Done |
| 5 | DeBERTa-v3 + BiLSTM + attention architecture | Done |
| 6 | Ablation evaluation, text-head comparison | Done |
| 7 | Face detection, quality scoring, FER baseline CNN | Done |
| 8 | Transfer learning (MobileNetV2/ResNet-18), Grad-CAM | Done |
| 9 | Audio validation, VAD, MFCC + spectral features | Done |
| 10 | CNN-BiLSTM speech emotion model, permutation importance | Done |
| 11 | Offline Whisper ASR, transcript quality assessment | Done |
| 12 | Aligned multimodal dataset builder, manifest validator | Done |
| 13 | Concat/mean-pool/weighted-sum fusion baselines | Done |
| 14 | Reliability-aware gated attention fusion, ECE calibration | Done |
| 15 | Full evaluation suite: metrics, ablation, fairness, domain-shift | Done |
| 16 | Crisis-first support policy engine, suggestion library | Done |
| 17 | LIME, SHAP, Grad-CAM, gate, attention explainability views | Done |
| 18 | FastAPI routes, privacy controls, rate limiting, 240+ tests | Done |
| 19 | React/TS frontend, consent gates, Docker compose | Done |
| 20 | Benchmark runner, model card, viva materials, final report | Done |

---

## Key scientific safeguards

| Safeguard | Enforcement |
|---|---|
| No cross-person multimodal pairings | `validate_aligned_manifest()` rejects unapproved datasets |
| Participant-disjoint splits | `build_aligned_manifest()` + manifest SHA-256 |
| Test set locked until final evaluation | `assert_not_final_test(locked=False)` guard |
| No diagnostic language in outputs | `FORBIDDEN_DIAGNOSTIC_PHRASES` + schema defaults |
| Crisis-first safety | `CRISIS_SIGNALS` frozenset bypasses all other logic |
| No prescriptive suggestions | Tested against forbidden-word list |
| In-memory biometric processing | No raw input persisted by default |
| Explainability disclaimers | Every explanation output tested for disclaimer text |

---

## Test suite

| Test file | Tests | Coverage |
|---|---|---|
| test_system_api.py | 2 | Health, model info |
| test_safety.py | 2 | Forbidden phrase filter |
| test_analysis_api.py | 24 | All 6 API routes |
| test_dataset_manifest.py | 6 | Manifest validation |
| test_prepare_dataset.py | 4 | Split pipeline |
| test_text_preprocess.py | 4 | Normaliser |
| test_text_baseline.py | 2 | Classical baseline safety |
| test_text_evaluate.py | 3 | Evaluation guards |
| test_attention_pooling.py | 1 | Masked attention |
| test_text_ablation_config.py | 2 | Variant configs |
| test_face_preprocess.py | 9 | Face quality |
| test_vision_transfer.py | 10 | Transfer model + Grad-CAM |
| test_audio_pipeline.py | 19 | Validation + acoustic features |
| test_speech_emotion.py | 15 | Speech emotion model |
| test_asr.py | 22 | ASR quality + routing |
| test_aligned_dataset.py | 22 | Multimodal dataset builder |
| test_fusion_baselines.py | 48 | 3 fusion variants |
| test_attention_fusion.py | 24 | Reliability-aware fusion |
| test_evaluation.py | 26 | Full metric suite |
| test_support_policy.py | 26 | Crisis + policy logic |
| test_explainability.py | 23 | LIME/SHAP/Grad-CAM/fusion |

**Total: 240+ tests across all phases**

---

## Why no trained model?

DAIC-WOZ requires an approved academic/research data-use agreement from USC
ICT. The pipeline is ready; access is pending. All code, architecture, safety
controls, and evaluation protocol are in place. No metrics have been fabricated.
