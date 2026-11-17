# Phase 5 — Research Limitations

**Date:** 2026-11-16  
**Project:** NeuroScreen  
**Status:** Development-only phase (locked test never touched)

---

## 1. Dataset Limitations

### 1.1 EATD (Chinese Multimodal Depression Corpus)

| Limitation | Evidence | Impact |
|---|---|---|
| Small development pool | 138 participants (45 depressed at SDS ≥ 50) | All confidence intervals are wide; some cross 0.5 |
| Participant-level CI width | Phase 2 ROC-AUC CIs were 0.19–0.43 wide | Point estimates are unreliable without CI |
| Session-to-participant aggregation | 3 prompted topics per participant are not independent | Session-level evaluation is prohibited |
| Label discrepancy | SDS ≥ 50 gives 53 depressed vs SDS ≥ 53 gives 30 depressed | Dataset README matches ≥ 53 but Phase 2 uses ≥ 50 |
| Author label confirmation | The corpus README does not state the threshold formula explicitly | Label definition carries residual uncertainty |
| Chinese-language corpus | All text transcripts are Chinese; E-T2 uses a multilingual/Chinese transformer | Models may not generalize to other languages |
| Prompted interview setting | Speech is from structured three-topic interviews | Does not represent naturalistic or clinical speech |

### 1.2 AVDD-TCMI

| Limitation | Evidence | Impact |
|---|---|---|
| No participant identity released | Dataset documentation explicitly states no speaker ID | All AVDD results are **sample-level only**; participant-level generalization cannot be claimed |
| Sample-level bootstrapping may be optimistic | Unknown if multiple samples come from same person | AVDD confidence intervals carry caveat: "sample-level resampling; clustering by speaker unknown" |
| HDS and SDS are separate populations | Different label instruments, gender distributions, protocols | Cannot pool without preregistered justification |
| No acoustic feature dictionary | 1,024 columns header-less (`f0..f1023`) | Acoustic features are uninterpretable; no clinical meaning can be assigned |
| OpenFace landmark coordinates | 484 of 533 visual features are raw 2D/3D landmark pixel coordinates | These confound identity, camera distance, head pose, and lighting |
| Frame count mismatch | 0 of 1,230 samples have equal acoustic/visual frame counts (429 distinct ratios) | No temporal frame-level multimodal alignment is possible |
| Disk space constraint | Archive extraction requires 33.7 GB vs 13.73 GB free | No full CSV extraction; compact caches required |
| CC-BY-NC-4.0 license | AVDD-TCMI | Non-commercial use only |

### 1.3 Harvard Tweets

- Numeric code → class mapping is **unresolved**: depositor documented 58,405 "depressed" and 41,185 "non-depressed" in a legend beside the data, but this legend does not state which numeric code means which class.
- **Status: BLOCKED_SUPERVISED** until depositor or authoritative documentation resolves the mapping.

### 1.4 DAIC-WOZ / RSDD / SMHD

- Not acquired. Require institutional Data Use Agreement and signed documentation.
- **Status: ACCESS_PENDING**

### 1.5 FER-2013 (Local Copy)

- Training split missing 3 of 7 emotion classes (neutral, sad, surprise).
- 9,034 images present vs ~28,709 expected.
- **Status: BLOCKED** — cannot train a valid seven-class emotion classifier.

---

## 2. Modality-Specific Limitations

### 2.1 Text (EATD)

- TF-IDF operates on prompted transcripts, not spontaneous speech. Features may capture topic compliance more than affect.
- Transformer embeddings (E-T2) are frozen; fine-tuning (E-T3) is gated due to 6 GB GPU and 45 positive samples.
- Out-of-fold threshold selection shows wide dispersion across 25 folds — the optimal operating point is not reliably identifiable at this sample size.

### 2.2 Voice (EATD)

- eGeMAPS features (E-V1, E-V2) provide statistically marginal signal (ROC-AUC 0.50–0.55 — confidence intervals include 0.5).
- E-V3 (Wav2Vec2) achieves ROC-AUC 0.6315 ± 0.097, but CI is very wide.
- One session (0.48 s) was excluded from audio modeling; participant retained 2 sessions.
- E-V4 (fine-tuning): Explicitly blocked due to hardware constraints and overfitting risk on a 45-positive development pool.

### 2.3 Acoustic Visual Features (AVDD-TCMI)

- Pre-extracted features reduce interpretability: the acoustic pipeline and visual feature extraction settings are not fully documented.
- Temporal aggregation (mean, std, median, IQR, p05, p95, slope) discards temporal dynamics, which may carry clinical signal.
- No temporal alignment between acoustic and visual streams is possible or attempted.

### 2.4 Facial Expressions

- **Emotion recognition ≠ depression detection.** CK+, FER-2013, and RAF-DB are emotion datasets. Any model trained on them cannot be presented as a validated depression classifier.
- Subject identifiers may be absent in some datasets, preventing leakage-safe subject-level splits.

---

## 3. Statistical Limitations

| Limitation | Consequence |
|---|---|
| Small positive class (45 dev, 24 test) | PR-AUC and calibration estimates are unstable |
| Wide fold-to-fold dispersion | SD of ROC-AUC across 25 folds is 0.09–0.11 — single fold estimates are uninformative |
| All CIs overlap 0.5 (EATD voice) | E-V1 and E-V2 have no defensible evidence of above-chance classification |
| Sample-level bootstrap (AVDD) | Clusters by person unknown; CIs may be liberal |
| Calibration with ≤30 participants per outer fold | ECE is a descriptive statistic only; no calibration claims per protocol |
| No correction for multiple comparisons | Comparison of 6+ fusion variants inflates Type I error |

---

## 4. Fusion and Multimodal Limitations

| Limitation | Consequence |
|---|---|
| Late fusion by probability averaging | Cannot exploit inter-modal correlations; no end-to-end learning |
| Cross-dataset fusion not attempted | EATD and AVDD use different label instruments; pooling is not justified |
| Phase 5H gated | AVDD visual E-A2 awaits Section 6.7 sign-off; E-A4 fusion gated behind E-A2 |
| EATD text+voice fusion: marginal improvement only | +0.0027 ROC-AUC with PR-AUC, BalAcc, and F1 all declining |

---

## 5. Clinical Disclaimer

**NeuroScreen is a research prototype only. No finding in Phase 5 constitutes evidence of clinical validity, diagnostic utility, or generalization to clinical populations.**

- No ground-truth clinical diagnosis was used as a label.
- Self-report SDS scores are proxy measures; clinical DSM-5 criteria were not applied.
- Accuracy figures cannot be compared to clinical screening thresholds without validated clinical datasets.
- All models carry `clinical_validation = false`.
