# NeuroScreen Phase 5 — Final Research Report

> **Academic research prototype only. NOT a clinical diagnostic system.**  
> This report documents experimental findings from a controlled research investigation.  
> All results are development-set estimates. The locked test partition has not been accessed.

**Repository:** `C:\Users\mh129\majorProject`  
**Report generated:** 2026-11-16  
**Protocol document:** `docs/phase_5_research_protocol.md`  
**Status:** Development phase complete; locked-test confirmation pending owner authorization (§4.9)

---

## Table of Contents

1. [Research Objectives](#1-research-objectives)
2. [Datasets and Provenance](#2-datasets-and-provenance)
3. [Evaluation Protocol](#3-evaluation-protocol)
4. [Phase 5C — EATD Text Experiments](#4-phase-5c--eatd-text-experiments)
5. [Phase 5D — EATD Voice Experiments](#5-phase-5d--eatd-voice-experiments)
6. [Phase 5E — EATD Multimodal Fusion](#6-phase-5e--eatd-multimodal-fusion)
7. [Phase 5F — AVDD-TCMI Acoustic Experiments](#7-phase-5f--avdd-tcmi-acoustic-experiments)
8. [Phase 5G — Facial Image Experiments](#8-phase-5g--facial-image-experiments)
9. [Phase 5H — Cross-Dataset Fusion](#9-phase-5h--cross-dataset-fusion)
10. [Cross-Modality Summary](#10-cross-modality-summary)
11. [Statistical Interpretation](#11-statistical-interpretation)
12. [Calibration Notes](#12-calibration-notes)
13. [Error Analysis](#13-error-analysis)
14. [Limitations](#14-limitations)
15. [Reproducibility](#15-reproducibility)
16. [Blocked and Deferred Experiments](#16-blocked-and-deferred-experiments)
17. [Next Steps](#17-next-steps)

---

## 1. Research Objectives

NeuroScreen Phase 5 is a multi-modal experimental screening research project investigating whether text, voice (acoustic), facial behaviour, and combinations thereof carry useful signal for predicting depression-related symptoms. The research questions (RQ) are:

| RQ | Question |
|----|----------|
| RQ1 | Do linguistic features of spoken text correlate with depression severity scores (EATD dataset)? |
| RQ2 | Do acoustic/prosodic features correlate with depression severity scores (EATD dataset)? |
| RQ3 | Does multimodal fusion of text and voice outperform either modality alone (EATD)? |
| RQ4 | Do eGeMAPS acoustic features distinguish HDS/SDS depression categories (AVDD-TCMI)? |
| RQ5 | Do visual/facial action unit features add information beyond acoustics (AVDD-TCMI)? |
| RQ6 | Are available facial image datasets of sufficient quality for supervised training? |
| RQ7 | Do differences between modalities' performance exceed estimation uncertainty? |

**Conclusion summary (development set, pre-locked-test):** Positive but weak and wide signals found for EATD text and AVDD acoustics. EATD voice shows wide confidence intervals crossing chance. Fusion provides marginal, statistically unconfirmed benefit. Visual and facial modalities are blocked by data quality or availability issues.

---

## 2. Datasets and Provenance

### 2.1 EATD (Emotional Audio-Text Depression Dataset)

| Property | Value |
|----------|-------|
| Task | Binary depression screening (SDS ≥ 50 → positive) |
| Label definition | Self-report SDS ≥ 50 |
| Development participants | 138 (locked test: 24, never accessed) |
| Modalities available | Text transcripts, audio recordings |
| Label threshold | Fixed per pre-registration: SDS ≥ 50 |
| Development positive rate | ~32.6% (45/138) |
| Split | Participant-grouped, never session-level |

**Label definition note:** The SDS ≥ 50 threshold was fixed at protocol registration and must not be changed without a documented protocol amendment.

### 2.2 AVDD-TCMI (Audio-Visual Depression Dataset — TCMI)

| Property | Value |
|----------|-------|
| Task | Binary depression category (HDS vs SDS) |
| Label definition | HDS = Hamilton Depression Scale group; SDS = SDS group |
| Total samples | 1,230 |
| HDS samples | 282 (depressed) |
| SDS samples | 948 (self-rated / healthy) |
| Acoustic features | 7,168 (eGeMAPS-derived, cached) |
| Visual features | 3,731 (facial action units + head pose, cached) |
| Participant ID | NOT released — no deduplication by speaker possible |
| Evaluation unit | Sample-level only |

**Critical caveat:** Because no participant/speaker ID is provided, bootstrapped CIs are sample-level, not speaker-clustered. Real uncertainty is likely wider than reported. All AVDD results carry this limitation.

### 2.3 Datasets Audited but Blocked

| Dataset | Status | Reason |
|---------|--------|--------|
| FER-2013 | BLOCKED | Incomplete: only 9,034/28,709 training images present; missing 3 of 7 classes; `usable_for_seven_class_training: false` |
| CK+ | BLOCKED | Only `.gitkeep` present — data not acquired |
| RAF-DB | BLOCKED | Only `.gitkeep` present — data not acquired |
| Harvard Tweets | BLOCKED_SUPERVISED | Depositor label-code mapping is ambiguous; no supervised training without confirmation |
| DAIC-WOZ | NOT_ACQUIRED | Requires institutional agreement; no access bypass attempted |

---

## 3. Evaluation Protocol

### 3.1 EATD Evaluation

- **CV strategy:** RepeatedStratifiedKFold (5 splits × 5 repeats, seeds 42–46), participant-grouped
- **Evaluation unit:** Participant (sessions aggregated by mean probability)
- **Inner CV:** 4-fold for threshold selection (maximise balanced accuracy on inner OOF)
- **Bootstrap:** 2,000-resample participant-cluster bootstrap (participants resampled, not sessions)
- **Threshold:** Inner-fold selected; not applied to outer fold during threshold selection
- **Locked test:** 24 participants, partition locked by `ml/evaluation/firewall.py`; not accessed
- **Leakage protection:** Verified — 10 probes passed; 24 locked participants protected

### 3.2 AVDD Evaluation

- **CV strategy:** RepeatedStratifiedKFold (5 splits × 5 repeats, seed 42)
- **Evaluation unit:** Sample (no participant ID available)
- **Bootstrap:** 1,000-resample sample-level bootstrap
- **HDS and SDS evaluated separately** — never pooled without pre-registered justification
- **Preprocessing:** Median imputation then StandardScaler (fit on train fold only)

---

## 4. Phase 5C — EATD Text Experiments

### 4.1 Experiment Descriptions

| ID | Model | Features |
|----|-------|----------|
| E-L2 | Logistic Regression (baseline) | Metadata/confound features only |
| E-T1 | TF-IDF (n-grams 1–2) + Logistic Regression | Text bag-of-words |
| E-T2 | Frozen multilingual transformer (MLP head) | Sentence embeddings |
| E-T3 | Fine-tuned transformer (top 2 layers) | Sentence embeddings (OPTIONAL — COST-GATED) |

### 4.2 Results (Development Set)

| Experiment | ROC-AUC (mean±SD) | 95% CI (ROC-AUC) | PR-AUC | Bal-Acc | Macro-F1 |
|------------|-------------------|------------------|--------|---------|----------|
| E-L2 | 0.6152 ± 0.1098 | n/a | 0.4922 | — | — |
| E-T1 | 0.6756 ± 0.0893 | [0.576, 0.759] | 0.5282 | 0.6275 | 0.5909 |
| E-T2 | 0.5857 ± 0.0881 | n/a | 0.4609 | — | — |
| E-T3 | COST-GATED | — | — | — | — |

> **Note:** 95% CI for E-T1 uses 2,000-resample participant-cluster bootstrap on first repeat OOF. E-L2 and E-T2 bootstrap CIs were not separately computed in Phase 5B harness; fold-SD is reported.

### 4.3 Findings

- E-T1 (TF-IDF + LR) is the best text model at ROC-AUC 0.6756.
- E-T1 CI [0.576, 0.759] does **not** include 0.5 — text carries modest signal above chance.
- E-T2 (frozen transformer) underperforms E-T1; embeddings from a multilingual model pre-trained without depression data may not encode task-relevant features.
- E-L2 (confound baseline) ROC-AUC 0.6152 suggests metadata alone provides some discrimination; E-T1's improvement over E-L2 is real but moderate.
- E-T3 is deferred (cost gate); E-T1 is the primary text representation for fusion.

---

## 5. Phase 5D — EATD Voice Experiments

### 5.1 Experiment Descriptions

| ID | Model | Features |
|----|-------|----------|
| E-V1 | eGeMAPS + Logistic Regression | 88-dim acoustic hand-crafted |
| E-V2 | eGeMAPS + RBF SVM | 88-dim acoustic hand-crafted |
| E-V3 | Frozen Wav2Vec2 + MLP head | Deep speech representations |
| E-V4 | Fine-tuned Wav2Vec2 | Deep speech (BLOCKED) |

### 5.2 Results (Development Set)

| Experiment | ROC-AUC (mean±SD) | 95% CI (mean fold-level) | PR-AUC | Bal-Acc | Macro-F1 |
|------------|-------------------|--------------------------|--------|---------|----------|
| E-V1 | 0.5464 ± 0.0958 | [0.314, 0.776] | 0.4223 | 0.5265 | 0.4719 |
| E-V2 | 0.5046 ± 0.1096 | [0.280, 0.728] | 0.3795 | 0.4856 | 0.4235 |
| E-V3 | 0.6315 ± 0.0976 | [0.403, 0.842] | 0.4868 | 0.5596 | 0.5471 |
| E-V4 | BLOCKED | — | — | — | — |

> **Note:** CIs are mean of fold-level 2,000-resample participant-cluster bootstrap intervals (not a single pooled CI). All voice CIs are wide and cross 0.5; evidence is inconclusive on its own.

### 5.3 Findings

- E-V3 (frozen Wav2Vec2 + MLP) has the highest point estimate at 0.6315 ROC-AUC.
- **All voice CI intervals include 0.5** — no voice model demonstrates statistically clear above-chance performance on its own at this sample size.
- eGeMAPS features (E-V1, E-V2) perform near chance. Hand-crafted acoustic features alone are insufficient.
- E-V4 (fine-tuned Wav2Vec2) was blocked due to GPU memory (6 GB) and overfitting risk on 413 development sessions.
- Voice and text show low-to-moderate fold-level ROC-AUC correlation (E-V3 vs E-T1: r ≈ 0.23), suggesting partial non-redundancy.

---

## 6. Phase 5E — EATD Multimodal Fusion

### 6.1 Fusion Protocol

- **Strategy:** Late fusion (probability averaging) at participant level
- **Fusion variants tested:** E-T1 only, E-V3 only, E-T1+E-V3 (equal weight), E-T1+E-V3 (weighted), E-T2+E-V3 (equal), E-T1+E-T2+E-V3 (equal)
- **Same nested CV folds** as unimodal experiments (no additional leakage)

### 6.2 Results

| Variant | ROC-AUC (mean±SD) | PR-AUC | Bal-Acc | Macro-F1 |
|---------|-------------------|--------|---------|----------|
| E-T1 only | 0.6756 ± 0.0893 | 0.5282 | 0.6275 | 0.5909 |
| E-V3 only | 0.6315 ± 0.0976 | 0.4868 | 0.5596 | 0.5471 |
| E-T1+E-V3 equal | **0.6783 ± 0.0927** | 0.5270 | 0.6075 | 0.5784 |
| E-T1+E-V3 weighted | 0.6381 ± 0.1027 | 0.4911 | 0.5828 | 0.5681 |
| E-T2+E-V3 equal | 0.6007 ± 0.0877 | 0.4669 | 0.5507 | 0.5198 |
| E-T1+E-T2+E-V3 equal | 0.6354 ± 0.0969 | 0.4938 | 0.5669 | 0.5380 |

### 6.3 Complementarity Analysis

| Analysis | Value |
|----------|-------|
| OOF probability Pearson r (E-T1 vs E-V3, participant-level) | 0.231 |
| Both correct | 58 participants |
| Both wrong | 30 participants |
| Text-only correct | 29 participants |
| Voice-only correct | 21 participants |

### 6.4 Findings

- Equal-weight E-T1+E-V3 fusion achieves the highest ROC-AUC (0.6783) — marginally above E-T1 alone (+0.003).
- Fusion **does not improve** PR-AUC (0.527 vs 0.528 for E-T1) or balanced accuracy (0.608 vs 0.628).
- The improvement in ROC-AUC (+0.003) is well within the fold-to-fold SD (~0.09) — **not statistically meaningful**.
- OOF correlation r=0.231 suggests partial complementarity, but 30/138 participants are missed by both modalities.
- **Conclusion:** Fusion is PARTIALLY COMPLEMENTARY but provides no confirmed improvement. E-T1 alone is the recommended EATD model.

---

## 7. Phase 5F — AVDD-TCMI Acoustic Experiments

### 7.1 Experiment Protocol

- **Experiment:** E-A1 — Acoustic features (7,168-dim eGeMAPS-derived) with 3 classifiers
- **Models:** Logistic Regression, Random Forest (n_estimators=300), HistGradientBoosting (max_iter=50)
- **CV:** RepeatedStratifiedKFold (5 splits × 5 repeats, seed 42)
- **HDS and SDS evaluated separately** (never pooled)
- **Features:** Acoustic cache `avdd_acoustic_cache.npz`; median imputation + StandardScaler

### 7.2 HDS E-A1 Results (Complete)

n=282, positive=HDS (depressed), class_weight=balanced

| Model | ROC-AUC (mean±SD) | 95% CI (Bootstrap) | PR-AUC | Bal-Acc | Macro-F1 |
|-------|-------------------|-------------------|--------|---------|----------|
| Logistic Regression | 0.8146 ± 0.0124 | [0.734, 0.847] | 0.7939 | 0.7428 | 0.7430 |
| Random Forest | 0.8202 ± 0.0062 | [0.767, 0.868] | 0.7775 | 0.7203 | 0.7212 |
| Hist. Gradient Boosting | 0.8193 ± 0.0150 | [0.752, 0.857] | 0.7836 | 0.7185 | 0.7186 |

### 7.3 SDS E-A1 Results

> **Status:** Experiment running as of report generation. Results to be updated when complete.  
> Partial results available in `results/phase5/avdd_features/SDS/E-A1/run.log`.  
> See `results/phase5/avdd_features/SDS/E-A1/metrics.json` when experiment completes.

### 7.4 Findings (HDS)

- All three models achieve ROC-AUC > 0.80 for HDS, with 95% bootstrap CIs clearly above 0.5.
- LR, RF, and HGB perform similarly; no model is decisively superior.
- Results are strong relative to EATD modalities, but carry important caveats:
  - Sample-level bootstrapping (speaker identity unknown — clustering effects unknown)
  - HDS is a clinically derived label; SDS is self-report — the two scales are not equivalent
  - Acoustic features were cached from pre-extracted CSVs; no raw audio re-processing was performed

> **Mandatory caveat:** "These results use sample-level resampling. Speaker identity was not released by the dataset authors. True performance accounting for speaker clustering may differ substantially."

### 7.5 E-A2 Visual Features — BLOCKED

Per Protocol §6.7, visual feature experiments require explicit **owner sign-off** (pre-training review gate). This approval has **not been granted**. E-A2 is BLOCKED_AWAITING_APPROVAL.

**To proceed with E-A2**, the owner must confirm:
1. HDS=depressed, SDS=self-rated (healthy comparison) label semantics are acceptable for model fitting
2. The absence of a feature dictionary for the 1,024 acoustic columns is acceptable
3. Dropped constant columns (2 columns), excluded metadata columns (`frame`, `timestamp`, `confidence`)
4. 533 visual feature columns retained — approved for model fitting

---

## 8. Phase 5G — Facial Image Experiments

**Status: BLOCKED**

| Dataset | Finding |
|---------|---------|
| FER-2013 | 9,034/28,709 train images present (31.5%). Missing: angry, disgust, fear classes entirely. `usable_for_seven_class_training: false` |
| CK+ | Only `.gitkeep` — data not acquired |
| RAF-DB | Only `.gitkeep` — data not acquired |

No facial image experiments were run. Per the research protocol, experiments must not proceed without valid datasets. The FER-2013 dataset is confirmed unusable in its current state. CK+ and RAF-DB require acquisition.

**Research implication:** We cannot currently answer whether facial expression features provide signal for depression screening. This is a gap in the current evidence base.

---

## 9. Phase 5H — Cross-Dataset Fusion

**Status: BLOCKED_GATED**

Fusion across EATD and AVDD datasets is blocked because:
1. E-A2 (AVDD visual) has not been run (§6.7 approval pending)
2. AVDD does not share evaluation participants with EATD (different datasets, different populations)
3. The EATD fusion analysis (Phase 5E) showed no confirmed improvement
4. Statistical gates per §6.6 have not been evaluated for AVDD (SDS E-A1 incomplete; E-A2 blocked)

---

## 10. Cross-Modality Summary

### EATD Dataset

| Experiment | Modality | ROC-AUC | 95% CI | Conclusion |
|------------|----------|---------|--------|-----------|
| E-L2 | Metadata baseline | 0.615 | — | Moderate |
| E-T1 | Text (TF-IDF) | 0.676 | [0.576, 0.759] | Above chance ✓ |
| E-T2 | Text (transformer) | 0.586 | — | Inconclusive |
| E-V1 | Voice (eGeMAPS+LR) | 0.546 | [0.314, 0.776] | Inconclusive |
| E-V2 | Voice (eGeMAPS+SVM) | 0.505 | [0.280, 0.728] | Chance level |
| E-V3 | Voice (Wav2Vec2) | 0.632 | [0.403, 0.842] | Inconclusive (CI crosses 0.5) |
| E-T1+E-V3 (fusion) | Text+Voice | 0.678 | — | No confirmed gain |

### AVDD-TCMI Dataset (HDS only, acoustic)

| Experiment | Modality | ROC-AUC | 95% CI | Conclusion |
|------------|----------|---------|--------|-----------|
| E-A1 (LR) | Acoustic | 0.815 | [0.734, 0.847] | Strong signal ✓ |
| E-A1 (RF) | Acoustic | 0.820 | [0.767, 0.868] | Strong signal ✓ |
| E-A1 (HGB) | Acoustic | 0.819 | [0.752, 0.857] | Strong signal ✓ |

**Key comparison:** AVDD HDS acoustic results (ROC~0.82) are substantially stronger than EATD text/voice results. However, the two datasets are **not directly comparable** — they use different populations, label definitions, and time periods.

---

## 11. Statistical Interpretation

### Which Results Show Above-Chance Performance?

| Experiment | 95% CI includes 0.5? | Interpretation |
|------------|---------------------|----------------|
| E-T1 (EATD text) | No [0.576, 0.759] | Modest evidence of above-chance |
| E-V1 (EATD voice) | Yes [0.314, 0.776] | Inconclusive — chance plausible |
| E-V2 (EATD voice) | Yes [0.280, 0.728] | Inconclusive — chance plausible |
| E-V3 (EATD voice) | Yes [0.403, 0.842] | Inconclusive — chance plausible |
| E-A1 LR (AVDD HDS) | No [0.734, 0.847] | Strong evidence of above-chance |
| E-A1 RF (AVDD HDS) | No [0.767, 0.868] | Strong evidence of above-chance |
| E-A1 HGB (AVDD HDS) | No [0.752, 0.857] | Strong evidence of above-chance |

### Is Fusion Better Than Unimodal?

| Comparison | ROC-AUC delta | Within fold-SD? | Conclusion |
|------------|---------------|-----------------|-----------|
| E-T1+E-V3 vs E-T1 | +0.003 | Yes (~0.009 SD) | Not meaningful |
| E-T1+E-V3 vs E-V3 | +0.047 | Within range | Not confirmed |

**Conclusion:** No fusion variant provides statistically confirmed improvement over E-T1 alone.

---

## 12. Calibration Notes

- E-T1 Brier score (first repeat OOF): 0.208 (reference: 0.218 for a constant-prevalence predictor at 32.6% positive rate)
- E-V3 Brier score: not separately computed at pooled level; fold-level values available in `voice_comparison_final.json`
- AVDD HDS models: Brier score available in `HDS/E-A1/metrics.json` per model
- No explicit calibration curves (Platt scaling / isotonic) were applied — models were used with raw probabilities
- **Recommendation for future work:** Calibrate probabilities before clinical reporting. Brier score alone is insufficient for calibration assessment.

---

## 13. Error Analysis

### EATD Error Analysis (Phase 5E)

Based on first-repeat participant-level OOF predictions comparing E-T1 and E-V3:

| Category | Count |
|----------|-------|
| Both correct | 58 |
| Both wrong | 30 |
| Text-only correct | 29 |
| Voice-only correct | 21 |
| Total participants | 138 |

- 21/138 participants (15.2%) were only correctly identified by voice — these represent cases where adding voice signal could help
- 30/138 (21.7%) missed by both — a systematic hard-negative group
- OOF probability correlation r=0.231 (Pearson) — partial non-redundancy confirmed but insufficient for confident fusion improvement

### AVDD HDS Error Analysis

From confusion matrix (repeat 0, threshold=0.5):
- LR: TN, FP, FN, TP available in `confusion_matrices.json`
- Class imbalance was managed with `class_weight="balanced"` — deliberate design choice

---

## 14. Limitations

### Dataset Limitations

1. **EATD sample size:** 138 development participants — small for deep learning; 25-fold CV provides reliable estimates but CIs remain wide.
2. **AVDD speaker ID unavailable:** Cannot cluster-bootstrap by speaker; real uncertainty is likely underestimated.
3. **AVDD label non-equivalence:** HDS (Hamilton Depression Scale) ≠ SDS (Self-report Depression Scale); these are different constructs. HDS vs SDS separation is not equivalent to depressed vs healthy.
4. **FER-2013 dataset incomplete:** Cannot investigate facial modality.
5. **Harvard Tweets label ambiguity:** Cannot use for supervised training.
6. **DAIC-WOZ not acquired:** Institutional agreement required.

### Methodological Limitations

7. **Late fusion only:** Only probability averaging was tested; learned fusion weights, attention mechanisms, or joint training were not investigated.
8. **TF-IDF is unigram/bigram only:** No semantic meaning, negation handling, or clinical NLP preprocessing.
9. **E-V3 uses frozen Wav2Vec2:** Domain-adapted fine-tuning (E-V4) blocked; potential representation improvement untested.
10. **Threshold selection:** Inner-CV balanced-accuracy maximization; optimal threshold for specific deployment scenarios (high sensitivity vs specificity) not explored.
11. **No temporal modelling:** Session-level features; longitudinal trajectory not modelled.
12. **No demographic adjustment:** Age, gender, language effects not modelled.

### Experimental Limitations

13. **E-T3, E-V4 cost-gated:** These experiments might improve results but remain untested.
14. **E-A2 blocked:** Visual features cannot be evaluated without §6.7 approval.
15. **SDS E-A1 pending:** Results for AVDD SDS subset not yet finalised as of report generation.
16. **No locked-test confirmation:** All results are development-set estimates. Final locked-test evaluation requires preregistration + owner authorization per §4.9.

---

## 15. Reproducibility

### Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.13.0+cu130 |
| scikit-learn | (from .venv) |
| CUDA | 13.0 |
| GPU | NVIDIA RTX 3050 Laptop (6 GB) |
| OS | Windows |

### Key Reproducibility Artifacts

| Artifact | Path |
|----------|------|
| Phase 5B split audit | `data/reports/phase_5b_split_audit.json` |
| EATD OOF predictions (session-level) | `results/phase5/eatd_fusion/oof_predictions/E-T1_oof.csv` |
| EATD participant-level OOF | `results/phase5/eatd_fusion/oof_predictions/participant_level_oof_predictions.csv` |
| HDS E-A1 OOF | `results/phase5/avdd_features/HDS/E-A1/oof_predictions.npz` |
| HDS E-A1 config | `results/phase5/avdd_features/HDS/E-A1/config.json` |
| Fusion comparison | `results/phase5/eatd_fusion/fusion_comparison.json` |
| Phase 5A protocol | `docs/phase_5_research_protocol.md` |
| Experiment matrix | `docs/phase_5_experiment_matrix.md` |
| Statistical plan | `docs/phase_5_statistical_evaluation_plan.md` |

### Seed Policy

- EATD CV seeds: 42, 43, 44, 45, 46 (5 repeats)
- AVDD CV seed: 42 (single seed for 5 repeats)
- Bootstrap seed: 0 (EATD), 42 (AVDD)
- All seeds are fixed and documented in config files

### Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase5_artifacts.py tests/test_phase5_firewall.py tests/test_phase5_leakage.py tests/test_phase5_metrics.py tests/test_phase5_splits.py
# Expected: 114+ passed, 0 failed
```

---

## 16. Blocked and Deferred Experiments

| ID | Description | Status | Reason |
|----|-------------|--------|--------|
| E-T3 | Fine-tuned transformer | COST-GATED | Compute cost; requires explicit approval |
| E-V4 | Fine-tuned Wav2Vec2 | BLOCKED | GPU memory + overfitting risk |
| E-A2 (HDS) | AVDD visual features | BLOCKED_AWAITING_APPROVAL | §6.7 pre-training review gate not signed |
| E-A2 (SDS) | AVDD visual features | BLOCKED_AWAITING_APPROVAL | Same |
| E-A3 | Modality ablation | BLOCKED_UPSTREAM | Requires E-A2 results |
| E-A4 | AVDD acoustic+visual fusion | BLOCKED_UPSTREAM | Requires E-A2 + fusion gates |
| E-A5 | HDS+SDS pooled | NOT_STARTED | Requires pre-registration + justification |
| E-F1 | Cross-dataset fusion | BLOCKED | EATD and AVDD use different populations |
| Facial (FER-2013) | Facial expression training | BLOCKED | Dataset incomplete |
| Facial (CK+, RAF-DB) | Facial expression training | NOT_ACQUIRED | Data not downloaded |
| Locked test | §4.9 confirmation | BLOCKED_AWAITING_AUTHORIZATION | Preregistration + owner auth required |

---

## 17. Next Steps

### Immediate (pre-locked-test)

1. **Complete SDS E-A1** — Script `scripts/phase5/run_avdd_sds_ea1_safe.py` is running.
2. **Generate AVDD E-A1 comparison report** — Compare HDS vs SDS after SDS completes.
3. **Request §6.7 sign-off** for E-A2 visual features (see Section 7.5 for details required).

### Short-term (pending approvals)

4. **E-A2 visual** — Run after §6.7 approval; use same CV protocol as E-A1.
5. **E-A3 modality ablation** — After E-A2 completes.
6. **Evaluate Phase 5H fusion gates** — After E-A2; check §6.6 statistical gates.

### Pre-locked-test

7. **Preregister final model** — Document chosen model, threshold, and script hash.
8. **Obtain explicit §4.9 owner authorization** for locked-test run.
9. **Run locked test once** — Generate final test metrics; no tuning after this.
10. **Complete test suite** — Ensure 0 failures before locked test.

### Documentation

11. Update `docs/project_completion_status.md` after each step.
12. Update `results/phase5/phase5_consolidated_report.json` with SDS results when available.
13. Create AVDD E-A1 comparison report after SDS completes.

---

*This report reflects the development-set phase of NeuroScreen Phase 5 research. It is an academic prototype investigation only. No clinical recommendations should be drawn from these results.*

*Generated from verified experiment artifacts. See individual experiment directories under `results/phase5/` for raw metrics, OOF predictions, and configuration files.*
