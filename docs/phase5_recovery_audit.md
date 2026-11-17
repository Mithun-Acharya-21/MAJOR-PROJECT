# Phase 5 — Comprehensive Recovery & Research Audit Report

**Date:** 2026-11-16  
**Project:** NeuroScreen (Multimodal Depression-Risk Screening Research)  
**Auditor:** Senior ML Research Auditor & Systems Engineer  
**Repository:** `C:\Users\mh129\majorProject`  
**Git Working Tree:** Pre-existing dirty tree preserved (296+ tracked/untracked changes strictly respected; no destructive Git commands executed).

---

## 1. System & Resource Environment

- **Operating System:** Windows 11
- **Python Environment:** Python 3.12 (`.\.venv\Scripts\python.exe`) / System Python 3.14
- **PyTorch / CUDA:** PyTorch `2.13.0+cu130`, CUDA available (`True`), GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- **Disk Free:** 13.73 GB (Full extraction of AVDD raw CSVs ~33.7 GB is prohibited; streaming/compact caches enforced)
- **RAM Available:** 3.46 GB free out of 15.71 GB total
- **Active Processes:** No lingering background Python training processes

---

## 2. Phase-by-Phase Status Matrix

| Phase | Description | Audit Status | Key Artifacts / Evidence | Next Action / Blocking Condition |
|---|---|---|---|---|
| **5A** | Research Protocol & Preregistration | **COMPLETE** | `docs/phase_5_research_protocol.md`, `docs/phase_5_experiment_matrix.md`, `data/reports/phase_5_experiment_inventory.json` | Protocol defines RQ1–RQ7, unit of observation, split preservation, locked test firewall. |
| **5B** | Evaluation Harness & Leakage Protection | **COMPLETE** | `ml/evaluation/`, `tests/test_phase5_*.py` (**114 passed, 0 failed**) | Leakage checks, participant grouping, threshold policy, and locked-test firewall operational. |
| **5C** | EATD Text Experiments | **COMPLETE (Core)** | `results/phase5/eatd/E-T1/`, `E-L2/`, `E-T2/` | E-T1 (ROC-AUC 0.6756, PR-AUC 0.5282); E-T3 (fine-tuning) is **OPTIONAL_COST_GATED**. |
| **5D** | EATD Voice Experiments | **COMPLETE (Core)** | `results/phase5/eatd_voice/E-V1/`, `E-V2/`, `E-V3/`, `E-V4/` | E-V3 (ROC-AUC 0.6315, PR-AUC 0.4868); E-V4 is **BLOCKED** due to 6GB GPU and overfitting risk. |
| **5E** | EATD Paired Error Analysis & Late Fusion | **COMPLETE** | `results/phase5/eatd_fusion/`, `participant_level_oof_predictions.csv`, `report.md` | Late fusion gives ROC-AUC 0.6783 vs 0.6756, but PR-AUC, BalAcc, and F1 drop; **fusion not superior**. |
| **5F** | AVDD-TCMI Feature Experiments | **PARTIAL** | Acoustic cache `(1230, 7168)`, Visual cache `(1230, 3731)`, HDS E-A1 complete | **HDS E-A1 complete** (ROC-AUC ~0.820); **SDS E-A1 partial/interrupted**; **E-A2 requires §6.7 owner sign-off**. |
| **5G** | Facial-Image Experiments | **BLOCKED** | `data/fer2013/validation_report.json`, `data/raw/face/` | FER-2013 train set missing 3 classes; CK+ and RAF-DB unacquired. |
| **5H** | Gated Multimodal Fusion | **BLOCKED_GATED** | Protocol Section 6.6 gate definition | Unimodal visual models await Section 6.7 sign-off; statistical non-redundancy gate not yet met. |
| **Ext** | Harvard Tweets | **BLOCKED** | `docs/harvard_tweets_label_audit.md` | Label code 0 vs 1 mapping ambiguous; supervised training prohibited. |
| **Ext** | DAIC-WOZ / RSDD / SMHD | **ACCESS_PENDING** | `docs/phase_4_access_status.md` | Institutional DUA required. |
| **Conf**| Final Locked-Test Confirmation | **BLOCKED** | Protocol Section 4.9 & Firewall | Firewall locked. Preregistration and explicit owner authorization required. |

---

## 3. Detailed Audit Findings

### 3.1 EATD Text & Voice (Phases 5C & 5D)
All primary experiments (E-T1, E-L2, E-T2, E-V1, E-V2, E-V3) are fully measured across 25 outer folds (5 outer folds × 5 repeat seeds 42–46) on the 138-participant development pool.
- **E-T1 (TF-IDF + LR)**: ROC-AUC = 0.6756 ± 0.0893, PR-AUC = 0.5282 ± 0.1039, Balanced Accuracy = 0.6275.
- **E-V3 (Frozen Chinese Wav2Vec2 + Head)**: ROC-AUC = 0.6315 ± 0.0976, PR-AUC = 0.4868 ± 0.1002, Balanced Accuracy = 0.5596.
- **E-T3 (Transformer Fine-Tuning)**: Gated due to resource constraints and limited dev sample size (45 positives).
- **E-V4 (Wav2Vec2 Fine-Tuning)**: Documented refusal due to 6 GB GPU memory ceiling and severe risk of overfitting.

### 3.2 EATD Late Fusion (Phase 5E)
- **Persisted Artifacts Added**: Generated and validated `participant_level_oof_predictions.csv` and `participant_level_oof_predictions.json` in `results/phase5/eatd_fusion/oof_predictions/` without modifying original session-level manifests.
- **Finding**: Late probability fusion (`E-T1 + E-V3 equal`) yields ROC-AUC 0.6783 ± 0.0927 vs 0.6756 ± 0.0893 (+0.0027, within fold SD). Balanced accuracy falls from 0.6275 to 0.6075, and Macro-F1 falls from 0.5909 to 0.5784. Multimodal fusion does not show empirical superiority over unimodal text.

### 3.3 AVDD-TCMI Cache & Acoustic Experiments (Phase 5F)
- **Acoustic Cache**: `data/processed/avdd_tcmi_phase5/avdd_acoustic_cache.npz` (1,230 samples × 7,168 features) verified finite and intact.
- **Visual Cache**: `data/processed/avdd_tcmi_phase5/avdd_visual_cache.npz` (1,230 samples × 3,731 features) verified finite and intact.
- **HDS E-A1**: Complete. Logistic Regression ROC-AUC 0.815 ± 0.012, Random Forest ROC-AUC 0.820 ± 0.006, HistGradientBoosting ROC-AUC 0.819 ± 0.015.
- **SDS E-A1**: Partial / interrupted. Completed repeat 0 (folds 1-5) before previous session restart.
- **Visual Pre-Training Review Gate (Section 6.7)**: Explicit owner sign-off is required before fitting E-A2 on visual features due to OpenFace coordinate confounding and absence of acoustic feature semantics.

### 3.4 Facial Image Datasets (Phase 5G)
- Local FER-2013 training split contains only 9,034 images (missing neutral, sad, surprise).
- CK+ and RAF-DB are empty (`.gitkeep`).
- Branch is strictly marked **BLOCKED**.

---

## 4. Preservation & Integrity Verification

1. Git working tree preserved: all modifications remain intact.
2. Locked test set: 24 participants protected behind the Phase 5B firewall; zero test evaluations performed.
3. Repositories and models: no Phase 2 checkpoints overwritten; all Phase 5 artifacts independently stored in `results/phase5/`.
