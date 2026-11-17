# Phase 5 — Experiment Execution Log

**Date:** 2026-11-16  
**Project:** NeuroScreen  
**Execution Environment:** Windows, Python 3.12 (`.\.venv\Scripts\python.exe`), PyTorch 2.13.0+cu130, RTX 3050 6GB Laptop GPU  

---

## 1. Execution History & Provenance

### Phase 5A: Research Design and Protocol Definition
- **Script/Design Artifacts:** `docs/phase_5_research_protocol.md`, `docs/phase_5_experiment_matrix.md`, `docs/phase_5_statistical_evaluation_plan.md`, `docs/phase_5_risk_register.md`.
- **Outcome:** Research questions RQ1–RQ7 codified. Frozen reference results from Phase 2 preserved. Locked test set strictly ring-fenced.

### Phase 5B: Evaluation Harness & Leakage Protection
- **Modules Created:** `ml/evaluation/` (provenance, splits, nested_cv, thresholds, calibration, bootstrap, firewall, artifacts, config, metrics).
- **Test Execution:**
  - Command: `.\.venv\Scripts\python.exe -m pytest tests/test_phase5_artifacts.py tests/test_phase5_firewall.py tests/test_phase5_leakage.py tests/test_phase5_metrics.py tests/test_phase5_splits.py`
  - Result: **114 passed, 1 warning (due to single-item class in synthetic test fixture), 0 failed** in 35.34s.
- **Outcome:** Verified participant-grouped splitting, 0-leakage outer/inner folds, Platt scaling calibration, and tamper-resistant firewall.

### Phase 5C: EATD Text Unimodal Experiments
- **Protocol:** Repeated nested stratified group CV (5 folds × seeds 42–46, 4 inner folds, participant-grouped).
- **Execution:**
  - `E-T1` (TF-IDF + Logistic Regression): Completed. Mean ROC-AUC: 0.6756 ± 0.0893, PR-AUC: 0.5282 ± 0.1039, Balanced Accuracy: 0.6275.
  - `E-L2` (Metadata/Confound Baseline): Completed. Mean ROC-AUC: 0.6152 ± 0.1098, PR-AUC: 0.4922 ± 0.1054. Confirms metadata provides substantial confounding signal.
  - `E-T2` (Frozen Multilingual Transformer + Classification Head): Completed. Mean ROC-AUC: 0.5857 ± 0.0881, PR-AUC: 0.4609 ± 0.0944.
  - `E-T3` (Top-2 Layer Transformer Fine-Tuning): Status **OPTIONAL_COST_GATED**. Gated by hardware constraints and protocol resource check.

### Phase 5D: EATD Voice Unimodal Experiments
- **Protocol:** Repeated nested stratified group CV (5 folds × seeds 42–46, participant-grouped).
- **Execution:**
  - `E-V1` (eGeMAPS + Logistic Regression): Completed. Mean ROC-AUC: 0.5464 ± 0.0958, PR-AUC: 0.4223 ± 0.0948.
  - `E-V2` (eGeMAPS + RBF SVM): Completed. Mean ROC-AUC: 0.5046 ± 0.1096, PR-AUC: 0.3795 ± 0.0754.
  - `E-V3` (Frozen Chinese Wav2Vec2 + MLP Head): Completed. Mean ROC-AUC: 0.6315 ± 0.0976, PR-AUC: 0.4868 ± 0.1002.
  - `E-V4` (Fine-tuned Wav2Vec2): Status **BLOCKED** due to 6 GB GPU memory ceiling and severe overfitting risk.

### Phase 5E: EATD Paired Error Analysis and Late Fusion
- **Execution:**
  - Script: `scripts/phase5/run_eatd_fusion.py`
  - Persistence Reproduction Script: `scripts/phase5/build_participant_oof.py`
  - Artifacts Generated:
    - `results/phase5/eatd_fusion/oof_predictions/participant_level_oof_predictions.csv`
    - `results/phase5/eatd_fusion/oof_predictions/participant_level_oof_predictions.json`
    - `results/phase5/eatd_fusion/report.md`
  - Findings: Equal-weight late fusion of E-T1 + E-V3 achieved ROC-AUC 0.6783 ± 0.0927 vs 0.6756 ± 0.0893 for text-only E-T1. However, PR-AUC decreased (0.5282 → 0.5270), Balanced Accuracy fell (0.6275 → 0.6075), and Macro-F1 fell (0.5909 → 0.5784). Conclusion: Multimodal fusion does not show empirical superiority.

### Phase 5F: AVDD-TCMI Pre-Extracted Feature Experiments
- **Feature Caches:**
  - Acoustic: `data/processed/avdd_tcmi_phase5/avdd_acoustic_cache.npz` (1,230 samples × 7,168 features)
  - Visual: `data/processed/avdd_tcmi_phase5/avdd_visual_cache.npz` (1,230 samples × 3,731 features)
- **Execution:**
  - `HDS E-A1` (Acoustic 282 samples): Completed.
    - Logistic Regression: ROC-AUC 0.815 ± 0.012, PR-AUC 0.794, BalAcc 0.743, F1 0.709
    - Random Forest: ROC-AUC 0.820 ± 0.006, PR-AUC 0.777, BalAcc 0.720, F1 0.681
    - HistGradientBoosting: ROC-AUC 0.819 ± 0.015, PR-AUC 0.784, BalAcc 0.718, F1 0.682
  - `SDS E-A1` (Acoustic 948 samples): Status **PARTIAL_INTERRUPTED**. Folds 1–5 (repeat 0) completed before server restart.
  - `E-A2` (Visual Features): Status **REQUIRES_OWNER_APPROVAL** per Protocol Section 6.7.
  - `E-A3` (Feature Analysis): Status **NOT_STARTED** (pending E-A1 and E-A2 completion).

### Phase 5G: Facial-Image Experiments
- **Audit Findings:** Local copy of FER-2013 is missing classes (`neutral`, `sad`, `surprise`) and contains only 9,034 images. CK+ and RAF-DB are unacquired (`.gitkeep` only).
- **Status:** **BLOCKED** due to missing/corrupted training data.

### Phase 5H: Gated Multimodal Fusion
- **Status:** **BLOCKED_GATED**. Gates in Protocol Section 6.6 require validated unimodal branches with non-overlapping errors and non-chance confidence intervals.
