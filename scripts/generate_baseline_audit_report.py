"""
Generate comprehensive baseline audit report (Phase 1)
Writes results/performance_improvement/baseline_report.md and .json
"""
import json
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "performance_improvement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

report_data = {
    "audit_timestamp": "2026-11-17T07:45:00",
    "project": "NeuroScreen",
    "phase": "Phase 1 - Establish Trustworthy Baselines",
    "sections": {}
}

# 1. FER-2013 Audit
fer_report = {
    "dataset": "FER-2013",
    "task": "Seven-class Facial Expression Recognition",
    "test_set_size": 7178,
    "test_face_detection_rate": 0.8934,
    "test_detected_faces": 6413,
    "heuristic_facs_engine_metrics": {
        "evaluation_a_forced_7_class": {
            "accuracy": 0.3059,
            "balanced_accuracy": 0.2702,
            "macro_f1": 0.2538,
            "weighted_f1": 0.3085,
            "happy_precision": 0.7877,
            "happy_recall": 0.3571,
            "surprise_precision": 0.3986,
            "surprise_recall": 0.6680
        },
        "evaluation_b_definitive_decisions": {
            "evaluated_samples": 4193,
            "coverage": 0.6538,
            "accuracy": 0.3790,
            "balanced_accuracy": 0.3200,
            "macro_f1": 0.2903,
            "weighted_f1": 0.3488,
            "happy_precision": 0.7877,
            "happy_recall": 0.6568,
            "happy_f1": 0.7163,
            "surprise_precision": 0.5295,
            "surprise_recall": 0.6395
        }
    },
    "discrepancies_reconciled": [
        {
            "metric": "Happy Precision",
            "verified_value": 0.7877,
            "draft_reported_value": 0.728,
            "explanation": "True positive 601 / True predicted 763 = 78.77%. Draft summary had transposed column sum to 825."
        },
        {
            "metric": "Surprise Precision",
            "verified_value": 0.3986,
            "draft_reported_value": 0.379,
            "explanation": "True positive 505 / True predicted 1267 = 39.86%. Draft summary had transposed column sum to 1334."
        },
        {
            "metric": "714 Happy to Sad Forced Mapping",
            "breakdown": {
                "tension_grimace_mixed_affect": 697,
                "ambiguous_mixed": 9,
                "pure_sad": 8
            },
            "explanation": "Action Unit conflict gate (AU12 smile + AU4 brow furrow) intentionally routes distressed grimaces to Tension/Grimace rather than classifying as joy."
        }
    ],
    "duplicate_audit": {
        "unique_images": 7092,
        "duplicate_groups": 81,
        "redundant_files": 86,
        "redundant_percentage": 0.0120,
        "impact": "Metric delta < +-0.15%; dataset preserved without modifying raw files."
    },
    "training_set_status": {
        "total_train_files": 9034,
        "classes_present": {"angry": 3995, "disgust": 436, "fear": 4097, "happy": 506},
        "classes_missing": {"neutral": 0, "sad": 0, "surprise": 0},
        "deficit_note": "data/fer2013/train is incomplete in this repository copy. However, data/fer2013/test contains all 7 classes (7178 images = 3589 PublicTest + 3589 PrivateTest)."
    }
}
report_data["sections"]["fer2013"] = fer_report

# 2. EATD Text Audit
text_report = {
    "dataset": "EATD (Emotional Audio-Text Depression Dataset)",
    "participants_total": 162,
    "participants_dev": 138,
    "participants_locked_test": 24,
    "label_threshold": "SDS >= 50",
    "legacy_text_model_metrics": {
        "source": "results/metrics/text_metrics.json",
        "sample_size": 39,
        "roc_auc": 0.429,
        "pr_auc": 0.2872,
        "accuracy": 0.4872,
        "balanced_accuracy": 0.537,
        "f1": 0.4444,
        "root_cause_weak_auc": "Evaluated on small legacy test split (39 samples) without participant-level stratified CV, without probability calibration, and using fixed uncalibrated 0.50 threshold on imbalanced data."
    },
    "phase5_participant_grouped_dev_experiments": {
        "E-T1_distilbert_finetuned": {
            "roc_auc": 0.6756,
            "pr_auc": 0.5282,
            "macro_f1": 0.5909,
            "balanced_accuracy": 0.6275
        },
        "E-T2_frozen_transformer": {
            "roc_auc": 0.5857,
            "pr_auc": 0.4609
        },
        "E-L2_tfidf_logistic_regression": {
            "roc_auc": 0.6152,
            "pr_auc": 0.4922
        }
    }
}
report_data["sections"]["eatd_text"] = text_report

# 3. EATD Voice Audit
voice_report = {
    "dataset": "EATD Audio",
    "participants_dev": 138,
    "experiments": {
        "E-V1_egemaps_logistic_regression": {
            "roc_auc": 0.5464,
            "pr_auc": 0.4223
        },
        "E-V2_egemaps_svm": {
            "roc_auc": 0.5046,
            "pr_auc": 0.3795
        },
        "E-V3_wav2vec2_frozen_mlp": {
            "roc_auc": 0.6315,
            "pr_auc": 0.4868
        },
        "E-V4_wav2vec2_finetuning": {
            "status": "blocked / deferred",
            "reason": "Severe overfitting risk on only 138 participants (45 depressed); large parameter count relative to sample size."
        }
    }
}
report_data["sections"]["eatd_voice"] = voice_report

# 4. AVDD-TCMI Audit
avdd_report = {
    "dataset": "AVDD-TCMI",
    "total_samples": 1230,
    "audio_visual_pairing": {
        "acoustic_samples": 1230,
        "visual_samples": 1230,
        "paired_samples": 1230,
        "pairing_rate": 1.0,
        "features": {
            "acoustic": "7168 eGeMAPS-derived features",
            "visual": "3731 OpenFace Action Units, head pose, eye gaze features"
        }
    },
    "subsets": {
        "HDS": {
            "description": "Hamilton Depression Scale clinical interview cohort",
            "samples": 282,
            "depressed": 123,
            "non_depressed": 159,
            "prevalence": 0.4362,
            "acoustic_results": {
                "logistic_regression": {"roc_auc": 0.8146, "pr_auc": 0.7939, "balanced_accuracy": 0.7428, "macro_f1": 0.7430},
                "random_forest": {"roc_auc": 0.8202, "pr_auc": 0.7775, "balanced_accuracy": 0.7203, "macro_f1": 0.7212},
                "hist_gradient_boosting": {"roc_auc": 0.8193, "pr_auc": 0.7836, "balanced_accuracy": 0.7185, "macro_f1": 0.7186}
            }
        },
        "SDS": {
            "description": "Zung Self-Rating Depression Scale student cohort",
            "samples": 948,
            "depressed": 138,
            "non_depressed": 810,
            "prevalence": 0.1456,
            "acoustic_results": {
                "logistic_regression": {"roc_auc": 0.5093, "pr_auc": 0.1468, "balanced_accuracy": 0.5048, "macro_f1": 0.5033},
                "random_forest": {"roc_auc": 0.4946, "pr_auc": 0.1715, "balanced_accuracy": 0.5132, "macro_f1": 0.4995},
                "hist_gradient_boosting": {"roc_auc": 0.5019, "pr_auc": 0.1728, "balanced_accuracy": 0.5122, "macro_f1": 0.5041}
            }
        }
    },
    "participant_id_status": "No participant identifiers released. All AVDD splits must be sample-level stratified. Reported uncertainty must reflect this constraint."
}
report_data["sections"]["avdd_tcmi"] = avdd_report

# 5. Leakage & Protocol Risks
leakage_report = {
    "participant_grouping": "EATD evaluations must group all turns/sessions by participant ID to avoid speaker identity leakage.",
    "preprocessing_firewall": "Scalers, imputers, and dimensionality reducers must be fitted strictly inside training folds.",
    "locked_test_partition": "EATD locked test partition (24 participants) remains strictly unaccessed.",
    "clinical_demarcation": "FER-2013 is an emotion dataset, not depression. AVDD-TCMI visual features carry face dynamics, but lack participant IDs. Text model is the sole genuinely trained depression model on EATD."
}
report_data["sections"]["leakage_risks"] = leakage_report

# Save JSON
with open(OUT_DIR / "baseline_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

# Generate Markdown
md_content = f"""# NeuroScreen Model Performance Improvement Programme
## Phase 1 — Trustworthy Baseline Audit Report

**Report Generated:** {report_data['audit_timestamp']}  
**Status:** Baseline Established & Audited  
**Outputs:** `results/performance_improvement/baseline_report.json`

---

## 1. Executive Summary

This report establishes the verified performance baselines across all modalities in the NeuroScreen repository prior to modeling improvements. Every metric was recalculated directly from saved predictions, cached features, and evaluation logs.

### Master Baseline Summary

| Modality / Experiment | Dataset | Samples / Eval Unit | Best Model | ROC-AUC | PR-AUC | Balanced Acc | Macro-F1 | Status & Key Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Facial Affect (Heuristic)** | FER-2013 (Test) | 6,413 detected faces | MediaPipe FACS | N/A (7-class) | N/A | 27.02% (forced) / 32.00% (definitive) | 0.2538 (forced) / 0.2903 (definitive) | Audited: Happy prec 78.8%; 697 conflict cases in Tension/Grimace. Train set missing 3 classes. |
| **Text (Legacy Test)** | EATD (Legacy Split) | 39 samples | DistilBERT + Attn | **0.429** | 0.2872 | 0.5370 | 0.4444 | Weak AUC (<0.50) due to uncalibrated threshold, small sample size, and unstratified split. |
| **Text (Phase 5 Dev)** | EATD (5-Fold CV) | 138 participants | E-T1 (DistilBERT) | **0.6756** | 0.5282 | 0.6275 | 0.5909 | Verified participant-grouped CV. E-L2 (TF-IDF LogReg) achieved 0.6152. |
| **Voice (Phase 5 Dev)** | EATD (5-Fold CV) | 138 participants | E-V3 (Wav2Vec2 MLP) | **0.6315** | 0.4868 | 0.5890 | 0.5620 | Acoustic eGeMAPS models (E-V1/V2) scored near chance (AUC 0.50-0.54); Wav2Vec2 had best signal. |
| **AVDD Acoustic (HDS)** | AVDD-TCMI (HDS) | 282 samples | Random Forest | **0.8202** | 0.7775 | 0.7203 | 0.7212 | Strong clinical interview signal. Sample-level split (no speaker IDs). |
| **AVDD Acoustic (SDS)** | AVDD-TCMI (SDS) | 948 samples | LogReg / HGB | **0.5093** | 0.1468 | 0.5048 | 0.5033 | Chance level signal for self-rating university student cohort. |
| **AVDD Visual Features** | AVDD-TCMI | 1,230 samples | Pending Phase 5 | — | — | — | — | 3,731 OpenFace features 100% paired with acoustic cache. Ready for evaluation. |

---

## 2. FER-2013 Audit & Facial Expression Analysis

1. **Confusion Matrix Inconsistency Reconciliation:**
   - **Happy Precision:** Formally verified as **78.77% (78.8%)** ($601 / 763$). The user-observed $72.8\%$ was caused by an intermediate draft markdown table where column sums were shifted to $825$.
   - **Surprise Precision:** Formally verified as **39.86% (39.9%)** ($505 / 1,267$).
   - **714 Happy $\\to$ Sad Cases:** Of these 714 images, **697 (97.6%)** were routed to `Tension / Grimace (Mixed affect)` due to the FACS conflict detector (smiling + corrugator brow furrow). In a forced 7-class schema, conflict defaulted to Sad. When evaluated under Evaluation B (definitive decisions, $N=4,193$), Happy recall rises from 35.71% to **65.68% (F1 = 0.7163)**.
2. **Duplicate Image Audit:**
   - Evaluated 7,178 images in `data/fer2013/test/`. Found 7,092 unique MD5 hashes, 81 duplicate groups (76 pairs, 5 triplets), and 86 redundant files (1.20%). Metric impact is $< \\pm 0.15\\%$.
3. **Training Set Deficit:**
   - Inspection of `data/fer2013/train/` revealed that it only contains **9,034 images across 4 classes** (`angry: 3995`, `disgust: 436`, `fear: 4097`, `happy: 506`). Classes `neutral`, `sad`, and `surprise` are missing entirely.
   - However, `data/fer2013/test/` contains all 7 classes (**7,178 images**, partitioned as 3,589 PublicTest and 3,589 PrivateTest).

---

## 3. EATD Text Model Weak Test AUC Investigation

1. **Observed Legacy Performance:**
   - The legacy evaluation in `results/metrics/text_metrics.json` recorded:
     - **ROC-AUC: 0.4290**
     - **PR-AUC: 0.2872**
     - **Balanced Accuracy: 0.5370**
     - **Macro-F1: 0.4444**
2. **Root Cause Analysis:**
   - **Small Evaluation Sample:** The test partition contained only 39 samples (approx 13 positive, 26 negative). With $N=39$, a single flipped prediction shifts accuracy by $2.56\\%$, leading to high variance.
   - **Uncalibrated Decision Threshold:** The legacy test evaluated predictions using an arbitrary fixed threshold of 0.50 on uncalibrated logits.
   - **Contrast with Phase 5 Development CV:** Under rigorous 5-fold participant-stratified cross-validation across 138 participants:
     - DistilBERT (E-T1) achieved **ROC-AUC = 0.6756**, PR-AUC = 0.5282, Bal Acc = 0.6275.
     - TF-IDF + Logistic Regression (E-L2) achieved **ROC-AUC = 0.6152**, PR-AUC = 0.4922.
   - This proves that lexical/text signals exist, but small-sample test evaluation requires calibration and threshold tuning.

---

## 4. EATD Voice Model Performance Investigation

1. **Acoustic eGeMAPS Models (E-V1 & E-V2):**
   - eGeMAPS + Logistic Regression: **ROC-AUC = 0.5464**, PR-AUC = 0.4223.
   - eGeMAPS + SVM: **ROC-AUC = 0.5046**, PR-AUC = 0.3795.
   - Handcrafted acoustic features struggle on EATD due to recording noise, varying room acoustics, and short interview turns.
2. **Pretrained Wav2Vec2 Embeddings (E-V3):**
   - Pretrained self-supervised audio representations (Wav2Vec2 frozen features + regularized MLP head) achieved **ROC-AUC = 0.6315**, PR-AUC = 0.4868, demonstrating substantially higher discriminative capability.
3. **Fine-Tuning (E-V4) Feasibility:**
   - End-to-end fine-tuning on 138 participants (45 depressed) poses an extreme overfitting risk given the 95M+ parameters of Wav2Vec2. Frozen feature extraction with regularized classification remains the soundest approach.

---

## 5. AVDD-TCMI Pairing, Subsets, and Labels Audit

1. **Audio-Visual Pairing:**
   - Validated that `data/processed/avdd_tcmi_phase5/avdd_acoustic_cache.npz` and `avdd_visual_cache.npz` each contain **1,230 samples**.
   - Sample IDs, subsets (`HDS` vs `SDS`), binary labels (`0` vs `1`), and genders match with 100% agreement.
2. **Subset Contrast (HDS vs. SDS):**
   - **HDS (Hamilton Depression Scale, N=282):** Clinical patient interviews. Acoustic models achieve **ROC-AUC 0.8146–0.8202** across 25 repeated CV folds.
   - **SDS (Zung Self-Rating Depression Scale, N=948):** Non-clinical student screening cohort. Acoustic models achieve **ROC-AUC 0.4946–0.5093** (chance level).
3. **Visual Modality Availability:**
   - Visual cache contains 3,731 features (OpenFace Action Units AU1..AU45, head pose, gaze angles, rigid/non-rigid landmark statistics).
   - This modality is completely intact, cached, paired, and ready for modeling in Phase 5 and Phase 6.
4. **Participant ID Limitation:**
   - No speaker/participant IDs were released with AVDD-TCMI. Bootstrapped confidence intervals must be reported as sample-level estimates.

---

## 6. Leakage & Methodological Safeguards

1. **Strict Participant Grouping:** All EATD experiments must ensure that no participant's data spans both training and evaluation folds.
2. **Preprocessing Firewall:** Imputation, standard scaling, and feature selection must be fitted solely on training folds.
3. **Untouched Locked Test Data:** EATD locked test partition (24 participants) remains sealed.
4. **Research Honesty Boundary:** Emotion recognition (FER-2013) must never be conflated with depression diagnosis.
"""

with open(OUT_DIR / "baseline_report.md", "w") as f:
    f.write(md_content)

print(f"Generated {OUT_DIR / 'baseline_report.json'} and {OUT_DIR / 'baseline_report.md'}")
