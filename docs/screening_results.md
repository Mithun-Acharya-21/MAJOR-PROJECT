# NeuroScreen — Patient Screening Results & Scoring Specification

**Document Version:** 1.0.0  
**Context:** Multimodal Scoring, Probability Calibration, Uncertainty Bounds, and Clinical Language  
**Target File:** `backend/app/doctor_routes.py`, `frontend/components/doctor/session-detail-view.tsx`

---

## 1. Screening Score & Category Formulation

NeuroScreen computes an experimental screening score using late-fusion stacking across active modalities:

$$S_{\text{composite}} = \sigma\left( w_0 + w_{\text{text}} \hat{p}_{\text{text}} + w_{\text{audio}} \hat{p}_{\text{audio}} + w_{\text{visual}} \hat{p}_{\text{visual}} \right)$$

Where:
- $\hat{p}_{\text{text}}$ is the output of the EATD DistilBERT + BiLSTM cross-attention model.
- $\hat{p}_{\text{audio}}$ is the output of the regularized logistic regression model trained on openSMILE eGeMAPSv02 descriptors.
- $\hat{p}_{\text{visual}}$ is the visual kinematic descriptor score (or auxiliary FACS blendshape affect heuristic).
- Weights $w$ are estimated via logistic meta-learning on 5-fold cross-validation.

### Category Thresholds
- **Elevated Screening Indicator:** $S_{\text{composite}} \ge 0.50$
- **Lower-Risk Screening Indicator:** $S_{\text{composite}} < 0.50$
- **Safety Override Trigger:** Automatically forces immediate elevated status and displays national crisis helpline resources regardless of numerical score if suicide ideation or crisis keywords are detected.

---

## 2. Calibrated Probability & Prediction Uncertainty

Uncalibrated neural network logits frequently exhibit overconfidence. NeuroScreen uses empirical probability calibration verified against held-out validation partitions:

| Metric | AVDD Multimodal Late Fusion (Clinical HDS) | AVDD Non-Clinical SDS |
| :--- | :---: | :---: |
| **Brier Score** | **0.1412** | **0.1245** |
| **Calibration Slope** | **0.984** | **0.871** |
| **Uncertainty Interval** | $\pm 0.15$ | $\pm 0.28$ |

### Brier Score Definition
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (f_i - o_i)^2$$
Where $f_i$ is the calibrated probability and $o_i \in \{0, 1\}$ is the clinical ground-truth label. A lower score signifies tighter alignment between predicted probabilities and actual positive rates.

---

## 3. Modality Breakdown Specification

Each modality is displayed with explicit availability flags, model identity, raw score, and descriptive summary:

### 3.1 Text Modality (Interview Transcripts)
- **Model:** DistilBERT + BiLSTM + Attention (EATD-Corpus)
- **Benchmark AUC:** 0.6756 | Balanced Accuracy: 59.78%
- **Evidence Output:** LIME token attributions highlight specific phrases contributing towards the score (e.g. "exhausted", "sleeping badly", "hard to focus") or towards protective factors (e.g. "walked outside").

### 3.2 Voice Modality (Acoustic Prosody & Pitch)
- **Model:** openSMILE eGeMAPSv02 / Wav2Vec2 + Logistic Regression
- **Benchmark AUC:** 0.8370 (AVDD Inpatient HDS) / 0.6384 (EATD Wav2Vec2)
- **Evidence Output:** Fundamental frequency ($F_0$) pitch mean, pitch standard deviation, pause frequency, and silence ratio.

### 3.3 Visual Affect Modality (Kinematics & Facial Expression)
- **Model:** OpenFace 2.0 AU Kinematics (AVDD) & MediaPipe FACS Blendshape / FER-2013 ResNet-18
- **Benchmark AUC:** 0.8434 (AVDD Kinematics HDS) / 0.5358 accuracy (FER-2013 Affect)
- **Scientific Limitation:** Facial expression is an **auxiliary affect classifier**, strictly separated from depression detection. A flat expression or grimace provides descriptive behavioral context only.

### 3.4 Multimodal Late-Fusion Meta-Learner
- **Model:** L2-regularized logistic stacking meta-learner
- **Benchmark AUC:** **0.8659** (AVDD Inpatient HDS) | Balanced Accuracy: **79.44%** | Macro-F1: **79.81%**

---

## 4. Responsible Clinical Disclaimers

Every screening result presentation must feature the immutable disclaimer:

> **"Experimental screening result — requires professional interpretation."**  
> *NeuroScreen is an investigational clinical research prototype. Outputs must not be used as an autonomous psychiatric diagnosis. Professional clinical evaluation by a certified clinician is mandatory.*
