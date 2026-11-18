# DAIC-WOZ Multimodal Depression Research Pipeline

**NeuroScreen Major Project**  
**Lead Architecture:** Multimodal Depression Screening Framework  
**Governing Dataset:** Distress Analysis Interview Corpus - Wizard-of-Oz (USC ICT)  

---

## 1. Overview

This module provides a rigorous, participant-level multimodal machine learning pipeline for experimental depression screening research using the official DAIC-WOZ corpus. 

The pipeline spans:
- **Linguistic Analysis (Text):** Participant-only transcript processing using TF-IDF + regularized classification and DistilBERT + BiLSTM + Attention.
- **Acoustic Analysis (Audio):** Participant vocal recordings parameterized into 88-dimensional openSMILE eGeMAPSv02 functionals and Wav2Vec2 representations.
- **Facial Kinematics (Visual):** OpenFace / CLNF summary dynamics of facial Action Units (AU01, AU04, AU12, AU15), head pose velocity, and eye gaze.
- **Multimodal Late Fusion:** Out-of-fold stacked logistic meta-learner combining cross-modal screening indicators with Platt probability calibration.

---

## 2. Directory Structure

```text
data/
├── raw/daic_woz/          # Official USC ICT archives (<id>_P/)
├── manifests/daic_woz/    # Manifests, checksums, and PHQ-8 audits
└── splits/daic_woz/       # Train, Validation, and Locked Test CSVs

models/daic_woz/
├── text/                  # TF-IDF & DistilBERT checkpoints
├── audio/                 # openSMILE eGeMAPS acoustic models
├── visual/                # OpenFace kinematics models
└── fusion/                # Multimodal late-fusion meta-learner

results/daic_woz/
├── reports/               # Quality, evaluation, and optimization reports
└── explainability/        # LIME and SHAP attributions

scripts/daic_woz/          # CLI pipelines for all phases
backend/app/daic_woz/      # FastAPI routes, schemas, loader, and services
frontend/app/daic-woz/     # Next.js research screening interface
docs/daic_woz/             # Full academic documentation package
```

---

## 3. Academic Access & Ethics

Access to DAIC-WOZ is governed by USC Institute for Creative Technologies under a controlled research agreement.
- Portal: `https://dcapswoz.ict.usc.edu/`
- Access status: [Phase 4 access status](../phase_4_access_status.md)
- Non-redistribution and participant confidentiality are strictly enforced.
