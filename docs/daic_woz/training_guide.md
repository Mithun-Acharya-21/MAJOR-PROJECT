# DAIC-WOZ Model Training & Execution Guide

---

## 1. Prerequisites & Execution Sequence

Execute all training stages sequentially from the project root using `.venv`:

```bash
# Step 1: Verify downloaded raw files
.\.venv\Scripts\python.exe scripts\daic_woz\verify_download.py

# Step 2: Build manifests and inspect dataset quality
.\.venv\Scripts\python.exe scripts\daic_woz\inspect_dataset.py

# Step 3: Generate participant-stratified leak-free splits
.\.venv\Scripts\python.exe scripts\daic_woz\create_splits.py

# Step 4: Train text baseline and contextual models
.\.venv\Scripts\python.exe scripts\daic_woz\train_text.py

# Step 5: Train openSMILE eGeMAPS acoustic models
.\.venv\Scripts\python.exe scripts\daic_woz\train_audio.py

# Step 6: Train OpenFace visual kinematics models
.\.venv\Scripts\python.exe scripts\daic_woz\train_visual.py

# Step 7: Train Multimodal Late Fusion Meta-Learner
.\.venv\Scripts\python.exe scripts\daic_woz\train_fusion.py

# Step 8: Run final locked test set evaluation
.\.venv\Scripts\python.exe scripts\daic_woz\evaluate_models.py

# Step 9: Generate LIME and SHAP explainability artifacts
.\.venv\Scripts\python.exe scripts\daic_woz\explain_models.py
```

---

## 2. Resource & VRAM Management

- Batch Size: For contextual neural models (DistilBERT), keep batch size $B=8$ or $B=16$.
- Feature Caching: Tabular features (eGeMAPS and OpenFace) are saved to disk to maintain RAM usage below 2 GB.
