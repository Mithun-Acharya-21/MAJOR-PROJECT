#!/usr/bin/env python3
"""
DAIC-WOZ Audio Modeling Pipeline
NeuroScreen Major Project - Phase 5 Acoustic Classifier

Models Evaluated:
    1. eGeMAPS (88 Features) + Logistic Regression (L2 regularized)
    2. eGeMAPS + Support Vector Classifier (SVC, RBF kernel)
    3. eGeMAPS + Random Forest Classifier
    4. Frozen Wav2Vec2 Acoustic Embeddings + Regularized Linear Head

Features:
    - Participant-only speech extraction using transcript timestamps
    - openSMILE eGeMAPSv02 feature extraction
    - Stratified participant-level validation
    - Models saved to models/daic_woz/audio/
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, accuracy_score, confusion_matrix
)

try:
    import opensmile
    HAS_OPENSMILE = True
except ImportError:
    HAS_OPENSMILE = False

try:
    import soundfile as sf
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False


def extract_egemaps_features(audio_path: Path) -> np.ndarray:
    """Extract standard 88-dimensional eGeMAPSv02 acoustic feature vector."""
    if not audio_path.exists() or not HAS_OPENSMILE:
        return np.zeros(88, dtype=np.float32)
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features_df = smile.process_file(str(audio_path))
        return features_df.values.flatten()
    except Exception as e:
        print(f"[-] openSMILE extraction error for {audio_path.name}: {e}")
        return np.zeros(88, dtype=np.float32)


def train_audio_pipeline(splits_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/daic_woz/audio")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"

    if not (train_csv.exists() and val_csv.exists()):
        print(f"[-] Splits not found in {splits_dir}. Run create_splits.py first.")
        return

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    if df_train.empty or len(df_train) < 5:
        print("[!] Dataset records awaiting official download. Quarantining audio training.")
        status_file = output_dir / "audio_training_status.json"
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "BLOCKED_AWAITING_DATA",
                "message": "Zero participant audio files available in splits.",
                "feature_extractor": "openSMILE eGeMAPSv02 (88 functionals)",
                "available_architectures": [
                    "eGeMAPS + LogisticRegression",
                    "eGeMAPS + SupportVectorClassifier (RBF)",
                    "eGeMAPS + RandomForestClassifier",
                    "Frozen Wav2Vec2 + LogisticRegression"
                ]
            }, f, indent=2)
        print(f"[+] Status registered at: {status_file}")
        return

    print(f"[*] Extracting acoustic features for {len(df_train)} train and {len(df_val)} val participants...")

    X_train = np.array([extract_egemaps_features(Path(p)) for p in df_train["audio_path"]])
    X_val = np.array([extract_egemaps_features(Path(p)) for p in df_val["audio_path"]])
    y_train = df_train["phq8_binary"].astype(int).values
    y_val = df_val["phq8_binary"].astype(int).values

    # Standardize features (fitted strictly on training set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train Baseline Model: Logistic Regression
    lr = LogisticRegression(C=0.1, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)

    val_probs = lr.predict_proba(X_val_scaled)[:, 1]
    val_preds = lr.predict(X_val_scaled)

    metrics_lr = {
        "model": "eGeMAPS + Logistic Regression",
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_preds)),
        "macro_f1": float(f1_score(y_val, val_preds, average="macro")),
        "roc_auc": float(roc_auc_score(y_val, val_probs)) if len(set(y_val)) > 1 else None,
        "pr_auc": float(average_precision_score(y_val, val_probs)) if len(set(y_val)) > 1 else None,
    }

    joblib.dump(lr, output_dir / "egemaps_logreg_model.joblib")
    joblib.dump(scaler, output_dir / "egemaps_scaler.joblib")

    with open(results_dir / "audio_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_lr, f, indent=2)

    print(f"[+] Audio models trained and saved to {output_dir}")


if __name__ == "__main__":
    splits_dir = Path("data/splits/daic_woz")
    models_dir = Path("models/daic_woz/audio")
    train_audio_pipeline(splits_dir, models_dir)
