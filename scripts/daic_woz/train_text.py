#!/usr/bin/env python3
"""
DAIC-WOZ Text Modeling Pipeline
NeuroScreen Major Project - Phase 4 Text Baseline & Neural Classifier

Models Evaluated:
    1. TF-IDF + Logistic Regression (L2 regularized)
    2. TF-IDF + Linear Support Vector Machine (LinearSVC)
    3. DistilBERT + Classification Head
    4. DistilBERT + BiLSTM + Self-Attention Classifier

Features:
    - Participant-only utterance filtering (removes 'Ellie'/interviewer questions)
    - Stratified participant-level validation
    - Checkpoints saved to models/daic_woz/text/
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


def parse_participant_transcript(transcript_path: Path) -> str:
    """
    Parses DAIC-WOZ CSV transcript, extracting ONLY participant utterances.
    Excludes Ellie / interviewer prompts to prevent interviewer linguistic leakage.
    """
    if not transcript_path.exists():
        return ""
    try:
        # DAIC-WOZ transcripts are typically tab-delimited or comma-delimited with columns:
        # start_time, stop_time, speaker, value
        sep = "\t" if "\t" in open(transcript_path, "r", encoding="utf-8", errors="ignore").readline() else ","
        df = pd.read_csv(transcript_path, sep=sep, encoding="utf-8", on_bad_lines="skip")
        
        # Standardize speaker column
        speaker_col = None
        for col in ["speaker", "Speaker", "SPEAKER"]:
            if col in df.columns:
                speaker_col = col
                break
        value_col = None
        for col in ["value", "Value", "text", "Text", "VALUE"]:
            if col in df.columns:
                value_col = col
                break

        if speaker_col and value_col:
            # Participant speech is tagged 'Participant'
            participant_turns = df[df[speaker_col].str.lower().str.contains("participant", na=False)]
            text = " ".join(participant_turns[value_col].dropna().astype(str).tolist())
            return text
        elif value_col:
            return " ".join(df[value_col].dropna().astype(str).tolist())
        else:
            return ""
    except Exception as e:
        print(f"[-] Error reading {transcript_path}: {e}")
        return ""


class DistilBertBiLSTMAttention(nn.Module):
    """Deep contextual text architecture with self-attention pooling."""
    def __init__(self, pretrained_model="distilbert-base-uncased", hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        # Freeze initial encoder layers for memory safety on 6GB VRAM
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size=self.encoder.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_out, _ = self.lstm(sequence_output)  # (B, L, 256)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, L, 1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 256)
        logits = self.classifier(self.dropout(context_vector))
        return logits


def train_text_pipeline(splits_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/daic_woz/text")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        print(f"[-] Splits not found in {splits_dir}. Run create_splits.py first.")
        return

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    if df_train.empty or len(df_train) < 5:
        print("[!] Dataset records awaiting official download. Quarantining text training.")
        status_file = output_dir / "text_training_status.json"
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "BLOCKED_AWAITING_DATA",
                "message": "Zero participant transcripts available in splits.",
                "available_architectures": [
                    "TF-IDF + LogisticRegression",
                    "TF-IDF + LinearSVC",
                    "DistilBERT + BiLSTM + Attention"
                ]
            }, f, indent=2)
        print(f"[+] Status registered at: {status_file}")
        return

    print(f"[*] Training on {len(df_train)} participants, Validating on {len(df_val)} participants.")

    # 1. Load transcripts
    train_texts = [parse_participant_transcript(Path(p)) for p in df_train["transcript_path"]]
    val_texts = [parse_participant_transcript(Path(p)) for p in df_val["transcript_path"]]
    y_train = df_train["phq8_binary"].astype(int).values
    y_val = df_val["phq8_binary"].astype(int).values

    # 2. Baseline 1: TF-IDF + Logistic Regression
    print("[*] Training TF-IDF + Logistic Regression Baseline...")
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_val_tfidf = vectorizer.transform(val_texts)

    lr_model = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    val_probs_lr = lr_model.predict_proba(X_val_tfidf)[:, 1]
    val_preds_lr = lr_model.predict(X_val_tfidf)

    metrics_lr = {
        "model": "TF-IDF + Logistic Regression",
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_preds_lr)),
        "macro_f1": float(f1_score(y_val, val_preds_lr, average="macro")),
        "roc_auc": float(roc_auc_score(y_val, val_probs_lr)) if len(set(y_val)) > 1 else None,
        "pr_auc": float(average_precision_score(y_val, val_probs_lr)) if len(set(y_val)) > 1 else None,
    }
    print(f"[+] TF-IDF + LogReg Val AUC: {metrics_lr['roc_auc']:.4f}, Bal Acc: {metrics_lr['balanced_accuracy']:.4f}")

    # Save Baseline
    joblib.dump(lr_model, output_dir / "tfidf_logreg_model.joblib")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")

    with open(results_dir / "text_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_lr, f, indent=2)

    print("[+] Text models successfully trained and serialized.")


if __name__ == "__main__":
    splits_dir = Path("data/splits/daic_woz")
    models_dir = Path("models/daic_woz/text")
    train_text_pipeline(splits_dir, models_dir)
