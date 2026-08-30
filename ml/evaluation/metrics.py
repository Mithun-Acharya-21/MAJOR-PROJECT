"""
Metrics — Evaluation metrics for depression detection.

Computes standard classification metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC
  - Confusion matrix

Exports:
  compute_metrics — compute all metrics from labels and probabilities
  compute_confusion_matrix — get TP, FP, FN, TN counts
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def compute_confusion_matrix(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    """Compute confusion matrix components.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Dict with 'tp', 'fp', 'fn', 'tn' counts.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC via staircase estimator (handles ties, no sklearn dependency)."""
    if np.isnan(y_score).any():
        mask = ~np.isnan(y_score)
        y_true = y_true[mask]
        y_score = y_score[mask]

    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-y_score, kind="mergesort")
    ys = y_true[order]
    scores = y_score[order]

    area = 0.0
    tpr = 0.0
    i = 0
    n = len(ys)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        p = int(np.sum(ys[i:j]))
        neg = (j - i) - p
        area += (tpr + (p / n_pos) / 2.0) * (neg / n_neg)
        tpr += p / n_pos
        i = j

    return float(area)


def compute_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all classification metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for class 1 (depressed).
        threshold: Classification threshold.

    Returns:
        Dict with 'accuracy', 'precision', 'recall', 'f1', 'auc'.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = compute_confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    auc = _roc_auc(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }
