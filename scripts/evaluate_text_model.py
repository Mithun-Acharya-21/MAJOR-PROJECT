"""
Evaluate the trained text model on the LOCKED test split (Phase 6).

Run once, after training and calibration fitting are complete. This is the
only place the EATD test split is scored. Metrics are computed from the
model's actual probabilities and written to results/metrics/text_metrics.json.

Usage:
    python scripts/evaluate_text_model.py [--calibrated]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.text_classifier import load_text_classifier
from ml.evaluation.metrics import compute_full_metrics

SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "eatd_splits" / "eatd_subject_splits.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
CONFUSION_DIR = PROJECT_ROOT / "results" / "confusion_matrices"


def load_test_rows() -> list[dict]:
    import csv

    rows = []
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and (r.get("text") or "").strip():
                rows.append({"text": r["text"], "label": int(r["label"]),
                             "subject_uid": r["subject_uid"]})
    if not rows:
        raise SystemExit("FATAL: test split is empty")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrated", action="store_true",
                    help="Also apply the fitted calibration model to test probabilities")
    args = ap.parse_args()

    clf = load_text_classifier()
    if clf is None or not clf.available:
        print("FATAL: no trained text model at models/final/text_model — run scripts/train_text_model.py first")
        return 1

    rows = load_test_rows()
    print(f"Evaluating on LOCKED test split: {len(rows)} samples, "
        f"{sum(r['label'] for r in rows)} depressed, {len(rows) - sum(r['label'] for r in rows)} control")

    y_true = [r["label"] for r in rows]
    y_prob_raw = [clf.predict(r["text"])["probability"] for r in rows]

    # --- Raw (uncalibrated) metrics ---
    m = compute_full_metrics(y_true, y_prob_raw)
    pred = [1 if p >= 0.5 else 0 for p in y_prob_raw]
    tp = sum(1 for t, p in zip(y_true, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, pred) if t == 0 and p == 0)

    meta = clf.metadata
    result = {
        "model": meta.get("model_name", "TextBranch + classification head"),
        "model_version": meta.get("model_version", "unknown"),
        "dataset": "EATD (Chinese interview texts)",
        "dataset_version": meta.get("dataset_version", "unknown"),
        "split": "test",
        "samples": len(rows),
        "subjects": len({r["subject_uid"] for r in rows}),
        "accuracy": round(float(m["accuracy"]), 4),
        "precision": round(float(m["precision"]), 4),
        "recall": round(float(m["recall"]), 4),
        "f1": round(float(m["f1"]), 4),
        "sensitivity": round(float(m["sensitivity"]), 4),
        "specificity": round(float(m["specificity"]), 4),
        "roc_auc": round(float(m["auc"]), 4),
        "pr_auc": round(float(m["pr_auc"]), 4),
        "balanced_accuracy": round(float(m["balanced_accuracy"]), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "probability_source": "sigmoid(logit) of trained classification head",
        "calibrated": False,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_criterion": meta.get("training", {}).get("selection_criterion"),
        "limitations": meta.get("limitations", []),
    }

    # --- Optional calibrated metrics (calibration fitted on VALIDATION only) ---
    calibrated_block = None
    if args.calibrated:
        from backend.app.services.calibration_service import load_calibration_model
        cal = load_calibration_model()
        if cal is not None:
            y_prob_cal = [cal.transform(p) for p in y_prob_raw]
            mc = compute_full_metrics(y_true, y_prob_cal)
            calibrated_block = {
                "method": cal.method,
                "accuracy": round(float(mc["accuracy"]), 4),
                "precision": round(float(mc["precision"]), 4),
                "recall": round(float(mc["recall"]), 4),
                "f1": round(float(mc["f1"]), 4),
                "roc_auc": round(float(mc["auc"]), 4),
                "balanced_accuracy": round(float(mc["balanced_accuracy"]), 4),
                "note": "Calibration fitted on VALIDATION split predictions only.",
            }
            result["calibrated"] = True
        else:
            print("WARNING: --calibrated requested but no calibration artifact exists")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "text_metrics.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    cm_out = CONFUSION_DIR / "text_model_test_confusion.json"
    cm_out.write_text(json.dumps({
        "model_version": result["model_version"],
        "dataset": result["dataset"],
        "split": "test",
        "labels": ["control(0)", "depressed(1)"],
        "matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "evaluated_at": result["evaluated_at"],
    }, indent=2), encoding="utf-8")

    # Test predictions retained for post-hoc calibration comparison (Phase 15)
    preds_out = PROJECT_ROOT / "models" / "final" / "text_model" / "test_predictions.json"
    preds_out.write_text(json.dumps({
        "y_true": y_true,
        "y_prob_raw": [round(p, 6) for p in y_prob_raw],
        "subject_uid": [r["subject_uid"] for r in rows],
        "model_version": result["model_version"],
        "created_at": result["evaluated_at"],
        "note": "Test predictions saved ONLY for evaluating fitted calibration; never used for fitting or selection.",
    }, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k != "limitations"}, indent=2))
    print(f"\nSaved: {out}")
    print(f"Saved: {cm_out}")
    print(f"Saved: {preds_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
