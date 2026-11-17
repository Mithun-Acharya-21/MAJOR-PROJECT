"""
Fit a probability calibrator for the screening pipeline (Phase 14).

Default input is the VALIDATION-split prediction file produced by
scripts/train_text_model.py (models/final/text_model/validation_predictions.json).
Calibration is fitted on VALIDATION data only — the test split is never
touched here.

Outputs:
    models/calibration.json              — calibration artifact
    results/metrics/calibration_metrics.json — raw vs calibrated comparison
        (Brier, ECE, reliability curve on the validation split)

Artifact fields: method, parameters, validation metrics, training timestamp,
model version, dataset version.

Usage:
    python scripts/fit_calibration.py                     # use validation_predictions.json
    python scripts/fit_calibration.py --predictions FILE  # custom validation predictions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.calibration_service import (  # noqa: E402
    evaluate_calibration,
    save_calibration_model,
    select_best_calibration,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREDICTIONS = PROJECT_ROOT / "models" / "final" / "text_model" / "validation_predictions.json"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "calibration.json"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "calibration_metrics.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a validated probability calibrator")
    parser.add_argument("--predictions", type=str, default=str(DEFAULT_PREDICTIONS),
                        help="JSON file with {y_true, y_prob} from the VALIDATION split")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}")
        print("Run scripts/train_text_model.py first (it writes validation_predictions.json).")
        return 1

    data = json.loads(pred_path.read_text(encoding="utf-8"))
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    if len(y_true) != len(y_prob) or not y_true:
        print("ERROR: y_true/y_prob must be non-empty and equal length")
        return 1
    if len(set(y_true)) < 2:
        print("ERROR: validation labels contain a single class — cannot fit calibration")
        return 1

    model_version = data.get("model_version", "unknown")
    dataset_version = data.get("dataset_version", "unknown")
    print(f"Source           : {pred_path}")
    print(f"Samples          : {len(y_true)} (validation split — test never used)")
    print(f"Model version    : {model_version}")
    print(f"Dataset version  : {dataset_version}")

    baseline = evaluate_calibration(y_true, y_prob)
    print(f"Baseline (raw)   : Brier={baseline['brier_score']:.4f}  ECE={baseline['ece']:.4f}")

    model, report = select_best_calibration(y_true, y_prob)
    print(f"Selected method  : {model.method}")
    for name, metrics in report.get("validation", {}).items():
        print(f"  {name:<10} Brier={metrics['brier_score']:.4f}  ECE={metrics['ece']:.4f}")

    # --- Artifact (Phase 14 required fields) ---
    artifact = model.to_dict()
    artifact.update({
        "model_version": model_version,
        "dataset_version": dataset_version,
        "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fit_input": str(pred_path.relative_to(PROJECT_ROOT)),
        "n_samples": len(y_true),
        "fit_split": "validation",
        "note": (
            "Fitted on validation-split predictions only. The test split was "
            "never used for fitting or method selection."
        ),
    })
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Saved artifact   : {ARTIFACT_PATH}")

    # --- Calibration metrics (Phase 15): raw vs calibrated on validation ---
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    metrics_doc = {
        **{k: v for k, v in existing.items() if k != "validation_comparison"},
        "model_version": model_version,
        "dataset_version": dataset_version,
        "calibration_version": f"{model.method}-fit-{artifact['fitted_at']}",
        "fit_split": "validation",
        "n_samples": len(y_true),
        "created_at": artifact["fitted_at"],
        "validation_comparison": {
            "raw": baseline,
            "calibrated": report["validation"].get(model.method, baseline),
            "selected_method": model.method,
            "all_methods": report["validation"],
        },
        "test_comparison": existing.get("test_comparison"),
        "test_comparison_note": (
            "Filled by scripts/evaluate_text_model.py --calibrated after test evaluation."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics_doc, indent=2), encoding="utf-8")
    print(f"Saved metrics    : {METRICS_PATH}")

    if model.method == "identity":
        print("\nNo validated improvement over uncalibrated probabilities.")
        print("The identity calibrator is saved explicitly; inference uses raw")
        print("probabilities and reports calibration_method='identity'.")
    else:
        print(f"\nCalibration improved BOTH Brier and ECE on the validation split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
