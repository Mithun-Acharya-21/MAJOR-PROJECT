"""Build dedicated persisted participant-level OOF predictions for Phase 5E.

Reads the existing session-level OOF CSVs (E-T1_oof.csv, E-T2_oof.csv, E-V3_oof.csv)
and computes participant-level aggregated probabilities (mean aggregation per Section 4.4)
across all 25 folds (5 repeats x 5 folds), producing:
  - results/phase5/eatd_fusion/oof_predictions/participant_level_oof_predictions.csv
  - results/phase5/eatd_fusion/oof_predictions/participant_level_oof_predictions.json
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOF_DIR = PROJECT_ROOT / "results" / "phase5" / "eatd_fusion" / "oof_predictions"

MODELS = ["E-T1", "E-T2", "E-V3"]

def main():
    participant_records = []
    
    for model_name in MODELS:
        csv_path = OOF_DIR / f"{model_name}_oof.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist.")
            continue
            
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        # Group by (repeat_seed, fold, participant_key)
        grouped = defaultdict(list)
        for r in rows:
            key = (int(r["repeat_seed"]), int(r["fold"]), r["participant_key"])
            grouped[key].append(r)
            
        for (repeat_seed, fold, pk), s_rows in grouped.items():
            probs = [float(r["probability"]) for r in s_rows]
            label = int(s_rows[0]["participant_label"])
            thr = float(s_rows[0]["threshold"])
            mean_prob = float(np.mean(probs))
            pred = int(mean_prob >= thr)
            
            participant_records.append({
                "model": model_name,
                "repeat_seed": repeat_seed,
                "fold": fold,
                "participant_key": pk,
                "participant_label": label,
                "n_sessions": len(s_rows),
                "probability_mean": round(mean_prob, 6),
                "threshold": round(thr, 6),
                "prediction": pred,
                "is_correct": int(pred == label)
            })
            
    print(f"Total participant-level records created: {len(participant_records)}")
    
    # Save CSV
    out_csv = OOF_DIR / "participant_level_oof_predictions.csv"
    fieldnames = [
        "model", "repeat_seed", "fold", "participant_key", "participant_label",
        "n_sessions", "probability_mean", "threshold", "prediction", "is_correct"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(participant_records)
    print(f"Saved {out_csv}")
    
    # Save JSON
    out_json = OOF_DIR / "participant_level_oof_predictions.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(participant_records, f, indent=2)
    print(f"Saved {out_json}")

if __name__ == "__main__":
    main()
