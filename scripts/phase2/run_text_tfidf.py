"""Run Experiment T1 (TF-IDF + Logistic Regression) on the verified EATD splits.

Usage:
    .venv/Scripts/python scripts/phase2/run_text_tfidf.py [--seed 42] [--no-write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.text_tfidf import run_text_tfidf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-write", action="store_true", help="evaluate without saving artifacts")
    args = ap.parse_args()

    meta = run_text_tfidf(seed=args.seed, n_boot=args.n_boot, write=not args.no_write)
    summary = {
        "model_version": meta["model_version"],
        "selected": meta["config"]["selected"],
        "validation_participant": {
            k: meta["validation_metrics"]["participant_level"][k]
            for k in ("accuracy", "f1", "roc_auc")
        },
        "test_participant": meta["test_metrics"]["participant_level"],
        "test_participant_roc_auc_ci": meta["test_metrics"]["participant_bootstrap_ci"]["ci"].get("roc_auc"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
