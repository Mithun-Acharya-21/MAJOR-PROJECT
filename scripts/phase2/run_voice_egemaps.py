"""Run Experiment V1 (openSMILE eGeMAPS + Logistic Regression).

Usage:
    .venv/Scripts/python scripts/phase2/run_voice_egemaps.py [--no-cache]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.voice_egemaps import run_voice_egemaps  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-cache", action="store_true", help="re-extract eGeMAPS features")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    meta = run_voice_egemaps(
        seed=args.seed,
        n_boot=args.n_boot,
        write=not args.no_write,
        reuse_cache=not args.no_cache,
    )
    print(json.dumps({
        "model_version": meta["model_version"],
        "extraction": meta["training"]["extraction"],
        "selected": meta["config"]["selected"],
        "test_participant": meta["test_metrics"]["participant_level"],
        "test_participant_roc_auc_ci": meta["test_metrics"]["participant_bootstrap_ci"]["ci"].get("roc_auc"),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
