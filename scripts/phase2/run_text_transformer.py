"""Run Experiment T2 (transformer text model) on the verified EATD splits.

T2a (default): frozen multilingual encoder embeddings + trained head.
T2b (--finetune): fine-tune the top encoder layers (exploratory).

Usage:
    .venv/Scripts/python scripts/phase2/run_text_transformer.py [--finetune]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.text_transformer import (  # noqa: E402
    DEFAULT_ENCODER,
    run_text_finetune,
    run_text_transformer,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--finetune", action="store_true", help="run the optional fine-tuning variant")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if args.finetune:
        meta = run_text_finetune(seed=args.seed, n_boot=args.n_boot, write=not args.no_write)
    else:
        meta = run_text_transformer(
            seed=args.seed,
            encoder_name=args.encoder,
            max_seq_len=args.max_seq_len,
            head_epochs=args.head_epochs,
            head_lr=args.head_lr,
            n_boot=args.n_boot,
            write=not args.no_write,
        )
    summary = {
        "model_version": meta["model_version"],
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
