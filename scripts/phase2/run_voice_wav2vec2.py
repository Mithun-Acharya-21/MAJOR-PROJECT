"""Run Experiment V2 (frozen Wav2Vec2 embeddings + trained head).

Usage:
    .venv/Scripts/python scripts/phase2/run_voice_wav2vec2.py [--encoder chinese|english] [--no-cache]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.voice_wav2vec2 import (  # noqa: E402
    ENCODERS,
    run_voice_wav2vec2,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="chinese")
    ap.add_argument("--chunk-sec", type=float, default=8.0)
    ap.add_argument("--max-chunks", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    meta = run_voice_wav2vec2(
        seed=args.seed,
        encoder_name=ENCODERS[args.encoder],
        chunk_sec=args.chunk_sec,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        n_boot=args.n_boot,
        write=not args.no_write,
        reuse_cache=not args.no_cache,
    )
    print(json.dumps({
        "model_version": meta["model_version"],
        "embedding": {k: v for k, v in meta["training"]["embedding"].items()
                      if k != "excluded_sessions"},
        "head": meta["config"]["head_training"],
        "test_participant": meta["test_metrics"]["participant_level"],
        "test_participant_roc_auc_ci": meta["test_metrics"]["participant_bootstrap_ci"]["ci"].get("roc_auc"),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
