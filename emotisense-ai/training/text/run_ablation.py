"""Run controlled text-architecture ablations on the same frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.text.deberta_model import DebertaTextConfig
from training.text.train_deberta import VARIANTS, train


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DeBERTa sequence heads on one frozen validation set.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    results = []
    for variant in VARIANTS:
        result = train(
            args.train,
            args.validation,
            args.output_dir / variant,
            DebertaTextConfig(epochs=args.epochs, batch_size=args.batch_size),
            variant=variant,
        )
        results.append(result)
    results.sort(key=lambda item: item["best_validation_f1"], reverse=True)
    (args.output_dir / "ablation-summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

