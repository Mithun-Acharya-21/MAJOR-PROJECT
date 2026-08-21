"""Evaluate a trained text model checkpoint and produce a full metric report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training.text.deberta_model import DebertaBiLSTMAttentionClassifier, DebertaTextConfig
from training.text.train_deberta import TextRecords, collate

DISCLAIMER = (
    "These are model-derived signals on a research dataset, "
    "not evidence of a mental-health condition."
)


def load_checkpoint(checkpoint_path: Path) -> tuple[DebertaBiLSTMAttentionClassifier, dict[str, int], str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = DebertaTextConfig(**checkpoint["config"])
    model = DebertaBiLSTMAttentionClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["label_to_id"], config.model_name


@torch.no_grad()
def collect_predictions(
    model: DebertaBiLSTMAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    actual, predicted, probs = [], [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        logits = outputs["logits"]
        soft = torch.softmax(logits, dim=1)[:, 1]
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        probs.extend(soft.cpu().tolist())
    return actual, predicted, probs


def compute_metrics(actual: list[int], predicted: list[int], probs: list[float]) -> dict:
    actual_arr = np.array(actual)
    probs_arr = np.array(probs)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual_arr, probs_arr)) if len(set(actual)) > 1 else None,
        "pr_auc": float(average_precision_score(actual_arr, probs_arr)) if len(set(actual)) > 1 else None,
        "brier_score": float(brier_score_loss(actual_arr, probs_arr)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(
    checkpoint_path: Path,
    eval_csv: Path,
    output_dir: Path,
    batch_size: int = 8,
) -> dict:
    # Refuse to accidentally use the final test set before the model is locked
    if "test" in eval_csv.name.casefold() and "validation" not in eval_csv.name.casefold():
        raise ValueError(
            "Final test evaluation is only permitted after the model is locked. "
            "Use the validation split for model development."
        )
    model, label_to_id, model_name = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    split_name = "validation"
    dataset = TextRecords(eval_csv, split_name, label_to_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate(tokenizer, model.config.max_length),
    )
    actual, predicted, probs = collect_predictions(model, loader, device)
    metrics = compute_metrics(actual, predicted, probs)
    report = classification_report(actual, predicted, target_names=sorted(label_to_id, key=label_to_id.get))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "disclaimer": DISCLAIMER,
        "checkpoint": str(checkpoint_path),
        "eval_file": str(eval_csv),
        "n_samples": len(actual),
        "metrics": metrics,
        "classification_report": report,
    }
    (output_dir / "eval-report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained text model checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.eval_csv, args.output_dir, args.batch_size)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
