"""Leakage-safe classical text baseline trainer.

This establishes a transparent reference before transformer fine-tuning. It
requires separately prepared training and validation CSVs, and intentionally
has no final-test argument.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from training.text.preprocess import normalize_text

REQUIRED_COLUMNS = {"record_id", "participant_id", "label", "text"}


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    sensitivity: float
    specificity: float
    support: int


def read_text_split(path: Path, expected_split: str) -> tuple[list[str], list[str], list[str]]:
    """Read one prepared split and reject ambiguous or test data."""

    if "test" in path.name.casefold():
        raise ValueError("the classical baseline never reads a final test split")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    texts: list[str] = []
    labels: list[str] = []
    participants: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        if row.get("split") and row["split"].strip() != expected_split:
            raise ValueError(f"line {row_number}: expected split {expected_split!r}")
        try:
            texts.append(normalize_text(row["text"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"line {row_number}: invalid text ({error})") from error
        labels.append(row["label"].strip())
        participants.append(row["participant_id"].strip())

    if len(set(labels)) != 2:
        raise ValueError("baseline currently requires exactly two label classes")
    if not all(participants):
        raise ValueError("participant_id is required")
    return texts, labels, participants


def assert_no_participant_overlap(train_participants: list[str], validation_participants: list[str]) -> None:
    overlap = set(train_participants) & set(validation_participants)
    if overlap:
        raise ValueError(f"participant leakage between train and validation: {sorted(overlap)[:10]}")


def build_pipeline() -> Pipeline:
    """Create a transparent, class-balanced reference model."""

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42)),
        ]
    )


def evaluate_binary(y_true: list[str], positive_probabilities: list[float], positive_label: str) -> BaselineMetrics:
    y_binary = [int(label == positive_label) for label in y_true]
    predictions = [int(score >= 0.5) for score in positive_probabilities]
    tn, fp, fn, tp = confusion_matrix(y_binary, predictions, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_binary, positive_probabilities) if len(set(y_binary)) == 2 else None
    pr_auc = average_precision_score(y_binary, positive_probabilities) if len(set(y_binary)) == 2 else None
    return BaselineMetrics(
        accuracy=accuracy_score(y_binary, predictions),
        precision=precision_score(y_binary, predictions, zero_division=0),
        recall=recall_score(y_binary, predictions, zero_division=0),
        f1=f1_score(y_binary, predictions, zero_division=0),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        sensitivity=tp / (tp + fn) if tp + fn else 0.0,
        specificity=tn / (tn + fp) if tn + fp else 0.0,
        support=len(y_true),
    )


def train_and_evaluate(train_csv: Path, validation_csv: Path, output_dir: Path) -> BaselineMetrics:
    train_texts, train_labels, train_participants = read_text_split(train_csv, "train")
    validation_texts, validation_labels, validation_participants = read_text_split(validation_csv, "validation")
    assert_no_participant_overlap(train_participants, validation_participants)

    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)
    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_label = sorted(classes)[-1]
    positive_index = classes.index(positive_label)
    probabilities = pipeline.predict_proba(validation_texts)[:, positive_index].tolist()
    metrics = evaluate_binary(validation_labels, probabilities, positive_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tfidf-logreg-baseline.pkl").open("wb") as stream:
        pickle.dump(pipeline, stream)
    (output_dir / "validation-metrics.json").write_text(
        json.dumps({"positive_label": positive_label, **asdict(metrics)}, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a text baseline using train and validation data only.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    metrics = train_and_evaluate(args.train, args.validation, args.output_dir)
    print(json.dumps(asdict(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

