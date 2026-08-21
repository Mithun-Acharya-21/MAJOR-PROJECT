"""Tests for the text evaluation module (Phase 6)."""

from __future__ import annotations

import pytest
from pathlib import Path


def test_evaluate_refuses_test_csv(tmp_path: Path) -> None:
    """Evaluation must refuse a file named 'test.csv' to protect the held-out set."""
    from training.text.evaluate_text import evaluate

    fake_checkpoint = tmp_path / "best.pt"
    fake_checkpoint.write_bytes(b"")  # doesn't need to be a real checkpoint for this guard
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("record_id,participant_id,label,text\n1,p1,depressed,hello\n")

    with pytest.raises(ValueError, match="locked"):
        evaluate(fake_checkpoint, test_csv, tmp_path / "out")


def test_evaluate_refuses_validation_test_csv(tmp_path: Path) -> None:
    """A file containing 'test' (but not 'validation') should be refused."""
    from training.text.evaluate_text import evaluate

    test_csv = tmp_path / "final_test_set.csv"
    test_csv.write_text("record_id,participant_id,label,text\n1,p1,depressed,hello\n")
    with pytest.raises(ValueError, match="locked"):
        evaluate(tmp_path / "best.pt", test_csv, tmp_path / "out")


def test_compute_metrics_binary() -> None:
    from training.text.evaluate_text import compute_metrics

    actual = [0, 0, 1, 1]
    predicted = [0, 0, 1, 0]
    probs = [0.1, 0.2, 0.9, 0.4]
    metrics = compute_metrics(actual, predicted, probs)
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert "brier_score" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
