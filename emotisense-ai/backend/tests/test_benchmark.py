"""Phase 20 — Tests for the benchmark runner and report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "test.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "record_id", "participant_id", "session_id",
            "dataset_id", "dataset_version", "label",
            "purpose", "alignment", "split",
        ])
        w.writeheader()
        w.writerow({
            "record_id": "T001", "participant_id": "P020", "session_id": "S020",
            "dataset_id": "daic_woz", "dataset_version": "1.0", "label": "depressed",
            "purpose": "multimodal_depression", "alignment": "participant_session_aligned",
            "split": "test",
        })
        w.writerow({
            "record_id": "T002", "participant_id": "P021", "session_id": "S021",
            "dataset_id": "daic_woz", "dataset_version": "1.0", "label": "control",
            "purpose": "multimodal_depression", "alignment": "participant_session_aligned",
            "split": "test",
        })
    return path


def _write_train_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "wrong.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "record_id", "participant_id", "session_id",
            "dataset_id", "dataset_version", "label",
            "purpose", "alignment", "split",
        ])
        w.writeheader()
        w.writerow({
            "record_id": "TR001", "participant_id": "P001", "session_id": "S001",
            "dataset_id": "daic_woz", "dataset_version": "1.0", "label": "depressed",
            "purpose": "multimodal_depression", "alignment": "participant_session_aligned",
            "split": "train",   # wrong — must be "test"
        })
    return path


# ---------------------------------------------------------------------------
# BenchmarkReport tests
# ---------------------------------------------------------------------------

class TestBenchmarkReport:

    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.run_benchmark import BenchmarkReport
        report = BenchmarkReport(
            run_timestamp="2026-08-21T00:00:00+00:00",
            hardware="test",
            status="no_checkpoint",
        )
        out = tmp_path / "report.json"
        report.save_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["status"] == "no_checkpoint"
        assert "disclaimer" in data

    def test_save_markdown_creates_file(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.run_benchmark import BenchmarkReport
        report = BenchmarkReport(status="no_checkpoint")
        out = tmp_path / "report.md"
        report.save_markdown(out)
        assert out.exists()
        content = out.read_text()
        assert "disclaimer" in content.lower()
        assert "EmotiSense AI" in content

    def test_sha256_file(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.run_benchmark import sha256_file
        f = tmp_path / "file.txt"
        f.write_bytes(b"hello")
        h = sha256_file(f)
        assert len(h) == 64
        assert sha256_file(f) == h  # deterministic

    def test_validate_test_manifest_accepts_valid(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.run_benchmark import validate_test_manifest
        path = _write_test_manifest(tmp_path)
        n_samples, n_parts = validate_test_manifest(path)
        assert n_samples == 2
        assert n_parts == 2

    def test_validate_test_manifest_rejects_train_rows(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.run_benchmark import validate_test_manifest
        path = _write_train_manifest(tmp_path)
        with pytest.raises(ValueError, match="Non-test row"):
            validate_test_manifest(path)

    def test_benchmark_report_disclaimer_present(self) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.run_benchmark import DISCLAIMER, BenchmarkReport
        report = BenchmarkReport()
        assert report.disclaimer == DISCLAIMER
        assert "research dataset" in DISCLAIMER.lower()
        assert "not evidence of clinical validity" in DISCLAIMER.lower()


# ---------------------------------------------------------------------------
# Report generator tests
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_empty_reports_dir_produces_no_results_section(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.generate_report import render_report
        out = tmp_path / "report.md"
        render_report([], out)
        content = out.read_text()
        assert "No benchmark reports found" in content
        assert "disclaimer" in content.lower()

    def test_report_with_data_includes_metrics_section(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.generate_report import render_report
        fake_report = {
            "run_timestamp": "2026-08-21",
            "hardware": "test",
            "status": "complete",
            "fusion_result": {
                "metrics": {
                    "f1": 0.72, "roc_auc": 0.81,
                    "accuracy": 0.78, "precision": 0.70,
                    "recall": 0.74, "pr_auc": 0.68,
                    "brier_score": 0.18, "sensitivity": 0.74, "specificity": 0.80,
                }
            },
            "notes": [],
        }
        out = tmp_path / "report.md"
        render_report([fake_report], out)
        content = out.read_text()
        assert "0.7200" in content
        assert "limitations" in content.lower()  # section heading uses "Key limitations"

    def test_generated_report_contains_disclaimer(self, tmp_path: Path) -> None:
        import sys
        sys.path.insert(0, str(Path("c:/Users/mh129/majorProject/emotisense-ai")))
        from experiments.scripts.generate_report import render_report
        out = tmp_path / "report.md"
        render_report([], out)
        content = out.read_text()
        assert "not evidence of clinical validity" in content.lower()


# ---------------------------------------------------------------------------
# Model card exists and is complete
# ---------------------------------------------------------------------------

def test_model_card_exists() -> None:
    path = Path("c:/Users/mh129/majorProject/emotisense-ai/docs/model-card.md")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    for required in (
        "EmotiSense AI",
        "Mandatory disclaimer",
        "medical",          # covers "not a medical\n> device" (word-wrapped in markdown)
        "Training data requirements",
        "Ethical considerations",
        "Caveats and known limitations",
        "How to reproduce",
    ):
        assert required in content, f"Model card missing section: {required!r}"


def test_viva_preparation_exists() -> None:
    path = Path("c:/Users/mh129/majorProject/emotisense-ai/docs/viva/viva-preparation.md")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    for topic in (
        "DeBERTa",
        "participant-disjoint",
        "temperature scaling",
        "Grad-CAM",
        "crisis-first",
        "LIME",
    ):
        assert topic in content, f"Viva prep missing topic: {topic!r}"


def test_system_summary_exists_and_complete() -> None:
    path = Path("c:/Users/mh129/majorProject/emotisense-ai/docs/viva/system-summary.md")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    # All 20 phases must appear in the summary
    for phase_n in range(1, 21):
        assert f"| {phase_n} |" in content or str(phase_n) in content, \
            f"Phase {phase_n} missing from system summary"
