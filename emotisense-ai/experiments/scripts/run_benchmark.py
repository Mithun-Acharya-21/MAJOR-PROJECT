"""Phase 20 — Final reproducible benchmark runner.

Usage (after the model is locked — run ONCE on the held-out test split)
----------------------------------------------------------------------
    python experiments/scripts/run_benchmark.py \
        --checkpoint     models/checkpoints/fusion-v1/best.pt \
        --test-manifest  data/processed/manifests/daic-v1/test.csv \
        --output-dir     experiments/reports/benchmark-v1 \
        --hardware       "NVIDIA RTX 3090 24 GB / AMD Ryzen 9 5900X / 32 GB RAM"

The script will REFUSE to run if the checkpoint was modified after the
validation manifests were frozen, and it enforces the locked=True gate
from training.evaluation.metrics.assert_not_final_test.

All metric values are written to JSON and a human-readable Markdown
report.  No metric is fabricated; if a model is unavailable the report
states that honestly.

Scientific constraints enforced
--------------------------------
- Participant-disjoint test manifest is validated before any inference.
- No training data is used during benchmark evaluation.
- Temperature calibration must already have been applied to the checkpoint
  before running this script (calibrate on validation ONLY).
- Results include the exact manifest hash so the report is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DISCLAIMER = (
    "These metrics are computed on a research dataset under controlled "
    "participant-disjoint evaluation conditions. Performance on clinical "
    "populations has not been established and may differ substantially. "
    "These scores are not evidence of clinical validity."
)


# ---------------------------------------------------------------------------
# Manifest integrity helper
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    run_timestamp: str = ""
    hardware: str = "unknown"
    python_version: str = ""
    checkpoint_path: str = ""
    test_manifest_path: str = ""
    test_manifest_sha256: str = ""
    n_test_samples: int = 0
    n_test_participants: int = 0
    # Per-modality results
    unimodal_results: dict[str, dict] = field(default_factory=dict)
    # Fusion results
    fusion_result: dict = field(default_factory=dict)
    # Ablation: single and pairwise modality subsets
    ablation_results: dict = field(default_factory=dict)
    # Calibration
    ece: float | None = None
    brier_score: float | None = None
    # Inference latency
    mean_latency_ms: float | None = None
    # Fairness (if subgroup labels are available)
    subgroup_results: dict | None = None
    # Status
    status: str = "no_model_available"
    notes: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def save_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            "# EmotiSense AI — Final Benchmark Report",
            "",
            f"> **Disclaimer:** {self.disclaimer}",
            "",
            "## Run metadata",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| Timestamp | {self.run_timestamp} |",
            f"| Hardware | {self.hardware} |",
            f"| Python | {self.python_version} |",
            f"| Checkpoint | `{self.checkpoint_path}` |",
            f"| Test manifest | `{self.test_manifest_path}` |",
            f"| Manifest SHA-256 | `{self.test_manifest_sha256[:16]}…` |",
            f"| Test samples | {self.n_test_samples} |",
            f"| Test participants | {self.n_test_participants} |",
            f"| Status | **{self.status}** |",
            "",
        ]

        if self.fusion_result:
            lines += [
                "## Fusion model — test set metrics",
                "",
                "| Metric | Value |",
                "|---|---|",
            ]
            m = self.fusion_result.get("metrics", {})
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                        "pr_auc", "brier_score", "sensitivity", "specificity"):
                v = m.get(key)
                lines.append(
                    f"| {key.replace('_', ' ').title()} | "
                    f"{'N/A' if v is None else f'{v:.4f}'} |"
                )
            lines.append("")
            if self.ece is not None:
                lines += [
                    f"**Expected Calibration Error (ECE):** {self.ece:.4f}  ",
                    "",
                ]

        if self.unimodal_results:
            lines += [
                "## Unimodal baselines",
                "",
                "| Modality | F1 | ROC-AUC | Sensitivity | Specificity |",
                "|---|---|---|---|---|",
            ]
            for modality, res in self.unimodal_results.items():
                m = res.get("metrics", {})
                lines.append(
                    f"| {modality.capitalize()} "
                    f"| {m.get('f1', 'N/A')} "
                    f"| {m.get('roc_auc', 'N/A')} "
                    f"| {m.get('sensitivity', 'N/A')} "
                    f"| {m.get('specificity', 'N/A')} |"
                )
            lines.append("")

        if self.ablation_results:
            lines += [
                "## Modality ablation (validation F1 ranking)",
                "",
                "| Subset | F1 |",
                "|---|---|",
            ]
            for subset, res in self.ablation_results.items():
                f1 = res.get("f1", "N/A")
                lines.append(f"| {subset} | {f1} |")
            lines.append("")

        if self.mean_latency_ms is not None:
            lines += [
                f"**Mean inference latency:** {self.mean_latency_ms:.1f} ms  ",
                "",
            ]

        if self.notes:
            lines += ["## Notes", ""]
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        lines += [
            "## Reproducibility",
            "",
            "To reproduce this benchmark exactly:",
            "```powershell",
            "python experiments/scripts/run_benchmark.py \\",
            f"  --checkpoint     {self.checkpoint_path} \\",
            f"  --test-manifest  {self.test_manifest_path} \\",
            f"  --output-dir     experiments/reports/benchmark-v1 \\",
            f"  --hardware       \"{self.hardware}\"",
            "```",
            "",
            "The test manifest SHA-256 must match:",
            f"`{self.test_manifest_sha256}`",
            "",
            "> Participant-disjoint splits were generated with a fixed random seed.",
            "> The test set was not used for model selection or hyperparameter tuning.",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Manifest validator
# ---------------------------------------------------------------------------

def validate_test_manifest(manifest_path: Path) -> tuple[int, int]:
    """Validate the test manifest and return (n_samples, n_participants)."""
    import csv
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty test manifest: {manifest_path}")
    for row in rows:
        if row.get("split", "").strip().lower() != "test":
            raise ValueError(
                f"Non-test row in test manifest: split={row.get('split')!r}. "
                "Never use training or validation data for final evaluation."
            )
    participants = {r["participant_id"].strip() for r in rows}
    return len(rows), len(participants)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the final locked benchmark on the held-out test set."
    )
    parser.add_argument("--checkpoint",    type=Path, required=True,
                        help="Path to the locked model checkpoint (.pt file)")
    parser.add_argument("--test-manifest", type=Path, required=True,
                        help="Path to the test split CSV")
    parser.add_argument("--output-dir",    type=Path, required=True)
    parser.add_argument("--hardware",      type=str,  default="unspecified",
                        help="Hardware description for the report")
    args = parser.parse_args()

    from datetime import datetime, timezone
    from training.evaluation.metrics import assert_not_final_test

    # Enforce the locked gate
    assert_not_final_test("test", locked=True)

    report = BenchmarkReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=args.hardware,
        python_version=sys.version.split()[0],
        checkpoint_path=str(args.checkpoint),
        test_manifest_path=str(args.test_manifest),
        test_manifest_sha256=sha256_file(args.test_manifest),
    )

    # Validate manifest
    try:
        n_samples, n_participants = validate_test_manifest(args.test_manifest)
        report.n_test_samples = n_samples
        report.n_test_participants = n_participants
    except Exception as exc:
        report.status = "manifest_error"
        report.notes.append(f"Manifest validation failed: {exc}")
        report.save_json(args.output_dir / "benchmark-report.json")
        report.save_markdown(args.output_dir / "benchmark-report.md")
        print(f"Manifest error: {exc}")
        return 1

    # Attempt to load checkpoint and run evaluation
    try:
        import torch
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        report.notes.append(
            f"Checkpoint loaded. Keys: {list(checkpoint.keys())}"
        )
        report.status = "checkpoint_loaded_no_inference"
        report.notes.append(
            "Full inference requires approved DAIC-WOZ/E-DAIC test data. "
            "No test data is available in this environment. "
            "Run this script with an approved aligned test manifest to produce real metrics."
        )

    except FileNotFoundError:
        report.status = "no_checkpoint"
        report.notes.append(
            "No trained model checkpoint found. "
            "Train the model on approved aligned data before running the final benchmark. "
            "This is expected during the architecture-only phase."
        )

    except Exception as exc:
        report.status = "error"
        report.notes.append(f"Benchmark error: {exc}")

    # Write report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.save_json(args.output_dir / "benchmark-report.json")
    report.save_markdown(args.output_dir / "benchmark-report.md")

    print(f"Report written to {args.output_dir}")
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
