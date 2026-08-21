"""Phase 20 — Reproducible experiment report generator.

Reads all benchmark JSON files from experiments/reports/ and produces
a consolidated Markdown report with tables, provenance, and limitations.

Usage
-----
    python experiments/scripts/generate_report.py \
        --reports-dir experiments/reports \
        --output      experiments/reports/full-report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


HEADER = """\
# EmotiSense AI — Consolidated Experiment Report

> **Research disclaimer:** All metrics are computed on research datasets under
> participant-disjoint evaluation conditions. Performance on clinical populations
> has not been established. These results are not evidence of clinical validity
> and must not be interpreted as diagnostic accuracy.

## System overview

EmotiSense AI is a multimodal early-risk research screening tool. It accepts
text, voice, and facial input and produces a calibrated risk score with
modality-level explanations.

Architecture: DeBERTa-v3 (text) + CNN-BiLSTM (voice) + MobileNetV2/ResNet-18
(face) → reliability-aware gated cross-modal attention fusion → risk head +
emotion profile + support policy engine.

"""

LIMITATIONS_SECTION = """
## Key limitations

- Depression-risk labels from research datasets are not clinical diagnoses.
- All model outputs are research signals; none constitute medical advice.
- Emotion recognition is noisy and culturally dependent.
- Acted emotion datasets (RAVDESS, CREMA-D, FER-2013) may not generalise to
  naturalistic expressions.
- Performance on DAIC-WOZ benchmark interviews may not generalise to casual
  recordings.
- Explainability outputs (LIME, SHAP, Grad-CAM) explain model behaviour, not
  clinical causality.
- All reported metrics require access to approved clinical datasets. No metrics
  are fabricated for this report.

"""

REPRODUCIBILITY_SECTION = """
## Reproducibility checklist

| Item | Status |
|---|---|
| Participant-disjoint splits with fixed random seed | Required |
| Split manifests SHA-256 recorded | Required |
| Model checkpoint SHA-256 recorded | Required |
| Temperature calibration on validation set only | Required |
| Final test evaluation after model lock | Required |
| Training hyperparameters logged to configs/model.yaml | Done |
| Dataset registry and data cards in configs/datasets.yaml | Done |
| All code versioned in git with commit hash | Recommended |

"""


def load_reports(reports_dir: Path) -> list[dict]:
    reports = []
    for f in sorted(reports_dir.rglob("benchmark-report.json")):
        try:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return reports


def render_report(reports: list[dict], output_path: Path) -> None:
    lines = [HEADER]

    if not reports:
        lines.append(
            "## Benchmark results\n\n"
            "No benchmark reports found. The model has not been trained on approved "
            "aligned data yet. Run `experiments/scripts/run_benchmark.py` after training "
            "to populate this section.\n"
        )
    else:
        lines.append("## Benchmark results\n")
        for report in reports:
            ts = report.get("run_timestamp", "unknown")
            hw = report.get("hardware", "unknown")
            status = report.get("status", "unknown")
            lines += [
                f"### Run: {ts}",
                f"**Hardware:** {hw}  **Status:** {status}",
                "",
            ]
            fusion = report.get("fusion_result", {})
            m = fusion.get("metrics", {})
            if m:
                lines += [
                    "#### Fusion model — test set",
                    "| Metric | Value |",
                    "|---|---|",
                ]
                for key in ("accuracy", "precision", "recall", "f1",
                            "roc_auc", "pr_auc", "brier_score",
                            "sensitivity", "specificity"):
                    v = m.get(key)
                    lines.append(f"| {key.replace('_', ' ').title()} | "
                                 f"{'—' if v is None else f'{v:.4f}'} |")
                lines.append("")

            for note in report.get("notes", []):
                lines.append(f"> {note}")
            lines.append("")

    lines.append(LIMITATIONS_SECTION)
    lines.append(REPRODUCIBILITY_SECTION)

    # Footer
    lines += [
        f"---\n*Generated: {datetime.now(timezone.utc).isoformat()}*\n"
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path,
                        default=Path("experiments/reports"))
    parser.add_argument("--output",      type=Path,
                        default=Path("experiments/reports/full-report.md"))
    args = parser.parse_args()

    reports = load_reports(args.reports_dir)
    render_report(reports, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
