"""Build the Phase 2 comparison report from the saved experiment results.

Reads every results/phase2/{text,voice}/*.json written by the experiment
runners and emits results/phase2/comparisons/comparison.{json,md}. Nothing is
computed that is not present in those files — a missing experiment is reported
as missing, never filled in.

Usage:
    .venv/Scripts/python scripts/phase2/build_comparison_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
OUT_DIR = RESULTS_DIR / "comparisons"

LABELS = {
    "eatd_text_tfidf_v1": "TF-IDF + Logistic Regression",
    "eatd_text_transformer_v1": "Transformer (frozen encoder + head)",
    "eatd_text_transformer_ft_v1": "Transformer (top-layer fine-tune)",
    "eatd_voice_egemaps_v1": "eGeMAPS + Logistic Regression",
    "eatd_voice_wav2vec2_v1": "Wav2Vec2 (frozen embeddings + head)",
}
ORDER = list(LABELS)

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "sensitivity", "specificity", "balanced_accuracy")


def _load_all() -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*/*.json")):
        if path.parent.name == "comparisons":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        exp = payload.get("experiment_id")
        if exp in LABELS:
            found[exp] = payload
    return found


def _row(exp: str, payload: dict) -> dict:
    val = payload.get("validation_metrics", {}).get("participant_level", {})
    test = payload.get("test_metrics", {}).get("participant_level", {})
    test_sess = payload.get("test_metrics", {}).get("session_level", {})
    ci = payload.get("test_metrics", {}).get("participant_bootstrap_ci", {}).get("ci", {})
    cal = payload.get("test_metrics", {}).get("calibration", {})
    training = payload.get("training", {})
    embed = training.get("embedding", {}) if isinstance(training.get("embedding"), dict) else {}
    return {
        "experiment_id": exp,
        "model": LABELS[exp],
        "modality": payload.get("modality"),
        "model_version": payload.get("model_version"),
        "validation_participant": {k: val.get(k) for k in METRICS},
        "test_participant": {k: test.get(k) for k in METRICS},
        "test_session": {k: test_sess.get(k) for k in METRICS},
        "test_participant_roc_auc_ci": ci.get("roc_auc"),
        "test_participant_f1_ci": ci.get("f1"),
        "test_confusion_matrix": test.get("confusion_matrix"),
        "test_calibration": {
            "brier": cal.get("brier_score"),
            "ece": cal.get("expected_calibration_error"),
        },
        "val_test_roc_auc_gap": (
            round(float(val["roc_auc"]) - float(test["roc_auc"]), 4)
            if val.get("roc_auc") is not None and test.get("roc_auc") is not None
            else None
        ),
        "training_seconds": training.get("fit_seconds"),
        "embedding_seconds": embed.get("embed_seconds") or embed.get("encode_seconds"),
        "inference_seconds_test": training.get("inference_seconds_test"),
        "peak_gpu_memory_gb": training.get("peak_gpu_memory_gb") or embed.get("peak_gpu_memory_gb"),
    }


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _md_table(rows: List[dict]) -> str:
    lines = [
        "| Model | Modality | Accuracy | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        t = r["test_participant"]
        lines.append(
            f"| {r['model']} | {r['modality']} | {_fmt(t['accuracy'])} | "
            f"{_fmt(t['f1'])} | {_fmt(t['roc_auc'])} | {_fmt(t['pr_auc'])} |"
        )
    return "\n".join(lines)


def _markdown(rows: List[dict], missing: List[str]) -> str:
    out: List[str] = []
    out.append("# Phase 2 — Unimodal baselines on the verified EATD participant splits\n")
    out.append(
        "**Test set: 24 participants, 8 in the positive (SDS index >= 50) class.** "
        "All test numbers are participant-level (mean of a participant's session "
        "probabilities) unless stated otherwise. 95% percentile CIs come from a "
        "participant-cluster bootstrap (2000 resamples). These intervals are wide by "
        "construction.\n"
    )
    out.append("## Headline comparison (locked test split, participant level)\n")
    out.append(_md_table(rows))
    out.append("\n## Accuracy / ROC-AUC with participant bootstrap 95% CI\n")
    out.append("| Model | ROC-AUC | ROC-AUC 95% CI | F1 | F1 95% CI |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        t = r["test_participant"]
        auc_ci = r["test_participant_roc_auc_ci"] or {}
        f1_ci = r["test_participant_f1_ci"] or {}
        out.append(
            f"| {r['model']} | {_fmt(t['roc_auc'])} | "
            f"{_fmt(auc_ci.get('lo'))}–{_fmt(auc_ci.get('hi'))} | {_fmt(t['f1'])} | "
            f"{_fmt(f1_ci.get('lo'))}–{_fmt(f1_ci.get('hi'))} |"
        )

    out.append("\n## Validation vs test (overfitting / selection optimism)\n")
    out.append("| Model | Val ROC-AUC | Test ROC-AUC | Gap | Val F1 | Test F1 |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['model']} | {_fmt(r['validation_participant']['roc_auc'])} | "
            f"{_fmt(r['test_participant']['roc_auc'])} | {_fmt(r['val_test_roc_auc_gap'])} | "
            f"{_fmt(r['validation_participant']['f1'])} | {_fmt(r['test_participant']['f1'])} |"
        )

    out.append("\n## Session-level test performance\n")
    out.append("| Model | Accuracy | F1 | ROC-AUC |")
    out.append("|---|---|---|---|")
    for r in rows:
        s = r["test_session"]
        out.append(f"| {r['model']} | {_fmt(s['accuracy'])} | {_fmt(s['f1'])} | {_fmt(s['roc_auc'])} |")

    out.append("\n## Confusion matrices (participant level, threshold 0.5)\n")
    out.append("| Model | TP | FP | FN | TN |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        cm = r["test_confusion_matrix"] or {}
        out.append(
            f"| {r['model']} | {cm.get('tp')} | {cm.get('fp')} | {cm.get('fn')} | {cm.get('tn')} |"
        )

    out.append("\n## Calibration and cost\n")
    out.append("| Model | Brier | ECE (10 bins) | Head train (s) | Embed (s) | Test inference (s) | Peak GPU (GB) |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["test_calibration"]
        out.append(
            f"| {r['model']} | {_fmt(c['brier'])} | {_fmt(c['ece'])} | "
            f"{_fmt(r['training_seconds'])} | {_fmt(r['embedding_seconds'])} | "
            f"{_fmt(r['inference_seconds_test'])} | {_fmt(r['peak_gpu_memory_gb'])} |"
        )
    out.append(
        "\n_Head train and embed times come from the run that produced the stored "
        "artifacts; embedding extraction is skipped on later runs that reuse the cache._"
    )

    below = [r["model"] for r in rows if (r["test_participant"].get("roc_auc") or 0) < 0.5]
    out.append("\n## Failed / negative results (reported, not hidden)\n")
    if below:
        out.append(
            "Below-chance test ROC-AUC (0.5 = chance): " + ", ".join(below) + "."
        )
    else:
        out.append("No model fell below chance on the test split.")
    out.append("")
    for r in rows:
        gap = r["val_test_roc_auc_gap"]
        if gap is not None and gap > 0.15:
            out.append(
                f"- **{r['model']}**: validation ROC-AUC minus test ROC-AUC = {gap:.3f}. "
                "With 24 validation participants this is consistent with selection optimism "
                "and small-sample variance, not evidence of a generalisable model."
            )
    if missing:
        out.append("\nMissing experiments (not run): " + ", ".join(missing))
    out.append(
        "\n## Caveats\n"
        "- 24 test participants / 8 positive → every point estimate is unstable.\n"
        "- Labels are EATD SDS bands (index >= 50), not PHQ-9 or clinical diagnoses.\n"
        "- No model here is clinically validated, and none is enabled for serving.\n"
    )
    return "\n".join(out)


def main() -> int:
    found = _load_all()
    rows = [_row(exp, found[exp]) for exp in ORDER if exp in found]
    missing = [LABELS[e] for e in ORDER if e not in found]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 2,
        "test_participants": 24,
        "test_positive_participants": 8,
        "aggregation": "participant probability = mean of session probabilities",
        "bootstrap": "participant-cluster percentile bootstrap, 2000 resamples, seed 0",
        "rows": rows,
        "missing_experiments": missing,
    }
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (OUT_DIR / "comparison.md").write_text(_markdown(rows, missing), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "missing": missing,
                      "out_dir": str(OUT_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
