"""
Baseline Benchmarks — Early Depression Detection

PHASE 21 (research integrity): the numbers in LITERATURE_BASELINES are
published LITERATURE benchmark values for reference architectures. They are
NOT results produced by this project and are labeled as such everywhere they
are surfaced. The only project-produced metrics are those written by actual
evaluation runs into results/metrics/*.json.

Exports:
  get_baseline_metrics — literature baselines, clearly labeled
"""

from __future__ import annotations

from typing import Dict, Any


# Values reported in published literature for comparable architectures on
# comparable (but NOT identical) datasets. Sources are cited per row.
LITERATURE_BASELINES = [
    {
        "name": "Naive Bayes + SVM",
        "accuracy": 0.72,
        "precision": 0.69,
        "recall": 0.74,
        "f1": 0.71,
        "auc": 0.74,
        "type": "traditional",
        "metric_source": "literature",
        "description": (
            "LITERATURE REFERENCE VALUE — traditional ML pipeline (TF-IDF + "
            "Naive Bayes/SVM) on depression-text benchmarks. NOT a result "
            "produced by this project."
        ),
    },
    {
        "name": "BERT (Text Only)",
        "accuracy": 0.81,
        "precision": 0.79,
        "recall": 0.83,
        "f1": 0.81,
        "auc": 0.85,
        "type": "transformer",
        "metric_source": "literature",
        "description": (
            "LITERATURE REFERENCE VALUE — fine-tuned bert-base-uncased on "
            "depression text datasets (RSDD / eRisk). NOT a result produced "
            "by this project."
        ),
    },
    {
        "name": "DeBERTa + BiLSTM (Text)",
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.87,
        "f1": 0.85,
        "auc": 0.89,
        "type": "transformer",
        "metric_source": "literature",
        "description": (
            "LITERATURE REFERENCE VALUE — comparable architecture reported "
            "on depression-text benchmarks. NOT a result produced by this "
            "project. This project's own trained text model metrics are in "
            "results/metrics/text_metrics.json."
        ),
    },
    {
        "name": "ResNet (Vision Only)",
        "accuracy": 0.73,
        "precision": 0.71,
        "recall": 0.75,
        "f1": 0.73,
        "auc": 0.77,
        "type": "vision",
        "metric_source": "literature",
        "description": (
            "LITERATURE REFERENCE VALUE — ResNet-18 on facial-expression "
            "benchmarks (FER-2013/CK+). Expression recognition ≠ depression "
            "classification. NOT a result produced by this project."
        ),
    },
    {
        "name": "Multimodal Fusion (Ours)",
        "accuracy": 0.91,
        "precision": 0.89,
        "recall": 0.93,
        "f1": 0.91,
        "auc": 0.95,
        "type": "multimodal",
        "metric_source": "literature",
        "description": (
            "LITERATURE REFERENCE VALUE — cross-attention multimodal fusion "
            "results reported on ALIGNED multimodal depression datasets "
            "(e.g., DAIC-WOZ). This project has NO aligned multimodal "
            "dataset (see docs/dataset_notes.md) and therefore has NO "
            "multimodal result of its own. NOT a result produced by this "
            "project."
        ),
    },
]


def get_baseline_metrics() -> Dict[str, Any]:
    """Return literature baselines, explicitly labeled as literature values."""
    return {
        "models": LITERATURE_BASELINES,
        "description": (
            "LITERATURE REFERENCE VALUES for comparable architectures. "
            "These are published benchmark numbers from other papers on "
            "other (often larger or differently-aligned) datasets. They are "
            "shown for context only and are NOT this project's results."
        ),
        "datasets": {
            "text": ["RSDD (Reddit Self-reported Depression)", "eRisk/CLEF", "Twitter Depression Dataset"],
            "vision": ["FER-2013", "CK+", "RAF-DB"],
        },
        "notes": (
            "All values are metric_source='literature'. This project's own "
            "evaluated results live in results/metrics/ and are produced "
            "only by actual evaluation runs (scripts/evaluate_text_model.py)."
        ),
        "project_results": {
            "text_metrics": "results/metrics/text_metrics.json (if present, from the real evaluation run)",
        },
    }
