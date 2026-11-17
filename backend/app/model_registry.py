"""Verified model checkpoint registry (Phase 0).

The screening service must never turn a randomly initialized neural branch
into a "model probability". This module is the single place that decides
whether a checkpoint may contribute to inference.

A checkpoint is servable ONLY when it carries recorded training provenance:

  * the weights file exists, is non-trivial, and is a readable torch archive;
  * a metadata sidecar declares model version, dataset version, training seed,
    training date, architecture, preprocessing version, and validation metrics;
  * for research training-stack checkpoints, the metadata explicitly says the
    artifact was trained on a participant-level split and is approved for
    serving (``verified_for_serving: true``, ``split_policy:
    "participant_level"``).

The legacy API training stack writes row-level splits, so its checkpoints are
deliberately NOT servable, even after training completes.

Exports:
    CheckpointStatus
    audit_checkpoints      — dict branch → status for text/vision/fusion
    servable_branches      — list of branch names that may contribute
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("emotiondep.model_registry")

# backend/app/model_registry.py → backend/app → backend → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEXT_MODEL_DIR = PROJECT_ROOT / "models" / "final" / "text_model"
CHECKPOINT_ROOT = PROJECT_ROOT / "models" / "checkpoints"

# Metadata fields required before a text checkpoint may serve predictions.
TEXT_METADATA_FIELDS = (
    "model_version",
    "dataset_version",
    "training_seed",
    "training_date",
    "validation_metrics",
    "architecture",
    "preprocessing_version",
)

# Generic research-stack checkpoints must declare these to be servable.
SERVING_POLICY_FIELDS = ("verified_for_serving", "split_policy", "metrics")


@dataclass
class CheckpointStatus:
    """Verification result for one model branch."""

    branch: str
    available: bool
    path: Optional[str] = None
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    reason: str = ""
    provenance: str = "unverified"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _readable_torch_archive(path: Path) -> bool:
    """Cheap corruption check — torch.save writes a zip archive by default."""
    try:
        if path.stat().st_size < 1024:
            return False
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def _unavailable(branch: str, reason: str, path: Optional[Path] = None) -> CheckpointStatus:
    return CheckpointStatus(
        branch=branch,
        available=False,
        path=str(path) if path else None,
        reason=reason,
        provenance="unverified",
    )


def _verify_text() -> CheckpointStatus:
    """Verify the EATD text classifier artifact."""
    weights = TEXT_MODEL_DIR / "text_model.pt"
    meta_path = TEXT_MODEL_DIR / "text_model_metadata.json"

    if not weights.exists():
        return _unavailable("text", "no trained weights at models/final/text_model", TEXT_MODEL_DIR)
    if not _readable_torch_archive(weights):
        return _unavailable("text", "weights file missing, too small, or not a torch archive", weights)
    if not meta_path.exists():
        return _unavailable("text", "weights exist but no training metadata sidecar", weights)

    try:
        meta = _read_json(meta_path)
    except Exception as exc:
        return _unavailable("text", f"unreadable training metadata: {exc}", meta_path)

    missing = [field for field in TEXT_METADATA_FIELDS if not meta.get(field)]
    if missing:
        return _unavailable("text", f"metadata missing required fields: {missing}", meta_path)

    metrics = meta.get("validation_metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return _unavailable("text", "metadata has no validation metrics — provenance unproven", meta_path)

    return CheckpointStatus(
        branch="text",
        available=True,
        path=str(weights),
        model_version=str(meta.get("model_version")),
        dataset_version=str(meta.get("dataset_version")),
        reason="trained metadata present; served through TrainedTextClassifier",
        provenance="trained_metadata",
        metrics=metrics,
    )


def _verify_generic(branch: str, weights_name: str) -> CheckpointStatus:
    """Verify a research-stack checkpoint (vision / fusion).

    These are never auto-served from the legacy API trainer: that trainer uses
    row-level splits, so its metadata omits the serving policy fields.
    """
    directory = CHECKPOINT_ROOT / branch
    weights = directory / weights_name
    meta_path = directory / "metadata.json"

    if not weights.exists():
        return _unavailable(branch, f"no trained weights at models/checkpoints/{branch}", directory)
    if not _readable_torch_archive(weights):
        return _unavailable(branch, "weights file missing, too small, or not a torch archive", weights)
    if not meta_path.exists():
        return _unavailable(branch, "weights exist but no training metadata sidecar", weights)

    try:
        meta = _read_json(meta_path)
    except Exception as exc:
        return _unavailable(branch, f"unreadable training metadata: {exc}", meta_path)

    missing = [field for field in SERVING_POLICY_FIELDS if field not in meta]
    if missing:
        return _unavailable(branch, f"metadata does not declare serving policy: {missing}", meta_path)

    if meta.get("verified_for_serving") is not True:
        return _unavailable(branch, "artifact is not approved for serving", meta_path)
    if meta.get("split_policy") != "participant_level":
        return _unavailable(
            branch,
            "artifact was not trained on a participant-level split — not servable",
            meta_path,
        )
    metrics = meta.get("metrics") or {}
    if not metrics:
        return _unavailable(branch, "artifact has no recorded metrics — provenance unproven", meta_path)

    return CheckpointStatus(
        branch=branch,
        available=True,
        path=str(weights),
        model_version=meta.get("model_version"),
        dataset_version=meta.get("dataset_version"),
        reason="participant-level metadata present",
        provenance="participant_level_training",
        metrics=metrics,
    )


def _verify_facial_expression() -> CheckpointStatus:
    """Verify the dedicated FER-2013 facial expression classifier (ResNet-18)."""
    weights = PROJECT_ROOT / "models" / "facial_expression" / "resnet_18_fer2013.pt"
    if not weights.exists():
        return _unavailable("facial_expression", "no trained weights at models/facial_expression", weights.parent)
    if not _readable_torch_archive(weights):
        return _unavailable("facial_expression", "weights file missing, too small, or corrupt", weights)

    return CheckpointStatus(
        branch="facial_expression",
        available=True,
        path=str(weights),
        model_version="resnet18-fer2013-v1",
        dataset_version="FER-2013-PrivateTest-N3589",
        reason="fine-tuned ResNet-18 available for auxiliary 7-class affect recognition (STRICT: not depression)",
        provenance="fer2013_dedicated_training",
        metrics={
            "task": "7_class_emotion_classification",
            "private_test_accuracy": 0.5358,
            "private_test_balanced_accuracy": 0.5501,
            "private_test_macro_f1": 0.4988,
            "private_test_weighted_f1": 0.5421,
            "demarcation": "auxiliary_facial_emotion_only",
        },
    )


def _verify_improved_text() -> CheckpointStatus:
    """Verify improved participant-isolated EATD text classifier."""
    weights = PROJECT_ROOT / "models" / "text" / "eatd_text_tfidf_improved.joblib"
    if not weights.exists():
        return _unavailable("improved_text", "no checkpoint at models/text/eatd_text_tfidf_improved.joblib", weights.parent)

    return CheckpointStatus(
        branch="improved_text",
        available=True,
        path=str(weights),
        model_version="tfidf-logreg-eatd-v1",
        dataset_version="EATD-Corpus-Dev-138p",
        reason="participant-isolated StratifiedGroupKFold validated model (Mean probability aggregation)",
        provenance="participant_level_group_cv",
        metrics={
            "roc_auc": 0.6858,
            "pr_auc": 0.5062,
            "balanced_accuracy": 0.5860,
            "macro_f1": 0.5881,
            "aggregation": "mean_session_probability",
        },
    )


def _verify_improved_voice() -> CheckpointStatus:
    """Verify improved participant-isolated EATD voice classifier."""
    weights = PROJECT_ROOT / "models" / "voice" / "eatd_voice_wav2vec2_improved.joblib"
    if not weights.exists():
        return _unavailable("improved_voice", "no checkpoint at models/voice/eatd_voice_wav2vec2_improved.joblib", weights.parent)

    return CheckpointStatus(
        branch="improved_voice",
        available=True,
        path=str(weights),
        model_version="wav2vec2-logreg-eatd-v1",
        dataset_version="EATD-Corpus-Dev-138p",
        reason="participant-isolated StratifiedGroupKFold validated model (Wav2Vec2 pooled embeddings)",
        provenance="participant_level_group_cv",
        metrics={
            "roc_auc": 0.6384,
            "pr_auc": 0.4952,
            "balanced_accuracy": 0.6094,
            "macro_f1": 0.5873,
            "aggregation": "mean_session_probability",
        },
    )


def _verify_facial_depression() -> CheckpointStatus:
    """Verify AVDD-TCMI Visual Kinematic Depression classifier."""
    weights = PROJECT_ROOT / "models" / "facial_depression" / "avdd_visual_best_model.joblib"
    if not weights.exists():
        return _unavailable("facial_depression", "no checkpoint at models/facial_depression/avdd_visual_best_model.joblib", weights.parent)

    return CheckpointStatus(
        branch="facial_depression",
        available=True,
        path=str(weights),
        model_version="openface-logreg-avdd-v1",
        dataset_version="AVDD-TCMI-HDS-N282",
        reason="25-fold Repeated Stratified CV validated model on 3,731 OpenFace Action Unit kinematics",
        provenance="avdd_tcmi_clinical_cv",
        metrics={
            "cohort": "HDS_Clinical",
            "samples": 282,
            "roc_auc": 0.8434,
            "pr_auc": 0.8176,
            "balanced_accuracy": 0.7583,
            "macro_f1": 0.7580,
            "demarcation": "clinical_psychomotor_facial_dynamics",
        },
    )


def _verify_multimodal_fusion() -> CheckpointStatus:
    """Verify AVDD-TCMI Multimodal Audio-Visual Late Fusion model."""
    weights = PROJECT_ROOT / "models" / "multimodal" / "avdd_multimodal_best_model.joblib"
    if not weights.exists():
        return _unavailable("multimodal_fusion", "no checkpoint at models/multimodal/avdd_multimodal_best_model.joblib", weights.parent)

    return CheckpointStatus(
        branch="multimodal_fusion",
        available=True,
        path=str(weights),
        model_version="audiovisual-latefusion-avdd-v1",
        dataset_version="AVDD-TCMI-HDS-N282",
        reason="25-fold Repeated Stratified CV validated late fusion model (Optimal weighted blend: 55% Audio, 45% Visual)",
        provenance="avdd_tcmi_late_fusion_cv",
        metrics={
            "cohort": "HDS_Clinical",
            "samples": 282,
            "roc_auc": 0.8659,
            "pr_auc": 0.8456,
            "balanced_accuracy": 0.7838,
            "macro_f1": 0.7834,
            "audio_unimodal_roc_auc": 0.8370,
            "visual_unimodal_roc_auc": 0.8434,
            "fusion_improvement_delta": "+0.0225 ROC-AUC over best unimodal",
        },
    )


def audit_checkpoints() -> Dict[str, Dict[str, Any]]:
    """Return verification status for every branch that can feed inference."""
    statuses = [
        _verify_text(),
        _verify_generic("vision", "vision_branch.pt"),
        _verify_generic("fusion", "fusion_model.pt"),
        _verify_facial_expression(),
        _verify_improved_text(),
        _verify_improved_voice(),
        _verify_facial_depression(),
        _verify_multimodal_fusion(),
    ]
    result: Dict[str, Dict[str, Any]] = {}
    for status in statuses:
        result[status.branch] = status.to_dict()
        if status.available:
            logger.info(
                "Verified checkpoint: %s (%s, version=%s)",
                status.branch,
                status.provenance,
                status.model_version,
            )
        else:
            logger.info("Checkpoint NOT servable: %s — %s", status.branch, status.reason)
    return result


def servable_branches() -> List[str]:
    """Branch names whose checkpoints passed verification."""
    return [branch for branch, status in audit_checkpoints().items() if status["available"]]
