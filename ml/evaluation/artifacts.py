"""
Experiment artifact metadata contract for Phase 5 (Phase 5B).

Every Phase 5 experiment writes one JSON artifact whose metadata is complete
enough that a reader can tell what was computed, on which data, with which code,
and what could not be concluded. This module defines that contract, validates
it, and writes it without ever overwriting a previous artifact.

Honesty rules enforced by validation:

* ``clinical_validation`` and ``serving`` must both be ``false``;
* ``known_limitations`` cannot be empty;
* metrics must be either absent (``status: not_measured``) or backed by a named
  result source — an artifact cannot contain a performance number nobody
  produced;
* every metric carries its evaluation unit, and a null value must carry a reason;
* threshold/calibration partitions cannot be ``outer_evaluation`` or
  ``locked_test``.

Phase 5B writes no experiment artifact (no model was trained); this module is
the contract plus its validator, exercised by
``tests/test_phase5_artifacts.py``.

Exports:
    ARTIFACT_SCHEMA_VERSION, REQUIRED_ARTIFACT_FIELDS, SCHEMA_PATH
    ArtifactValidationError
    build_experiment_artifact, validate_experiment_artifact,
    write_experiment_artifact, load_artifact_schema
    collect_environment, git_revision, sha256_of_files
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

ARTIFACT_SCHEMA_VERSION = "phase5_experiment_artifact_v1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "ml" / "configs" / "phase5" / "experiment_artifact.schema.json"

#: Directories the harness must never write into (frozen Phase 2 outputs,
#: trained checkpoints, and the deployable model).
PROTECTED_OUTPUT_DIRS = (
    "results/phase2",
    "models/experiments",
    "models/final",
)

SELECTION_FORBIDDEN_PARTITIONS = ("outer_evaluation", "locked_test")

REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "phase",
    "experiment_id",
    "dataset",
    "modality",
    "feature_configuration",
    "label_definition",
    "evaluation",
    "preprocessing",
    "model",
    "selection",
    "results",
    "environment",
    "timestamps",
    "code_revision",
    "known_limitations",
    "clinical_validation",
    "serving",
)

_REQUIRED_NESTED: Mapping[str, Sequence[str]] = {
    "dataset": ("name", "version", "provenance_reference", "sha256"),
    "label_definition": ("rule", "positive_class", "definition"),
    "evaluation": (
        "unit",
        "n_units",
        "n_positive",
        "split_strategy",
        "seeds",
        "fold_assignments_reference",
        "fold_assignments_sha256",
    ),
    "preprocessing": ("configuration", "fit_partition"),
    "model": ("configuration", "hyperparameters"),
    "selection": (
        "threshold_method",
        "threshold_partition",
        "calibration_method",
        "calibration_partition",
    ),
    "results": ("status", "metrics", "confidence_intervals"),
    "environment": ("software_versions", "hardware"),
    "timestamps": ("started_at", "finished_at"),
}

_VALID_RESULT_STATUS = ("not_measured", "partial", "measured")
_VALID_UNITS = ("participant", "session", "sample", "group")


class ArtifactValidationError(ValueError):
    """The artifact metadata does not satisfy the Phase 5 contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_revision() -> str:
    """Current commit, or ``unknown`` when git is unavailable (never faked)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        rev = out.stdout.strip()
        return rev or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_of_files(paths: Iterable[Path | str]) -> str:
    """Aggregate SHA-256 over a set of files (missing files become ``missing:``)."""
    digest = hashlib.sha256()
    for path in paths:
        p = Path(path)
        digest.update(str(p.as_posix()).encode("utf-8"))
        if p.is_file():
            digest.update(p.read_bytes())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_environment(*, hardware_notes: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """Software versions and hardware description for the artifact."""
    try:
        import numpy
    except Exception:  # pragma: no cover
        numpy = None
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:  # pragma: no cover
        sklearn_version = None

    software: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": getattr(numpy, "__version__", None),
        "scikit_learn": sklearn_version,
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "pyyaml": _package_version("pyyaml"),
    }
    torch_version = _package_version("torch")
    if torch_version is not None:
        software["torch"] = torch_version
        try:
            import torch

            software["cuda_available"] = bool(torch.cuda.is_available())
            software["cuda_version"] = getattr(torch.version, "cuda", None)
        except Exception:  # pragma: no cover
            software["cuda_available"] = None

    hardware: Dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "system": platform.system(),
        "system_release": platform.release(),
    }
    if torch_version is not None:
        try:
            import torch

            hardware["accelerator"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            )
        except Exception:  # pragma: no cover
            hardware["accelerator"] = "unknown"
    else:
        hardware["accelerator"] = "not_applicable"
    if hardware_notes:
        hardware.update(dict(hardware_notes))
    return {"software_versions": software, "hardware": hardware}


def load_artifact_schema() -> Dict[str, object]:
    """Load the published JSON Schema for the artifact contract."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def build_experiment_artifact(
    *,
    experiment_id: str,
    dataset: Mapping[str, object],
    feature_configuration: Mapping[str, object],
    label_definition: Mapping[str, object],
    evaluation: Mapping[str, object],
    preprocessing: Mapping[str, object],
    model: Mapping[str, object],
    selection: Mapping[str, object],
    modality: str = "text",
    phase: str = "5B",
    results: Optional[Mapping[str, object]] = None,
    environment: Optional[Mapping[str, object]] = None,
    timestamps: Optional[Mapping[str, object]] = None,
    code_revision: Optional[str] = None,
    known_limitations: Sequence[str] = (),
) -> Dict[str, object]:
    """Assemble an artifact dict; validation is a separate, explicit step.

    Missing results default to the honest ``not_measured`` block rather than to
    placeholder numbers.
    """
    artifact: Dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "phase": phase,
        "experiment_id": str(experiment_id),
        "dataset": dict(dataset),
        "modality": str(modality),
        "feature_configuration": dict(feature_configuration),
        "label_definition": dict(label_definition),
        "evaluation": dict(evaluation),
        "preprocessing": dict(preprocessing),
        "model": dict(model),
        "selection": dict(selection),
        "results": dict(
            results
            or {
                "status": "not_measured",
                "metrics": {},
                "confidence_intervals": {},
                "source": "",
                "note": "no predictions were produced in this phase",
            }
        ),
        "environment": dict(environment or collect_environment()),
        "timestamps": dict(
            timestamps or {"started_at": _now(), "finished_at": _now()}
        ),
        "code_revision": code_revision or git_revision(),
        "known_limitations": list(known_limitations),
        "clinical_validation": False,
        "serving": False,
    }
    return artifact


def validate_experiment_artifact(
    artifact: Mapping[str, object], *, raise_on_error: bool = True
) -> List[str]:
    """Validate the artifact contract; returns the list of problems found."""
    errors: List[str] = []

    if not isinstance(artifact, Mapping):
        errors.append("artifact must be a JSON object")
        if raise_on_error:
            raise ArtifactValidationError("; ".join(errors))
        return errors

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field!r}")

    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {ARTIFACT_SCHEMA_VERSION!r}; got "
            f"{artifact.get('schema_version')!r}"
        )

    exp_id = artifact.get("experiment_id")
    if not isinstance(exp_id, str) or not exp_id.strip():
        errors.append("experiment_id must be a non-empty string")

    if artifact.get("modality") in (None, ""):
        errors.append("modality must be recorded")

    if artifact.get("clinical_validation") is not False:
        errors.append(
            "clinical_validation must be false: this research prototype is not a "
            "clinical tool"
        )
    if artifact.get("serving") is not False:
        errors.append("serving must be false: no Phase 5 model is deployable")

    limitations = artifact.get("known_limitations")
    if not isinstance(limitations, (list, tuple)) or not limitations:
        errors.append("known_limitations must be a non-empty list")

    if not isinstance(artifact.get("code_revision"), str) or not artifact.get(
        "code_revision"
    ):
        errors.append("code_revision must be recorded (commit hash or 'unknown')")

    for field, required in _REQUIRED_NESTED.items():
        block = artifact.get(field)
        if not isinstance(block, Mapping):
            errors.append(f"{field} must be an object")
            continue
        for key in required:
            if key not in block:
                errors.append(f"{field}.{key} is required")

    # -- evaluation block ---------------------------------------------------
    evaluation = artifact.get("evaluation")
    if isinstance(evaluation, Mapping):
        unit = evaluation.get("unit")
        if unit not in _VALID_UNITS:
            errors.append(
                f"evaluation.unit must be one of {list(_VALID_UNITS)}; got {unit!r}"
            )
        for key in ("n_units", "n_positive"):
            value = evaluation.get(key)
            if not isinstance(value, int) or value < 0:
                errors.append(f"evaluation.{key} must be a non-negative integer")
        if (
            isinstance(evaluation.get("n_units"), int)
            and isinstance(evaluation.get("n_positive"), int)
            and evaluation["n_positive"] > evaluation["n_units"]
        ):
            errors.append("evaluation.n_positive cannot exceed evaluation.n_units")
        seeds = evaluation.get("seeds")
        if not isinstance(seeds, (list, tuple, Mapping)) or not seeds:
            errors.append("evaluation.seeds must record the seeds used")
        if evaluation.get("unit") == "sample" and "partition" in str(
            evaluation.get("split_strategy", "")
        ).lower():
            pass  # valid: AVDD uses a fixed partition

    # -- selection block ----------------------------------------------------
    selection = artifact.get("selection")
    if isinstance(selection, Mapping):
        for key in ("threshold_partition", "calibration_partition"):
            part = selection.get(key)
            if part in SELECTION_FORBIDDEN_PARTITIONS:
                errors.append(
                    f"selection.{key} = {part!r} is forbidden: thresholds and "
                    "calibrators are fitted on inner OOF data only"
                )

    # -- preprocessing block ------------------------------------------------
    preprocessing = artifact.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        fit_partition = preprocessing.get("fit_partition")
        if fit_partition in SELECTION_FORBIDDEN_PARTITIONS:
            errors.append(
                f"preprocessing.fit_partition = {fit_partition!r} is forbidden: "
                "fit on training data only"
            )

    # -- results block (no fabricated numbers) -----------------------------
    results = artifact.get("results")
    if isinstance(results, Mapping):
        status = results.get("status")
        metrics = results.get("metrics")
        intervals = results.get("confidence_intervals")
        if status not in _VALID_RESULT_STATUS:
            errors.append(
                f"results.status must be one of {list(_VALID_RESULT_STATUS)}; "
                f"got {status!r}"
            )
        if not isinstance(metrics, Mapping) or not isinstance(intervals, Mapping):
            errors.append("results.metrics and results.confidence_intervals must be objects")
            metrics, intervals = {}, {}
        if status == "not_measured":
            if metrics or intervals:
                errors.append(
                    "results.status is 'not_measured' but metrics/intervals are "
                    "present; remove the numbers or mark them measured"
                )
        else:
            if not metrics:
                errors.append(f"results.status {status!r} requires metrics")
            if not str(results.get("source", "")).strip():
                errors.append(
                    f"results.status {status!r} requires a named results.source "
                    "(the file or run that produced the numbers)"
                )
        for name, entry in metrics.items():
            if not isinstance(entry, Mapping):
                errors.append(f"results.metrics.{name} must be an object")
                continue
            if "value" not in entry:
                errors.append(f"results.metrics.{name} needs a 'value' key")
            if entry.get("value") is None and not entry.get("reason"):
                errors.append(
                    f"results.metrics.{name} is null without a reason; undefined "
                    "metrics must state why"
                )
            if entry.get("unit") not in _VALID_UNITS:
                errors.append(
                    f"results.metrics.{name}.unit must be one of {list(_VALID_UNITS)}"
                )

    # -- environment block --------------------------------------------------
    environment = artifact.get("environment")
    if isinstance(environment, Mapping):
        software = environment.get("software_versions")
        if not isinstance(software, Mapping) or not software:
            errors.append("environment.software_versions must be a non-empty object")
        elif not software.get("python"):
            errors.append("environment.software_versions must record the Python version")

    timestamps = artifact.get("timestamps")
    if isinstance(timestamps, Mapping):
        for key in ("started_at", "finished_at"):
            if not str(timestamps.get(key, "")).strip():
                errors.append(f"timestamps.{key} must be recorded")

    if artifact.get("dataset") is not None and isinstance(
        artifact.get("dataset"), Mapping
    ):
        dataset = artifact["dataset"]
        if not str(dataset.get("name", "")).strip():
            errors.append("dataset.name must be recorded")
        if not str(dataset.get("sha256", "")).strip():
            errors.append(
                "dataset.sha256 must be recorded so the artifact can be tied to "
                "an exact data snapshot"
            )

    if errors and raise_on_error:
        raise ArtifactValidationError(
            "artifact contract violation(s):\n- " + "\n- ".join(errors)
        )
    return errors


def _is_protected(path: Path) -> Optional[str]:
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover
        return None
    for protected in PROTECTED_OUTPUT_DIRS:
        target = (PROJECT_ROOT / protected).resolve()
        if resolved == target or target in resolved.parents:
            return protected
    return None


def write_experiment_artifact(
    artifact: Mapping[str, object],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate then write the artifact; never overwrites by default.

    Raises:
        ArtifactValidationError: the artifact violates the contract.
        FileExistsError: an artifact already exists and ``overwrite`` is False.
        PermissionError: the target is a frozen Phase 2 / model directory.
    """
    validate_experiment_artifact(artifact)
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target

    protected = _is_protected(target)
    if protected is not None:
        raise PermissionError(
            f"refusing to write into {protected!r}: Phase 2 results and model "
            "checkpoints are immutable for the evaluation harness"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists; experiment artifacts are immutable "
            "(pass overwrite=True only for a deliberate re-run)"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return target
