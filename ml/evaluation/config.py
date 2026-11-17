"""
Typed access to the Phase 5 evaluation configuration (Phase 5B).

The YAML file at ``ml/configs/phase5/evaluation.yaml`` is the single declared
source for split seeds, fold counts, bootstrap defaults, metric set, guard rails
and the threshold/calibration policy. Loading it through this module means the
harness and its tests cannot drift from the written protocol.

Exports:
    DEFAULT_CONFIG_PATH, EvaluationConfigError, EvaluationConfig,
    load_evaluation_config
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml" / "configs" / "phase5" / "evaluation.yaml"

REQUIRED_SECTIONS = (
    "phase",
    "dataset",
    "evaluation_unit",
    "seeds",
    "outer_cv",
    "inner_cv",
    "bootstrap",
    "threshold",
    "calibration",
    "metrics",
    "guards",
    "artifact",
    "locked_test",
)


class EvaluationConfigError(RuntimeError):
    """The evaluation configuration is missing or inconsistent."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Declared Phase 5 evaluation settings."""

    raw: Dict[str, Any] = field(default_factory=dict)
    path: Path = DEFAULT_CONFIG_PATH

    # -- convenient typed accessors ----------------------------------------

    @property
    def phase(self) -> str:
        return str(self.raw["phase"])

    @property
    def dataset(self) -> str:
        return str(self.raw["dataset"])

    @property
    def evaluation_unit(self) -> str:
        return str(self.raw["evaluation_unit"])

    @property
    def seeds(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in self.raw["seeds"])

    @property
    def outer_n_splits(self) -> int:
        return int(self.raw["outer_cv"]["n_splits"])

    @property
    def inner_n_splits(self) -> int:
        return int(self.raw["inner_cv"]["n_splits"])

    @property
    def splitter(self) -> str:
        return str(self.raw["outer_cv"]["splitter"])

    @property
    def threshold_policy(self) -> str:
        return str(self.raw["threshold"]["primary_policy"])

    @property
    def reference_threshold(self) -> float:
        return float(self.raw["threshold"]["reference_value"])

    @property
    def calibration_method(self) -> str:
        return str(self.raw["calibration"]["primary"])

    @property
    def calibration_fit_in_this_phase(self) -> bool:
        return bool(self.raw["calibration"]["fit_in_phase_5b"])

    @property
    def bootstrap(self) -> Dict[str, Any]:
        return dict(self.raw["bootstrap"])

    @property
    def guards(self) -> Dict[str, Any]:
        return dict(self.raw["guards"])

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self.raw["metrics"])

    @property
    def locked_test(self) -> Dict[str, Any]:
        return dict(self.raw["locked_test"])

    @property
    def artifact(self) -> Dict[str, Any]:
        return dict(self.raw["artifact"])

    def to_dict(self) -> Dict[str, Any]:
        return {"path": str(self.path), "config": json.loads(json.dumps(self.raw))}


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise EvaluationConfigError(
            "PyYAML is required to read the evaluation configuration"
        ) from exc
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationConfigError(f"evaluation config not found: {path}") from exc
    if not isinstance(loaded, dict):
        raise EvaluationConfigError(f"{path} must contain a YAML mapping")
    return loaded


def _validate(raw: Mapping[str, Any], path: Path) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise EvaluationConfigError(
            f"{path} is missing section(s): {missing}"
        )
    if not raw["seeds"]:
        raise EvaluationConfigError("seeds must not be empty")
    if int(raw["outer_cv"]["n_splits"]) < 2:
        raise EvaluationConfigError("outer_cv.n_splits must be >= 2")
    if int(raw["inner_cv"]["n_splits"]) < 2:
        raise EvaluationConfigError("inner_cv.n_splits must be >= 2")
    if str(raw["outer_cv"]["splitter"]) != "StratifiedGroupKFold":
        raise EvaluationConfigError(
            "outer_cv.splitter must be StratifiedGroupKFold; an ungrouped "
            "splitter would leak participant sessions"
        )
    if str(raw["evaluation_unit"]) != "participant":
        raise EvaluationConfigError(
            "EATD evaluation unit must be participant; sessions are not independent"
        )
    if bool(raw["calibration"].get("fit_in_phase_5b", False)):
        raise EvaluationConfigError(
            "Phase 5B defines calibration interfaces only; "
            "calibration.fit_in_phase_5b must be false"
        )
    if bool(raw["locked_test"].get("read_in_phase_5b", True)):
        raise EvaluationConfigError(
            "locked_test.read_in_phase_5b must be false"
        )


def load_evaluation_config(path: Path | str | None = None) -> EvaluationConfig:
    """Load and validate the Phase 5 evaluation configuration."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    raw = _read_yaml(resolved)
    _validate(raw, resolved)
    return EvaluationConfig(raw=raw, path=resolved)
