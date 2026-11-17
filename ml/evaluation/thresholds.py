"""
Threshold selection for Phase 5 (Phase 5B).

Two rules govern every threshold in this project:

1. A threshold is chosen on **inner** out-of-fold predictions only. The
   provenance passed to :func:`select_threshold` is checked, and an outer
   evaluation or locked-test provenance is rejected outright.
2. The primary policy is **balanced accuracy**, pre-registered in
   ``docs/phase_5_statistical_evaluation_plan.md``. The commonly used 0.5 cut is
   recorded separately as a *reference* result so a reader can see both.

No threshold is fitted to real data in Phase 5B; these functions are exercised
with synthetic predictions in ``tests/test_phase5_splits.py``.

Exports:
    ThresholdSelection, ThresholdError
    select_threshold, reference_threshold, apply_threshold
    POLICY_* constants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ml.evaluation.metrics import _counts, _validate_inputs
from ml.evaluation.provenance import (
    OPERATION_THRESHOLD_SELECTION,
    Provenance,
    ProvenanceError,
)

POLICY_BALANCED_ACCURACY = "balanced_accuracy"
POLICY_YOUDEN_J = "youden_j"
POLICY_SENSITIVITY_AT_SPECIFICITY = "sensitivity_at_specificity"
POLICY_FIXED = "fixed"

#: Pre-registered primary policy.
PRIMARY_POLICY = POLICY_BALANCED_ACCURACY
#: Pre-declared reference cut, reported alongside (never tuned).
REFERENCE_THRESHOLD = 0.5
#: Default specificity target for the alternative operating-point policy.
DEFAULT_TARGET_SPECIFICITY = 0.8

SELECTED_POLICIES: Tuple[str, ...] = (
    POLICY_BALANCED_ACCURACY,
    POLICY_YOUDEN_J,
    POLICY_SENSITIVITY_AT_SPECIFICITY,
)
ALL_POLICIES: Tuple[str, ...] = SELECTED_POLICIES + (POLICY_FIXED,)

#: Scores within this distance are treated as tied.
_TIE_TOL = 1e-12


class ThresholdError(RuntimeError):
    """A threshold cannot be selected under the requested policy."""


@dataclass(frozen=True)
class ThresholdSelection:
    """A threshold with the rule, data partition and evidence behind it."""

    threshold: float
    policy: str
    data_partition: str
    provenance: Optional[Provenance] = None
    score: Optional[float] = None
    achieved: Dict[str, float] = field(default_factory=dict)
    n_units: int = 0
    n_positive: int = 0
    target_specificity: Optional[float] = None
    n_candidates: int = 0
    selected_from: str = ""
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.policy not in ALL_POLICIES:
            raise ValueError(f"unknown threshold policy {self.policy!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    @property
    def is_selected_from_data(self) -> bool:
        return self.policy != POLICY_FIXED

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "threshold": round(float(self.threshold), 6),
            "policy": self.policy,
            "selected_from_data": self.is_selected_from_data,
            "data_partition": self.data_partition,
            "n_units": self.n_units,
            "n_positive": self.n_positive,
            "score": None if self.score is None else round(float(self.score), 6),
            "achieved": {k: round(float(v), 6) for k, v in self.achieved.items()},
            "n_candidates": self.n_candidates,
            "selected_from": self.selected_from,
            "notes": list(self.notes),
        }
        if self.target_specificity is not None:
            out["target_specificity"] = float(self.target_specificity)
        if self.provenance is not None:
            out["provenance"] = self.provenance.to_dict()
        return out


def _criterion(
    policy: str,
    sens: float,
    spec: float,
    *,
    target_specificity: Optional[float],
) -> Optional[float]:
    if policy == POLICY_BALANCED_ACCURACY:
        return (sens + spec) / 2.0
    if policy == POLICY_YOUDEN_J:
        return sens + spec - 1.0
    if policy == POLICY_SENSITIVITY_AT_SPECIFICITY:
        if spec >= float(target_specificity) - 1e-12:
            return sens
        return None
    raise ThresholdError(f"policy {policy!r} is not a data-selected policy")


def select_threshold(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    *,
    provenance: Provenance,
    policy: str = PRIMARY_POLICY,
    target_specificity: float = DEFAULT_TARGET_SPECIFICITY,
    unit: str = "participant",
) -> ThresholdSelection:
    """Choose a decision threshold from labelled predictions.

    Args:
        y_true: binary labels of the data the threshold is chosen on.
        y_prob: predicted probabilities of the positive class.
        provenance: where those predictions came from. Must be fit-safe: outer
            evaluation and locked-test provenance raise
            :class:`~ml.evaluation.provenance.ProvenanceError`.
        policy: one of ``balanced_accuracy`` (primary), ``youden_j``,
            ``sensitivity_at_specificity``.
        target_specificity: required specificity for the last policy.
        unit: evaluation unit of the supplied rows (recorded, not inferred).

    Returns:
        :class:`ThresholdSelection` with the chosen threshold, achieved
        sensitivity/specificity, candidate count and full provenance.

    Raises:
        ProvenanceError: the predictions must not be used for selection.
        ThresholdError: no candidate threshold satisfies the policy.
    """
    if policy not in SELECTED_POLICIES:
        raise ThresholdError(
            f"select_threshold cannot use policy {policy!r}; expected one of "
            f"{list(SELECTED_POLICIES)}"
        )
    if provenance is None:
        raise ProvenanceError(
            "threshold selection requires provenance: state which data partition "
            "the predictions came from"
        )
    provenance.assert_fit_safe(OPERATION_THRESHOLD_SELECTION)

    y, p = _validate_inputs(y_true, y_prob, unit, f"threshold_selection[{policy}]")
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ThresholdError(
            f"cannot select a threshold on single-class data "
            f"(pos={n_pos}, neg={n_neg})"
        )

    candidates = np.unique(p)[::-1]
    best: Optional[Tuple[float, float, float, float]] = None  # (criterion, t, sens, spec)
    for t in candidates:
        cm = _counts(y, (p >= t).astype(np.int64))
        sens = cm["tp"] / n_pos
        spec = cm["tn"] / n_neg
        score = _criterion(
            policy, sens, spec, target_specificity=target_specificity
        )
        if score is None:
            continue
        chosen = (score, float(t), sens, spec)
        if best is None:
            best = chosen
            continue
        if score > best[0] + _TIE_TOL:
            best = chosen
        elif abs(score - best[0]) <= _TIE_TOL:
            # Deterministic tie-break, applied identically for every policy:
            #   1. higher sensitivity (fewer missed cases),
            #   2. higher specificity,
            #   3. threshold closest to 0.5,
            #   4. lower threshold.
            if sens > best[2] + _TIE_TOL:
                best = chosen
            elif abs(sens - best[2]) <= _TIE_TOL:
                if spec > best[3] + _TIE_TOL:
                    best = chosen
                elif abs(spec - best[3]) <= _TIE_TOL:
                    better_distance = abs(chosen[1] - 0.5) < abs(best[1] - 0.5) - _TIE_TOL
                    if better_distance or (
                        abs(abs(chosen[1] - 0.5) - abs(best[1] - 0.5)) <= _TIE_TOL
                        and chosen[1] < best[1]
                    ):
                        best = chosen

    if best is None:
        raise ThresholdError(
            f"policy {policy!r} cannot be satisfied: no candidate threshold "
            f"reaches specificity >= {target_specificity}"
        )

    score, threshold, sens, spec = best
    notes = [
        "threshold selected from the supplied partition only; it is not applied "
        "to any other partition",
    ]
    if policy == POLICY_SENSITIVITY_AT_SPECIFICITY:
        notes.append(f"target specificity {target_specificity}")
    if provenance.partition != "inner_validation":
        notes.append(
            f"partition {provenance.partition!r} is fit-safe but is not the "
            "pre-registered inner-OOF partition"
        )
    return ThresholdSelection(
        threshold=threshold,
        policy=policy,
        data_partition=provenance.partition,
        provenance=provenance,
        score=score,
        achieved={"sensitivity": sens, "specificity": spec},
        n_units=provenance.n_units,
        n_positive=provenance.n_positive,
        target_specificity=(
            target_specificity
            if policy == POLICY_SENSITIVITY_AT_SPECIFICITY
            else None
        ),
        n_candidates=int(candidates.size),
        selected_from=" ".join(
            [
                f"inner OOF probabilities ({provenance.unit}-level, "
                f"n={provenance.n_units}, positives={provenance.n_positive})"
            ]
        ),
        notes=tuple(notes),
    )


def reference_threshold(value: float = REFERENCE_THRESHOLD) -> ThresholdSelection:
    """The pre-declared 0.5 cut, recorded as a separate reference result.

    No data is consulted, so no provenance is attached and the value is not a
    tuned quantity. Report it next to the selected threshold, never instead of it.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError("reference threshold must be in [0, 1]")
    return ThresholdSelection(
        threshold=float(value),
        policy=POLICY_FIXED,
        data_partition="none",
        notes=(
            "pre-declared reference cut-off, not selected or tuned on any data",
        ),
    )


def apply_threshold(
    selection: ThresholdSelection, y_prob: Iterable[float]
) -> np.ndarray:
    """Apply a selection with ``prob >= threshold`` semantics (class 1 positive)."""
    p = np.asarray(list(y_prob), dtype=np.float64)
    if not np.isfinite(p).all():
        raise ThresholdError("probabilities must be finite")
    if (p < 0).any() or (p > 1).any():
        raise ThresholdError("probabilities must lie in [0, 1]")
    return (p >= selection.threshold).astype(np.int64)
