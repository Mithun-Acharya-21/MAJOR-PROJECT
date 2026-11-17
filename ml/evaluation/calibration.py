"""
Calibration interfaces for Phase 5 (Phase 5B).

Phase 5B defines the calibation *contract* and enforces where calibration data
may come from. It deliberately does **not** fit a calibrator: fitting belongs to
Phase 5C, on inner out-of-fold predictions, and any fit performed here would be
an unreported modelling step.

The contract:

* a calibrator maps raw probabilities to calibrated probabilities (never labels,
  never a threshold);
* it is fitted on inner-OOF predictions and applied to held-out probabilities;
* its fit provenance is checked with the same guard as threshold selection;
* the method and the fit partition travel with every use.

Exports:
    CalibrationPlan, CalibrationNotImplementedError, CalibrationProvenanceError
    CALIBRATION_METHODS, plan_calibration, fit_calibrator, apply_calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ml.evaluation.provenance import (
    OPERATION_CALIBRATION_FIT,
    Provenance,
    ProvenanceError,
)

METHOD_PLATT = "platt"
METHOD_ISOTONIC = "isotonic"
METHOD_NONE = "none"
CALIBRATION_METHODS = (METHOD_PLATT, METHOD_ISOTONIC, METHOD_NONE)
PRIMARY_METHOD = METHOD_PLATT


class CalibrationError(RuntimeError):
    """Calibration contract violated."""


class CalibrationNotImplementedError(CalibrationError, NotImplementedError):
    """Fitting is out of scope for Phase 5B."""


@runtime_checkable
class Calibrator(Protocol):
    """Probability recalibrator (implemented in Phase 5C)."""

    method: str

    def fit(self, y_prob: Sequence[float], y_true: Sequence[int]) -> "Calibrator":
        ...

    def predict(self, y_prob: Sequence[float]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class CalibrationPlan:
    """What will be calibrated, with what method, on which partition."""

    method: str
    provenance: Provenance
    target: str = "probability_of_positive_class"
    fitted: bool = False
    notes: tuple = ()

    def __post_init__(self) -> None:
        if self.method not in CALIBRATION_METHODS:
            raise CalibrationError(
                f"unknown calibration method {self.method!r}; expected one of "
                f"{list(CALIBRATION_METHODS)}"
            )

    def assert_fit_safe(self, operation: str = OPERATION_CALIBRATION_FIT) -> None:
        """Re-verify at the point of fitting that the data may be fitted on."""
        self.provenance.assert_fit_safe(operation)

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "target": self.target,
            "fitted": self.fitted,
            "fit_partition": self.provenance.partition,
            "n_fit_units": self.provenance.n_units,
            "n_fit_positive": self.provenance.n_positive,
            "provenance": self.provenance.to_dict(),
            "notes": list(self.notes),
        }


def plan_calibration(
    method: str,
    *,
    provenance: Provenance,
    target: str = "probability_of_positive_class",
    notes: tuple = (),
) -> CalibrationPlan:
    """Declare a calibration method for a fit partition.

    Raises:
        ProvenanceError: the fit data includes outer evaluation or locked test.
        CalibrationError: unknown method.
    """
    if provenance is None:
        raise ProvenanceError(
            "calibration requires provenance: state which partition the "
            "calibration data came from"
        )
    provenance.assert_fit_safe(OPERATION_CALIBRATION_FIT)
    base_notes = (
        "platt = sigmoid fit on the logit of the raw probability; "
        "isotonic = monotone step fit; none = raw probabilities, reported as such",
    )
    return CalibrationPlan(
        method=method,
        provenance=provenance,
        target=target,
        fitted=False,
        notes=tuple(base_notes) + tuple(notes),
    )


def fit_calibrator(plan: CalibrationPlan, *_: object, **__: object) -> Calibrator:
    """Refuse to fit in Phase 5B.

    Phase 5B is an evaluation-harness phase: no calibration curve is estimated
    from real data, so that later phases cannot report a calibration fitted on a
    partition nobody recorded.
    """
    plan.assert_fit_safe()
    raise CalibrationNotImplementedError(
        f"calibration fitting ({plan.method!r}) is not implemented in Phase 5B. "
        f"Calibrators are fitted in Phase 5C on inner-OOF predictions with "
        f"provenance partition 'inner_validation' and the resulting parameters "
        f"are recorded in the experiment artifact. Plan provenance: "
        f"partition={plan.provenance.partition!r}, "
        f"n={plan.provenance.n_units}, positives={plan.provenance.n_positive}."
    )


def apply_calibration(
    calibrator: Calibrator,
    y_prob: Sequence[float],
    *,
    plan: Optional[CalibrationPlan] = None,
) -> np.ndarray:
    """Apply a fitted calibrator, refusing unfitted or out-of-range output."""
    if plan is not None and not plan.fitted:
        raise CalibrationError(
            "calibration plan is not fitted; call fit_calibrator in the phase "
            "that is authorised to fit and set plan.fitted=True"
        )
    p = np.asarray(list(y_prob), dtype=np.float64)
    if p.size == 0:
        raise CalibrationError("no probabilities supplied")
    if not np.isfinite(p).all():
        raise CalibrationError("probabilities must be finite")
    out = np.asarray(calibrator.predict(p), dtype=np.float64).reshape(-1)
    if out.shape != p.shape:
        raise CalibrationError(
            f"calibrator returned {out.size} values for {p.size} inputs"
        )
    if not np.isfinite(out).all():
        raise CalibrationError("calibrator produced non-finite probabilities")
    if (out < 0).any() or (out > 1).any():
        raise CalibrationError("calibrator produced probabilities outside [0, 1]")
    return out
