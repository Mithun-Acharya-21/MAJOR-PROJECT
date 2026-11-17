"""
Provenance and fit-safety guards for Phase 5 evaluation (Phase 5B).

Every Phase 5 selection operation — threshold choice, calibration fitting,
preprocessing fitting, feature selection — must say which data partition it was
allowed to see, and must be refused if that partition includes an outer
evaluation fold or the locked test set.

This module is deliberately dependency-free (no numpy, no sklearn) so it can be
imported by every other harness module without import cycles.

Partition vocabulary
--------------------
======================  =====================================================
partition               meaning
======================  =====================================================
``development``         the full EATD development pool (train + validation
                        legacy split, 138 participants). Read-only surveys
                        and dataset-level audits.
``outer_train``         the training part of one outer fold.
``outer_evaluation``    the held-out part of one outer fold. **Forbidden for
                        any fitting operation.**
``inner_train``         the training part of one inner fold.
``inner_validation``    the held-out part of one inner fold; the partition
                        that threshold selection and calibration fitting are
                        allowed (and required) to use.
``locked_test``         the 24-participant locked EATD test set. **Forbidden
                        for every operation in Phase 5B.**
======================  =====================================================

Exports:
    PARTITION_* constants, SELECTION_PARTITIONS, FORBIDDEN_FOR_SELECTION
    Provenance, ProvenanceError, ProvenanceLog
    build_provenance, assert_fit_on_training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

PARTITION_DEVELOPMENT = "development"
PARTITION_OUTER_TRAIN = "outer_train"
PARTITION_OUTER_EVALUATION = "outer_evaluation"
PARTITION_INNER_TRAIN = "inner_train"
PARTITION_INNER_VALIDATION = "inner_validation"
PARTITION_LOCKED_TEST = "locked_test"

ALL_PARTITIONS = frozenset(
    {
        PARTITION_DEVELOPMENT,
        PARTITION_OUTER_TRAIN,
        PARTITION_OUTER_EVALUATION,
        PARTITION_INNER_TRAIN,
        PARTITION_INNER_VALIDATION,
        PARTITION_LOCKED_TEST,
    }
)

#: Partitions a fitting/selection operation is allowed to read.
SELECTION_PARTITIONS = frozenset(
    {
        PARTITION_DEVELOPMENT,
        PARTITION_OUTER_TRAIN,
        PARTITION_INNER_TRAIN,
        PARTITION_INNER_VALIDATION,
    }
)

#: Partitions that must never reach a fitting/selection operation.
FORBIDDEN_FOR_SELECTION = frozenset(
    {PARTITION_OUTER_EVALUATION, PARTITION_LOCKED_TEST}
)

VALID_UNITS = frozenset({"participant", "session", "sample", "group"})

# Operation labels used across the harness. They only appear in error messages
# and in the provenance record, so they are plain strings.
OPERATION_THRESHOLD_SELECTION = "threshold_selection"
OPERATION_CALIBRATION_FIT = "calibration_fit"
OPERATION_PREPROCESSING_FIT = "preprocessing_fit"
OPERATION_FEATURE_SELECTION = "feature_selection"
OPERATION_HYPERPARAMETER_SELECTION = "hyperparameter_selection"
OPERATION_MODEL_FIT = "model_fit"
OPERATION_OOF_PREDICTION = "oof_prediction"


class ProvenanceError(RuntimeError):
    """Raised when an operation is asked to use forbidden data."""


@dataclass(frozen=True)
class Provenance:
    """What data a Phase 5 operation used, and whether it was allowed to.

    ``partition`` is the partition the operation's *own* inputs came from.
    ``contributing_partitions`` lists any additional partitions that shaped the
    inputs (e.g. an inner-OOF prediction file was produced by a model fitted on
    ``inner_train`` and applied to ``inner_validation``). The fit-safety check
    inspects both.
    """

    partition: str
    operation: str
    dataset: str
    unit: str
    n_units: int
    n_positive: int
    n_groups: Optional[int] = None
    seeds: Tuple[int, ...] = ()
    source: str = ""
    contributing_partitions: Tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.partition not in ALL_PARTITIONS:
            raise ValueError(
                f"unknown partition {self.partition!r}; expected one of "
                f"{sorted(ALL_PARTITIONS)}"
            )
        unknown = [p for p in self.contributing_partitions if p not in ALL_PARTITIONS]
        if unknown:
            raise ValueError(f"unknown contributing partition(s) {unknown!r}")
        if self.unit not in VALID_UNITS:
            raise ValueError(f"unknown evaluation unit {self.unit!r}")
        if self.n_units < 0 or self.n_positive < 0:
            raise ValueError("n_units and n_positive must be non-negative")
        if self.n_positive > self.n_units:
            raise ValueError("n_positive cannot exceed n_units")
        if self.n_groups is not None and self.n_groups < 0:
            raise ValueError("n_groups must be non-negative")
        if not self.operation:
            raise ValueError("operation must be a non-empty label")

    # -- fit safety ---------------------------------------------------------

    @property
    def partitions(self) -> Tuple[str, ...]:
        """``partition`` plus every contributing partition, de-duplicated."""
        seen: list[str] = []
        for p in (self.partition, *self.contributing_partitions):
            if p not in seen:
                seen.append(p)
        return tuple(seen)

    @property
    def forbidden_partitions(self) -> Tuple[str, ...]:
        return tuple(p for p in self.partitions if p in FORBIDDEN_FOR_SELECTION)

    @property
    def is_fit_safe(self) -> bool:
        return not self.forbidden_partitions

    def assert_fit_safe(self, operation: Optional[str] = None) -> "Provenance":
        """Raise :class:`ProvenanceError` if these inputs must not be fitted on.

        Args:
            operation: label of the operation being attempted. Defaults to the
                operation recorded on this provenance.

        Returns:
            ``self`` so callers can chain.
        """
        op = operation or self.operation
        if self.forbidden_partitions:
            raise ProvenanceError(
                f"refusing {op!r}: input provenance includes forbidden "
                f"partition(s) {list(self.forbidden_partitions)} "
                f"(all partitions: {list(self.partitions)}). Selection, "
                f"threshold, calibration, preprocessing and feature-selection "
                f"operations may only use {sorted(SELECTION_PARTITIONS)}."
            )
        return self

    def derive(
        self,
        *,
        partition: str,
        operation: str,
        n_units: Optional[int] = None,
        n_positive: Optional[int] = None,
        n_groups: Optional[int] = None,
        source: str = "",
        details: Optional[Mapping[str, object]] = None,
    ) -> "Provenance":
        """Return a new provenance that inherits this one's contributing parts.

        Used when, for example, threshold selection consumes an inner-OOF
        prediction object: the derived record keeps both the prediction's
        partition and the selection's own partition.
        """
        contributing = tuple(
            p for p in (*self.partitions, partition) if p != partition
        )
        return build_provenance(
            partition=partition,
            operation=operation,
            dataset=self.dataset,
            unit=self.unit,
            n_units=self.n_units if n_units is None else n_units,
            n_positive=self.n_positive if n_positive is None else n_positive,
            n_groups=self.n_groups if n_groups is None else n_groups,
            seeds=self.seeds,
            source=source or self.source,
            contributing_partitions=contributing,
            details={**dict(self.details), **dict(details or {})},
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "partition": self.partition,
            "contributing_partitions": list(self.contributing_partitions),
            "partitions": list(self.partitions),
            "operation": self.operation,
            "dataset": self.dataset,
            "unit": self.unit,
            "n_units": self.n_units,
            "n_positive": self.n_positive,
            "n_groups": self.n_groups,
            "seeds": list(self.seeds),
            "source": self.source,
            "details": dict(self.details),
            "fit_safe": self.is_fit_safe,
        }


def build_provenance(
    *,
    partition: str,
    operation: str,
    dataset: str,
    unit: str,
    labels: Optional[Sequence[float]] = None,
    n_units: Optional[int] = None,
    n_positive: Optional[int] = None,
    n_groups: Optional[int] = None,
    seeds: Iterable[int] = (),
    source: str = "",
    contributing_partitions: Iterable[str] = (),
    details: Optional[Mapping[str, object]] = None,
) -> Provenance:
    """Build a :class:`Provenance` record.

    If ``labels`` is supplied, ``n_units`` and ``n_positive`` are derived from
    it (and labels are validated as binary); otherwise both must be given.
    """
    if labels is not None:
        vals = [int(v) for v in labels]
        bad = [v for v in vals if v not in (0, 1)]
        if bad:
            raise ValueError(f"labels must be binary 0/1; got {sorted(set(bad))}")
        if n_units is not None and int(n_units) != len(vals):
            raise ValueError("n_units contradicts len(labels)")
        if n_positive is not None and int(n_positive) != int(sum(vals)):
            raise ValueError("n_positive contradicts sum(labels)")
        n_units = len(vals)
        n_positive = int(sum(vals))
    if n_units is None or n_positive is None:
        raise ValueError("provide labels, or both n_units and n_positive")

    return Provenance(
        partition=partition,
        operation=operation,
        dataset=dataset,
        unit=unit,
        n_units=int(n_units),
        n_positive=int(n_positive),
        n_groups=None if n_groups is None else int(n_groups),
        seeds=tuple(int(s) for s in seeds),
        source=source,
        contributing_partitions=tuple(contributing_partitions),
        details=dict(details or {}),
    )


def assert_fit_on_training(
    provenance: Provenance,
    operation: str = OPERATION_PREPROCESSING_FIT,
) -> Provenance:
    """Guard for anything fitted (scaler, vectorizer, PCA, feature selector).

    Phase 5B does not fit these on real data; this function exists so later
    phases — and the leakage probes — call one audited entry point instead of
    each re-implementing the check.
    """
    return provenance.assert_fit_safe(operation)


@dataclass
class ProvenanceLog:
    """Ordered record of the provenance behind every decision in a run."""

    records: list = field(default_factory=list)

    def record(self, provenance: Provenance, *, operation: Optional[str] = None) -> Provenance:
        entry = provenance.to_dict()
        if operation is not None:
            entry["recorded_operation"] = operation
        self.records.append(entry)
        return provenance

    def for_operation(self, operation: str) -> list:
        return [r for r in self.records if r.get("operation") == operation]

    def to_dict(self) -> Dict[str, object]:
        return {"n_records": len(self.records), "records": list(self.records)}
