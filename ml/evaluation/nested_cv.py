"""
Nested-CV out-of-fold prediction runner for Phase 5 (Phase 5B).

This module runs the *plumbing* of repeated nested CV, not the modelling. The
caller supplies a prediction callback; the runner only decides which unit ids
may be trained on, which must be predicted, and what provenance the resulting
predictions carry.

Callback contract
-----------------
``predict_fn(request) -> Sequence[float]``

* ``request`` is a :class:`PredictionRequest` exposing ``train_unit_ids``,
  ``eval_unit_ids``, their participant groups, ``stage``, ``repeat_seed``,
  ``fold`` and ``n_splits``.
* The callback must fit on ``request.train_unit_ids`` only — never on
  ``eval_unit_ids``, never on anything outside the request.
* It returns one probability of the positive class per eval unit, in the order
  of ``request.eval_unit_ids``. Length, finiteness and range are all checked;
  nothing is clipped or reordered silently.

Because the outer evaluation partition is labelled ``outer_evaluation`` in the
returned provenance, threshold selection and calibration fitting on outer OOF
predictions are refused automatically. Inner OOF predictions are labelled
``inner_validation`` and are the only predictions the selection rules accept.

Phase 5B never supplies a real callback; the tests use synthetic predictors.

Exports:
    PredictionRequest, OutOfFoldPredictions, PredictionContractError
    generate_oof_predictions, generate_inner_oof
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml.evaluation import metrics as M
from ml.evaluation.provenance import (
    OPERATION_OOF_PREDICTION,
    PARTITION_INNER_TRAIN,
    PARTITION_INNER_VALIDATION,
    PARTITION_OUTER_EVALUATION,
    PARTITION_OUTER_TRAIN,
    Provenance,
    ProvenanceError,
    build_provenance,
)
from ml.evaluation.splits import (
    INNER_FOLDS,
    Fold,
    Repetition,
    Unit,
    assert_repetition_integrity,
    group_labels,
    inner_group_folds,
    unit_index,
)

DATASET_DEFAULT = "eatd"


class PredictionContractError(RuntimeError):
    """The prediction callback broke the OOF contract."""


@dataclass(frozen=True)
class PredictionRequest:
    """What one fold of the CV asks the callback to predict."""

    stage: str
    repeat_seed: int
    fold: int
    n_splits: int
    train_unit_ids: Tuple[str, ...]
    eval_unit_ids: Tuple[str, ...]
    train_group_ids: Tuple[str, ...]
    eval_group_ids: Tuple[str, ...]
    outer_fold: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage": self.stage,
            "repeat_seed": self.repeat_seed,
            "fold": self.fold,
            "n_splits": self.n_splits,
            "outer_fold": self.outer_fold,
            "n_train_units": len(self.train_unit_ids),
            "n_eval_units": len(self.eval_unit_ids),
            "n_train_groups": len(self.train_group_ids),
            "n_eval_groups": len(self.eval_group_ids),
        }


@dataclass(frozen=True)
class OutOfFoldPredictions:
    """Assembled OOF predictions with labels, units and provenance."""

    y_true: np.ndarray
    y_prob: np.ndarray
    unit_ids: Tuple[str, ...]
    group_ids: Tuple[str, ...]
    fold_ids: Tuple[int, ...]
    stage: str
    repeat_seed: int
    provenance: Provenance
    dataset: str
    n_folds: int

    def __post_init__(self) -> None:
        if self.y_true.shape != self.y_prob.shape:
            raise PredictionContractError("labels and probabilities are misaligned")
        if len(self.unit_ids) != self.y_true.size:
            raise PredictionContractError("unit ids do not match predictions")
        if len(self.group_ids) != self.y_true.size:
            raise PredictionContractError("group ids do not match predictions")
        if len(self.fold_ids) != self.y_true.size:
            raise PredictionContractError("fold ids do not match predictions")

    def to_dict(self) -> Dict[str, object]:
        """Summary (no row-level predictions) for reports and artifacts."""
        per_fold = []
        for fold in sorted(set(self.fold_ids)):
            mask = np.asarray(self.fold_ids) == fold
            n = int(mask.sum())
            n_pos = int(self.y_true[mask].sum())
            per_fold.append(
                {
                    "fold": int(fold),
                    "n_units": n,
                    "n_positive": n_pos,
                    "n_negative": n - n_pos,
                }
            )
        return {
            "stage": self.stage,
            "repeat_seed": self.repeat_seed,
            "n_folds": self.n_folds,
            "n_rows": int(self.y_true.size),
            "n_positive_rows": int(self.y_true.sum()),
            "n_groups": len(set(self.group_ids)),
            "per_fold": per_fold,
            "provenance": self.provenance.to_dict(),
        }

    def aggregate_to_participants(self, *, aggregation: str = "mean") -> Dict[str, object]:
        """Collapse sessions to one row per participant for participant-level use.

        The returned provenance reflects the change of unit, so a threshold
        selected on the aggregated data still records ``participant``.
        """
        from ml.evaluation.bootstrap import aggregate_by_participant

        aggregated = aggregate_by_participant(
            self.y_true, self.y_prob, self.group_ids, aggregation=aggregation
        )
        provenance = build_provenance(
            partition=self.provenance.partition,
            operation=OPERATION_OOF_PREDICTION,
            dataset=self.dataset,
            unit="participant",
            labels=aggregated["labels"],
            n_groups=aggregated["n_units"],
            seeds=self.provenance.seeds,
            source=self.provenance.source,
            contributing_partitions=self.provenance.contributing_partitions,
            details={
                "aggregation": aggregation,
                "from_unit": self.provenance.unit,
                "n_input_rows": int(self.y_true.size),
            },
        )
        aggregated["provenance"] = provenance
        return aggregated

    def compute_metrics(
        self,
        *,
        threshold: float = 0.5,
        unit: Optional[str] = None,
        **kwargs: object,
    ) -> Dict[str, object]:
        """Metrics on these predictions only; the unit is recorded, not guessed."""
        return M.compute_evaluation_metrics(
            self.y_true,
            self.y_prob,
            unit=unit or self.provenance.unit,
            threshold=threshold,
            **kwargs,  # type: ignore[arg-type]
        )


def _validate_partition(stage: str, partition: str) -> None:
    expected = (
        PARTITION_OUTER_EVALUATION if stage == "outer" else PARTITION_INNER_VALIDATION
    )
    if partition != expected:
        raise PredictionContractError(
            f"stage {stage!r} OOF predictions must carry partition {expected!r}; "
            f"got {partition!r}. Mislabeling would let outer-evaluation "
            "predictions pass a fit-safety check."
        )


def generate_oof_predictions(
    units: Mapping[str, Unit],
    repetition: Repetition,
    predict_fn: Callable[[PredictionRequest], Sequence[float]],
    *,
    partition: str,
    dataset: str = DATASET_DEFAULT,
    unit: str = "session",
    firewall: Optional[object] = None,
    operation: str = OPERATION_OOF_PREDICTION,
    source: str = "prediction_callback",
) -> OutOfFoldPredictions:
    """Assemble out-of-fold predictions from a caller-supplied callback.

    Args:
        units: unit_id → Unit for the whole development pool (labels + groups).
        repetition: folds produced by :mod:`ml.evaluation.splits`.
        predict_fn: the modelling callback (see module docstring).
        partition: ``outer_evaluation`` for outer folds, ``inner_validation``
            for inner folds. Enforced against ``repetition.stage``.
        dataset: dataset name recorded in the provenance.
        unit: evaluation unit of the returned rows (``session`` for EATD before
            participant aggregation).
        firewall: optional :class:`ml.evaluation.firewall.LockedTestFirewall`.
        operation, source: recorded on the provenance.

    Returns:
        :class:`OutOfFoldPredictions` covering every evaluated unit exactly once.

    Raises:
        PredictionContractError: callback output violates the contract.
        LockedTestAccessError: a fold touches a locked-test participant.
        ProvenanceError: impossible partition (guarded earlier).
    """
    if not callable(predict_fn):
        raise PredictionContractError("predict_fn must be callable")
    _validate_partition(repetition.stage, partition)
    # Verify coverage against exactly the units this repetition can see: the
    # whole development pool for outer folds, the outer training part for inner.
    referenced = {
        unit_id
        for fold in repetition.folds
        for unit_id in (*fold.train_unit_ids, *fold.eval_unit_ids)
    }
    missing = sorted(referenced - set(units))
    if missing:
        raise PredictionContractError(
            f"repetition references {len(missing)} unknown unit(s): {missing[:5]}"
        )
    assert_repetition_integrity(
        {unit_id: units[unit_id] for unit_id in referenced}, repetition
    )

    all_units: list[str] = []
    all_probs: list[float] = []
    all_labels: list[float] = []
    all_groups: list[str] = []
    all_folds: list[int] = []

    for fold in repetition.folds:
        request = PredictionRequest(
            stage=fold.stage,
            repeat_seed=fold.repeat_seed,
            fold=fold.fold,
            n_splits=fold.n_splits,
            train_unit_ids=fold.train_unit_ids,
            eval_unit_ids=fold.eval_unit_ids,
            train_group_ids=fold.train_group_ids,
            eval_group_ids=fold.eval_group_ids,
            outer_fold=fold.outer_fold,
        )
        if set(request.train_unit_ids) & set(request.eval_unit_ids):
            raise PredictionContractError(
                f"fold {fold.fold}: train and evaluation units overlap"
            )
        if firewall is not None:
            firewall.assert_not_locked(
                (*fold.train_group_ids, *fold.eval_group_ids),
                operation=f"{operation} (fold {fold.fold}, seed {fold.repeat_seed})",
            )

        raw = predict_fn(request)
        if raw is None:
            raise PredictionContractError(
                f"fold {fold.fold}: predict_fn returned None; it must return one "
                "probability per evaluation unit"
            )
        probs = np.asarray(raw, dtype=np.float64).reshape(-1)
        expected = len(fold.eval_unit_ids)
        if probs.size != expected:
            raise PredictionContractError(
                f"fold {fold.fold}: predict_fn returned {probs.size} "
                f"probabilities for {expected} evaluation units"
            )
        if not np.isfinite(probs).all():
            raise PredictionContractError(
                f"fold {fold.fold}: predict_fn returned non-finite probabilities"
            )
        if (probs < 0).any() or (probs > 1).any():
            raise PredictionContractError(
                f"fold {fold.fold}: predict_fn returned values outside [0, 1] "
                f"(min={float(probs.min()):.4f}, max={float(probs.max()):.4f})"
            )

        for unit_id, prob in zip(fold.eval_unit_ids, probs.tolist()):
            record = units[unit_id]
            all_units.append(unit_id)
            all_probs.append(float(prob))
            all_labels.append(float(record.label))
            all_groups.append(str(record.group_id))
            all_folds.append(int(fold.fold))

    if len(set(all_units)) != len(all_units):
        raise PredictionContractError(
            "a unit was predicted more than once; OOF predictions must be unique"
        )

    y_true = np.asarray(all_labels, dtype=np.float64)
    y_prob = np.asarray(all_probs, dtype=np.float64)
    groups = tuple(all_groups)

    # Every group predicted must be internally label-consistent.
    group_labels([units[u] for u in all_units])

    is_inner = repetition.stage == "inner"
    contributing = []
    if repetition.folds:
        contributing = [
            PARTITION_INNER_TRAIN if is_inner else PARTITION_OUTER_TRAIN
        ]
    provenance = build_provenance(
        partition=partition,
        operation=operation,
        dataset=dataset,
        unit=unit,
        labels=y_true,
        n_groups=len(set(groups)),
        seeds=(repetition.seed,),
        source=source,
        contributing_partitions=contributing,
        details={
            "stage": repetition.stage,
            "n_folds": repetition.n_splits,
            "outer_fold": repetition.outer_fold,
        },
    )

    return OutOfFoldPredictions(
        y_true=y_true,
        y_prob=y_prob,
        unit_ids=tuple(all_units),
        group_ids=groups,
        fold_ids=tuple(all_folds),
        stage=repetition.stage,
        repeat_seed=repetition.seed,
        provenance=provenance,
        dataset=dataset,
        n_folds=repetition.n_splits,
    )


def generate_inner_oof(
    units: Mapping[str, Unit],
    outer_fold: Fold,
    predict_fn: Callable[[PredictionRequest], Sequence[float]],
    *,
    n_splits: int = INNER_FOLDS,
    seed: Optional[int] = None,
    locked_group_ids: Optional[Sequence[str]] = None,
    dataset: str = DATASET_DEFAULT,
    unit: str = "session",
    firewall: Optional[object] = None,
) -> OutOfFoldPredictions:
    """Inner-CV OOF predictions for one outer fold (fit-safe by construction).

    The inner folds are built from ``outer_fold.train_unit_ids`` only, so outer
    evaluation participants cannot appear. The result carries partition
    ``inner_validation`` and is the only prediction object accepted by
    threshold selection and calibration planning.
    """
    inner = inner_group_folds(
        units,
        outer_fold,
        n_splits=n_splits,
        seed=seed,
        locked_group_ids=locked_group_ids,
    )
    return generate_oof_predictions(
        units,
        inner,
        predict_fn,
        partition=PARTITION_INNER_VALIDATION,
        dataset=dataset,
        unit=unit,
        firewall=firewall,
        source=f"inner_oof_outer_fold_{outer_fold.fold}",
    )
