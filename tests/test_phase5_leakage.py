"""Phase 5B — ten leakage probes against the harness itself (Task G).

Each probe constructs an invalid condition with synthetic fixtures and asserts
that the harness *rejects* it. No locked-test label is read anywhere: the locked
set enters only as a participant-key set.

Probes:
 1. the same participant in train and evaluation
 2. a participant in more than one outer evaluation fold
 3. the same session in two partitions
 4. development/locked-test participant overlap
 5. an inner fold containing an outer evaluation participant
 6. threshold selection on outer evaluation predictions
 7. calibration fitted on outer evaluation predictions
 8. preprocessing fitted on evaluation data
 9. feature selection using evaluation labels
10. grouping by bare subject id when ids collide across source partitions
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.calibration import (  # noqa: E402
    CalibrationNotImplementedError,
    fit_calibrator,
    plan_calibration,
)
from ml.evaluation.provenance import (  # noqa: E402
    OPERATION_FEATURE_SELECTION,
    PARTITION_INNER_TRAIN,
    PARTITION_INNER_VALIDATION,
    PARTITION_LOCKED_TEST,
    PARTITION_OUTER_EVALUATION,
    ProvenanceError,
    assert_fit_on_training,
    build_provenance,
)
from ml.evaluation.splits import (  # noqa: E402
    Fold,
    GroupingError,
    Repetition,
    SplitIntegrityError,
    Unit,
    assert_group_key_qualified,
    assert_repetition_integrity,
    inner_group_folds,
    repeated_stratified_group_folds,
    unit_index,
)


def scenario(n_groups: int = 20, sessions: int = 2, positive_every: int = 2):
    units = [
        Unit(
            unit_id=f"p{g:03d}/s{s}",
            group_id=f"p{g:03d}",
            label=1 if g % positive_every == 0 else 0,
        )
        for g in range(n_groups)
        for s in range(sessions)
    ]
    index = unit_index(units)
    repetition = repeated_stratified_group_folds(units, n_splits=5, seeds=(42,))[0]
    return units, index, repetition


# -- probe 1 ---------------------------------------------------------------

def test_probe01_same_participant_in_train_and_eval_is_rejected():
    _, _, repetition = scenario()
    fold = repetition.folds[0]
    with pytest.raises(SplitIntegrityError, match="both train and evaluation"):
        Fold(
            stage="outer",
            repeat_seed=42,
            fold=0,
            n_splits=5,
            train_unit_ids=fold.train_unit_ids,
            eval_unit_ids=fold.eval_unit_ids,
            train_group_ids=fold.train_group_ids,
            eval_group_ids=(fold.train_group_ids[0], *fold.eval_group_ids),
        )


# -- probe 2 ---------------------------------------------------------------

def test_probe02_participant_in_two_outer_folds_is_rejected():
    _, index, repetition = scenario()
    first, second = repetition.folds[0], repetition.folds[1]
    leaked_group = first.eval_group_ids[0]
    leaked_units = tuple(
        u for u in first.eval_unit_ids if index[u].group_id == leaked_group
    )
    assert leaked_group in second.train_group_ids

    # Additional sessions of the same participant, so every unit id stays unique
    # and the rejection has to come from the participant-level check.
    extra = {}
    for unit_id in leaked_units:
        clone_id = f"{unit_id}/clone"
        extra[clone_id] = Unit(
            unit_id=clone_id,
            group_id=leaked_group,
            label=index[unit_id].label,
        )
    extended = {**index, **extra}

    forged_second = replace(
        second,
        train_unit_ids=tuple(
            u for u in second.train_unit_ids if u not in leaked_units
        ),
        train_group_ids=tuple(
            g for g in second.train_group_ids if g != leaked_group
        ),
        eval_unit_ids=second.eval_unit_ids + tuple(extra),
        eval_group_ids=second.eval_group_ids + (leaked_group,),
    )
    forged = Repetition(
        seed=42,
        stage="outer",
        n_splits=5,
        folds=(first, forged_second, *repetition.folds[2:]),
    )
    with pytest.raises(SplitIntegrityError) as excinfo:
        assert_repetition_integrity(extended, forged)
    message = str(excinfo.value)
    assert leaked_group in message
    assert (
        "more than one outer evaluation fold" in message
        or "sessions in both partitions" in message
        or "evaluated more than once" in message
    ), message

    # Reusing the same session in two folds is rejected even earlier, at fold
    # construction.
    with pytest.raises(SplitIntegrityError, match="both train and evaluation"):
        replace(second, eval_unit_ids=second.eval_unit_ids + leaked_units)


# -- probe 3 ---------------------------------------------------------------

def test_probe03_same_session_in_two_partitions_is_rejected():
    _, _, repetition = scenario()
    fold = repetition.folds[0]
    duplicated = fold.eval_unit_ids[0]
    with pytest.raises(SplitIntegrityError, match="both train and evaluation"):
        Fold(
            stage="outer",
            repeat_seed=42,
            fold=0,
            n_splits=5,
            train_unit_ids=(*fold.train_unit_ids, duplicated),
            eval_unit_ids=fold.eval_unit_ids,
            train_group_ids=fold.train_group_ids,
            eval_group_ids=fold.eval_group_ids,
        )


# -- probe 4 ---------------------------------------------------------------

def test_probe04_locked_test_participant_in_development_units_is_rejected():
    units, _, _ = scenario()
    locked = units[0].group_id
    with pytest.raises(SplitIntegrityError, match="locked-test participant"):
        repeated_stratified_group_folds(units, seeds=(42,), locked_group_ids={locked})

    with pytest.raises(SplitIntegrityError, match="locked-test participant"):
        assert_repetition_integrity(
            unit_index(units),
            repeated_stratified_group_folds(units, seeds=(42,))[0],
            locked_group_ids={locked},
        )


# -- probe 5 ---------------------------------------------------------------

def test_probe05_outer_eval_participant_cannot_enter_inner_cv():
    _, index, repetition = scenario()
    fold = repetition.folds[0]
    leaked_unit = fold.eval_unit_ids[0]
    # Forge an outer fold that hides one evaluation session inside the training
    # part while its participant is still recorded as an evaluation participant.
    forged_outer = replace(
        fold,
        train_unit_ids=(*fold.train_unit_ids, leaked_unit),
        eval_unit_ids=tuple(u for u in fold.eval_unit_ids if u != leaked_unit),
    )
    with pytest.raises(SplitIntegrityError, match="outer evaluation participant"):
        inner_group_folds(index, forged_outer)


# -- probes 6-9: selection operations on forbidden partitions ---------------

def _provenance(partition, labels=(1, 0, 1, 0, 0, 0)):
    return build_provenance(
        partition=partition,
        operation="leakage_probe",
        dataset="synthetic",
        unit="participant",
        labels=list(labels),
    )


def test_probe06_threshold_selection_on_outer_eval_is_rejected():
    from ml.evaluation.thresholds import select_threshold

    y = [1, 0, 1, 0, 0, 0]
    p = [0.8, 0.2, 0.7, 0.3, 0.4, 0.1]
    with pytest.raises(ProvenanceError, match="threshold_selection"):
        select_threshold(y, p, provenance=_provenance(PARTITION_OUTER_EVALUATION))

    # Inner OOF data is the allowed source, so the guard is not simply always on.
    selection = select_threshold(
        y, p, provenance=_provenance(PARTITION_INNER_VALIDATION)
    )
    assert selection.data_partition == PARTITION_INNER_VALIDATION


def test_probe07_calibration_fitted_on_outer_eval_is_rejected():
    with pytest.raises(ProvenanceError, match="calibration_fit"):
        plan_calibration("platt", provenance=_provenance(PARTITION_OUTER_EVALUATION))
    with pytest.raises(ProvenanceError, match="calibration_fit"):
        plan_calibration("isotonic", provenance=_provenance(PARTITION_LOCKED_TEST))

    inner_plan = plan_calibration(
        "platt", provenance=_provenance(PARTITION_INNER_VALIDATION)
    )
    assert inner_plan.provenance.partition == PARTITION_INNER_VALIDATION
    assert inner_plan.fitted is False
    # Even with a valid plan, Phase 5B cannot fit: no unreported calibration step.
    with pytest.raises(CalibrationNotImplementedError):
        fit_calibrator(inner_plan)


def test_probe08_preprocessing_fitted_on_evaluation_data_is_rejected():
    for partition in (PARTITION_OUTER_EVALUATION, PARTITION_LOCKED_TEST):
        with pytest.raises(ProvenanceError, match="preprocessing_fit"):
            assert_fit_on_training(_provenance(partition))
    assert assert_fit_on_training(_provenance(PARTITION_INNER_TRAIN)).is_fit_safe


def test_probe09_feature_selection_on_evaluation_labels_is_rejected():
    for partition in (PARTITION_OUTER_EVALUATION, PARTITION_LOCKED_TEST):
        with pytest.raises(ProvenanceError, match="feature_selection"):
            assert_fit_on_training(_provenance(partition), OPERATION_FEATURE_SELECTION)
    assert assert_fit_on_training(
        _provenance(PARTITION_INNER_TRAIN), OPERATION_FEATURE_SELECTION
    ).is_fit_safe


# -- probe 10 ---------------------------------------------------------------

def test_probe10_subject_ids_colliding_across_partitions_must_stay_separate():
    qualified = [
        {"source_partition": "train", "subject_id": "58", "group_id": "eatd_train_58"},
        {
            "source_partition": "validation",
            "subject_id": "58",
            "group_id": "eatd_validation_58",
        },
    ]
    assert_group_key_qualified(qualified)

    collapsed = [dict(record, group_id="eatd_58") for record in qualified]
    with pytest.raises(GroupingError, match="more than one participant"):
        assert_group_key_qualified(collapsed)

    # If such a collapsed key reaches the units, the label conflict is caught.
    merged = [
        Unit(unit_id="train58/s0", group_id="eatd_58", label=1),
        Unit(unit_id="validation58/s0", group_id="eatd_58", label=0),
        Unit(unit_id="train58/s1", group_id="eatd_train_58", label=1),
        Unit(unit_id="validation58/s1", group_id="eatd_validation_58", label=0),
    ]
    with pytest.raises(GroupingError, match="inconsistent labels"):
        repeated_stratified_group_folds(merged, n_splits=2, seeds=(42,))


def test_probe10_real_eatd_subject_ids_do_collide_across_partitions():
    from ml.evaluation.splits import audit_eatd_development_data

    report = audit_eatd_development_data()
    collisions = report["group_key"][
        "numeric_subject_ids_shared_by_two_development_participants"
    ]
    assert collisions > 0, (
        "the audit must report subject ids that would merge distinct people; "
        "if this is zero, the probe no longer reflects the data"
    )
