"""Phase 5B — EATD split audit, repeated grouped CV, inner CV and OOF plumbing.

Tasks A (counts and split integrity), B (repeated outer grouped CV), C (inner
grouped CV isolated from outer evaluation data) and the split-related half of
Task D (prediction callback contract, threshold policy).

Real EATD data is only read (manifests and split assignments). Everything else
uses synthetic fixtures; no model is trained and no performance is reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.nested_cv import (  # noqa: E402
    PredictionContractError,
    generate_inner_oof,
    generate_oof_predictions,
)
from ml.evaluation.provenance import (  # noqa: E402
    PARTITION_INNER_VALIDATION,
    PARTITION_LOCKED_TEST,
    PARTITION_OUTER_EVALUATION,
    ProvenanceError,
    build_provenance,
)
from ml.evaluation.splits import (  # noqa: E402
    INNER_FOLDS,
    OUTER_FOLDS,
    REPEAT_SEEDS,
    GroupingError,
    SplitError,
    SplitIntegrityError,
    Unit,
    assert_group_key_qualified,
    assert_repetition_integrity,
    audit_eatd_development_data,
    eatd_development_units,
    eatd_locked_test_group_ids,
    group_labels,
    inner_group_folds,
    participant_group_key,
    repeated_stratified_group_folds,
    unit_index,
)
from ml.evaluation.thresholds import (  # noqa: E402
    POLICY_BALANCED_ACCURACY,
    POLICY_SENSITIVITY_AT_SPECIFICITY,
    POLICY_YOUDEN_J,
    ThresholdError,
    reference_threshold,
    select_threshold,
)


def synthetic_units(n_groups: int = 40, sessions: int = 3, positive_every: int = 3):
    """Balanced-ish grouped fixture: 3 sessions per participant."""
    return [
        Unit(
            unit_id=f"p{g:03d}/s{s}",
            group_id=f"p{g:03d}",
            label=1 if g % positive_every == 0 else 0,
        )
        for g in range(n_groups)
        for s in range(sessions)
    ]


def provenance_for(labels, partition=PARTITION_INNER_VALIDATION, unit="participant"):
    return build_provenance(
        partition=partition,
        operation="threshold_selection",
        dataset="synthetic",
        unit=unit,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Task A — verified counts and split integrity on the real EATD manifests
# ---------------------------------------------------------------------------

def test_eatd_development_pool_counts_match_protocol():
    units = eatd_development_units()
    index = unit_index(units)
    groups = {u.group_id for u in units}
    labels = group_labels(units)

    assert len(groups) == 138, "development pool must hold 138 participants"
    assert len(units) == 414, "development pool must hold 414 sessions"
    assert sum(labels.values()) == 45, "45 positive participants expected"
    assert len(index) == len(units)
    assert {len([u for u in units if u.group_id == g]) for g in groups} == {3}


def test_locked_test_participants_are_excluded_and_never_labelled():
    locked = eatd_locked_test_group_ids()
    units = eatd_development_units()
    assert len(locked) == 24
    assert not ({u.group_id for u in units} & set(locked))


def test_phase2_split_integrity_assertions_still_pass():
    from ml.phase2.dataset import assert_split_integrity, load_participants, load_splits

    counts = assert_split_integrity(load_splits(), load_participants())
    assert counts == {"train": 114, "validation": 24, "test": 24}


def test_participant_key_keeps_source_partition_so_colliding_ids_stay_apart():
    key_a = participant_group_key("train", "58")
    key_b = participant_group_key("validation", "58")
    assert key_a != key_b

    records = [
        {"source_partition": "train", "subject_id": "58", "group_id": key_a},
        {"source_partition": "validation", "subject_id": "58", "group_id": key_b},
    ]
    assert_group_key_qualified(records)

    collapsed = [
        {"source_partition": "train", "subject_id": "58", "group_id": "eatd_58"},
        {"source_partition": "validation", "subject_id": "58", "group_id": "eatd_58"},
    ]
    with pytest.raises(GroupingError, match="more than one participant"):
        assert_group_key_qualified(collapsed)


def test_audit_report_reproduces_expected_counts_and_coverage():
    report = audit_eatd_development_data()
    counts = report["verified_counts"]
    assert counts["development_participants"] == 138
    assert counts["development_sessions"] == 414
    assert counts["development_positive_participants"] == 45
    assert counts["locked_test_participants"] == 24
    assert counts["locked_test_positive_participants"] == 8
    checks = report["checks"]
    for name, value in checks.items():
        if isinstance(value, bool):
            assert value is True, name
    assert checks["subject_ids_colliding_across_partitions"] == 45
    assert report["group_key"]["numeric_subject_ids_shared_by_two_development_participants"] == 45
    for expected_field, expected_value in report["expectations"].items():
        assert report["verified_counts"][expected_field] == expected_value, expected_field
    assert report["status"] == "audit_only_no_training"
    assert report["training_performed"] is False
    assert report["locked_test_evaluated"] is False
    for seed, coverage in report["outer_cv"]["coverage"].items():
        assert coverage["each_participant_once"] is True, seed
        assert coverage["covers_all_development_units"] is True, seed
        assert coverage["units_unique"] == 414


# ---------------------------------------------------------------------------
# Task B — repeated outer stratified group CV
# ---------------------------------------------------------------------------

def test_outer_cv_shape_matches_pre_registered_design():
    units = synthetic_units()
    repetitions = repeated_stratified_group_folds(units, seeds=REPEAT_SEEDS[:2])
    assert len(repetitions) == 2
    assert [r.seed for r in repetitions] == [42, 43]
    for repetition in repetitions:
        assert repetition.n_splits == OUTER_FOLDS
        assert len(repetition.folds) == OUTER_FOLDS
        assert [f.fold for f in repetition.folds] == list(range(OUTER_FOLDS))


def test_every_participant_is_evaluated_exactly_once_per_repetition():
    units = synthetic_units()
    index = unit_index(units)
    repetition = repeated_stratified_group_folds(units, seeds=(42,))[0]
    assert_repetition_integrity(index, repetition)

    eval_groups = [g for f in repetition.folds for g in f.eval_group_ids]
    eval_units = [u for f in repetition.folds for u in f.eval_unit_ids]
    assert len(eval_groups) == len(set(eval_groups)) == 40
    assert len(eval_units) == len(set(eval_units)) == 120


def test_sessions_of_one_participant_never_span_a_fold_boundary():
    units = synthetic_units()
    index = unit_index(units)
    repetition = repeated_stratified_group_folds(units, seeds=(42,))[0]
    for fold in repetition.folds:
        assert set(fold.train_unit_ids) & set(fold.eval_unit_ids) == set()
        assert set(fold.train_group_ids) & set(fold.eval_group_ids) == set()
        for group in fold.train_group_ids:
            members = {u for u in index if index[u].group_id == group}
            assert members <= set(fold.train_unit_ids)
        for group in fold.eval_group_ids:
            members = {u for u in index if index[u].group_id == group}
            assert members <= set(fold.eval_unit_ids)


def test_repeated_cv_is_deterministic_for_a_seed_and_varies_across_seeds():
    units = synthetic_units()
    first = repeated_stratified_group_folds(units, seeds=(42,))[0]
    again = repeated_stratified_group_folds(units, seeds=(42,))[0]
    other = repeated_stratified_group_folds(units, seeds=(43,))[0]

    assert [f.eval_group_ids for f in first.folds] == [
        f.eval_group_ids for f in again.folds
    ]
    assert [f.eval_group_ids for f in first.folds] != [
        f.eval_group_ids for f in other.folds
    ]


def test_fold_metadata_records_ids_counts_and_overlap_checks():
    units = synthetic_units()
    index = unit_index(units)
    fold = repeated_stratified_group_folds(units, seeds=(42,))[0].folds[0]
    meta = fold.to_dict(index)

    for side in ("train", "eval"):
        assert meta[side]["n_units"] == len(meta[side]["unit_ids"])
        assert meta[side]["n_groups"] == len(meta[side]["group_ids"])
        assert (
            meta[side]["n_positive_units"] + meta[side]["n_negative_units"]
            == meta[side]["n_units"]
        )
        assert (
            meta[side]["n_positive_groups"] + meta[side]["n_negative_groups"]
            == meta[side]["n_groups"]
        )
    assert meta["checks"]["unit_overlap"] == 0
    assert meta["checks"]["group_overlap"] == 0
    assert meta["checks"]["eval_has_both_classes"] is True


def test_unusable_grouped_split_raises_instead_of_falling_back():
    # One positive participant in ten: some evaluation fold must be single-class.
    units = synthetic_units(n_groups=10, positive_every=10)
    with pytest.raises(SplitError, match="single class"):
        repeated_stratified_group_folds(units, n_splits=5, seeds=(42,))


def test_units_without_a_group_key_are_rejected():
    units = synthetic_units(n_groups=10)
    units.append(Unit(unit_id="orphan/s0", group_id=None, label=0))
    with pytest.raises(GroupingError, match="no group_id"):
        repeated_stratified_group_folds(units, n_splits=5, seeds=(42,))


def test_participant_with_two_labels_is_rejected():
    units = synthetic_units(n_groups=10)
    units[1] = Unit(unit_id=units[1].unit_id, group_id=units[1].group_id, label=0)
    with pytest.raises(GroupingError, match="inconsistent labels"):
        repeated_stratified_group_folds(units, n_splits=5, seeds=(42,))


def test_duplicate_unit_ids_are_rejected():
    units = synthetic_units(n_groups=10)
    units.append(units[0])
    with pytest.raises(SplitIntegrityError, match="duplicate unit_id"):
        repeated_stratified_group_folds(units, n_splits=5, seeds=(42,))


def test_locked_test_participants_cannot_enter_outer_cv():
    units = synthetic_units(n_groups=10)
    with pytest.raises(SplitIntegrityError, match="locked-test participant"):
        repeated_stratified_group_folds(
            units, n_splits=5, seeds=(42,), locked_group_ids={"p000"}
        )


# ---------------------------------------------------------------------------
# Task C — inner CV inside the outer training partition
# ---------------------------------------------------------------------------

def test_inner_cv_is_built_from_the_outer_training_part_only():
    units = synthetic_units()
    index = unit_index(units)
    outer = repeated_stratified_group_folds(units, seeds=(42,))[0]

    for fold in outer.folds:
        inner = inner_group_folds(index, fold)
        assert inner.n_splits == INNER_FOLDS
        assert inner.outer_fold == fold.fold
        assert inner.seed == fold.repeat_seed, "inner seed defaults to outer seed"

        outer_eval = set(fold.eval_group_ids)
        seen_groups: set = set()
        eval_units = []
        for inner_fold in inner.folds:
            seen_groups |= set(inner_fold.train_group_ids)
            seen_groups |= set(inner_fold.eval_group_ids)
            assert not (set(inner_fold.train_group_ids) & outer_eval)
            assert not (set(inner_fold.eval_group_ids) & outer_eval)
            eval_units.extend(inner_fold.eval_unit_ids)

        assert seen_groups == set(fold.train_group_ids)
        assert not (seen_groups & outer_eval)
        assert len(eval_units) == len(set(eval_units)) == len(fold.train_unit_ids)


def test_inner_cv_excludes_locked_participants():
    units = synthetic_units()
    index = unit_index(units)
    fold = repeated_stratified_group_folds(units, seeds=(42,))[0].folds[0]
    with pytest.raises(SplitIntegrityError, match="locked-test participant"):
        inner_group_folds(index, fold, locked_group_ids={fold.train_group_ids[0]})


def test_inner_cv_rejects_an_inner_fold_as_input():
    units = synthetic_units()
    index = unit_index(units)
    outer_fold = repeated_stratified_group_folds(units, seeds=(42,))[0].folds[0]
    inner = inner_group_folds(index, outer_fold)
    with pytest.raises(SplitError, match="outer fold"):
        inner_group_folds(index, inner.folds[0])


# ---------------------------------------------------------------------------
# Task D (plumbing) — prediction callback and OOF provenance
# ---------------------------------------------------------------------------

def _oracle_callback(index):
    def predict(request):
        return [0.9 if index[u].label == 1 else 0.1 for u in request.eval_unit_ids]

    return predict


def test_oof_predictions_cover_each_unit_once_and_are_labelled_outer_evaluation():
    units = synthetic_units()
    index = unit_index(units)
    repetition = repeated_stratified_group_folds(units, seeds=(42,))[0]

    oof = generate_oof_predictions(
        index,
        repetition,
        _oracle_callback(index),
        partition=PARTITION_OUTER_EVALUATION,
        unit="session",
    )
    assert oof.y_true.size == len(units)
    assert len(set(oof.unit_ids)) == len(units)
    assert oof.provenance.partition == PARTITION_OUTER_EVALUATION
    assert oof.provenance.is_fit_safe is False
    assert oof.provenance.n_units == len(units)
    assert oof.provenance.n_groups == 40
    assert list(oof.provenance.seeds) == [42]
    summary = oof.to_dict()
    assert summary["n_folds"] == OUTER_FOLDS
    assert summary["n_rows"] == len(units)
    assert len(summary["per_fold"]) == OUTER_FOLDS


def test_outer_oof_predictions_cannot_be_used_for_threshold_selection():
    units = synthetic_units()
    index = unit_index(units)
    repetition = repeated_stratified_group_folds(units, seeds=(42,))[0]
    oof = generate_oof_predictions(
        index,
        repetition,
        _oracle_callback(index),
        partition=PARTITION_OUTER_EVALUATION,
        unit="session",
    )
    aggregated = oof.aggregate_to_participants()
    assert aggregated["provenance"].unit == "participant"
    assert aggregated["provenance"].partition == PARTITION_OUTER_EVALUATION
    with pytest.raises(ProvenanceError, match="forbidden partition"):
        select_threshold(
            aggregated["labels"],
            aggregated["probs"],
            provenance=aggregated["provenance"],
        )


def test_inner_oof_predictions_are_fit_safe_and_participant_level_aggregable():
    units = synthetic_units()
    index = unit_index(units)
    outer_fold = repeated_stratified_group_folds(units, seeds=(42,))[0].folds[0]
    inner = generate_inner_oof(index, outer_fold, _oracle_callback(index))

    assert inner.provenance.partition == PARTITION_INNER_VALIDATION
    assert inner.provenance.is_fit_safe is True
    assert set(inner.group_ids) == set(outer_fold.train_group_ids)

    aggregated = inner.aggregate_to_participants()
    assert aggregated["n_units"] == len(outer_fold.train_group_ids)
    selection = select_threshold(
        aggregated["labels"], aggregated["probs"], provenance=aggregated["provenance"]
    )
    assert 0.0 <= selection.threshold <= 1.0
    assert selection.data_partition == PARTITION_INNER_VALIDATION


def test_mislabelled_oof_partition_is_refused():
    units = synthetic_units()
    index = unit_index(units)
    repetition = repeated_stratified_group_folds(units, seeds=(42,))[0]
    with pytest.raises(PredictionContractError, match="must carry partition"):
        generate_oof_predictions(
            index,
            repetition,
            _oracle_callback(index),
            partition=PARTITION_INNER_VALIDATION,
        )


@pytest.mark.parametrize(
    "bad_output, message",
    [
        ("wrong_length", "probabilities for"),
        ("nan", "non-finite"),
        ("out_of_range", "outside"),
        ("none", "returned None"),
    ],
)
def test_prediction_callback_contract_is_enforced(bad_output, message):
    units = synthetic_units()
    index = unit_index(units)
    repetition = repeated_stratified_group_folds(units, seeds=(42,))[0]

    def predict(request):
        n = len(request.eval_unit_ids)
        if bad_output == "wrong_length":
            return [0.5] * (n - 1)
        if bad_output == "nan":
            return [float("nan")] * n
        if bad_output == "out_of_range":
            return [1.4] * n
        return None

    with pytest.raises(PredictionContractError, match=message):
        generate_oof_predictions(
            index, repetition, predict, partition=PARTITION_OUTER_EVALUATION
        )


# ---------------------------------------------------------------------------
# Threshold policy
# ---------------------------------------------------------------------------

def test_balanced_accuracy_selects_the_operating_point_it_claims():
    y = [1, 1, 0, 0]
    p = [0.9, 0.6, 0.4, 0.1]
    selection = select_threshold(
        y, p, provenance=provenance_for(y), policy=POLICY_BALANCED_ACCURACY
    )
    assert selection.threshold == pytest.approx(0.6)
    assert selection.score == pytest.approx(1.0)
    assert selection.achieved["sensitivity"] == pytest.approx(1.0)
    assert selection.achieved["specificity"] == pytest.approx(1.0)
    assert selection.is_selected_from_data is True
    assert selection.to_dict()["provenance"]["partition"] == PARTITION_INNER_VALIDATION


def test_alternative_threshold_policies_are_available_and_recorded():
    y = [1, 0, 1, 0]
    p = [0.8, 0.6, 0.4, 0.2]
    youden = select_threshold(
        y, p, provenance=provenance_for(y), policy=POLICY_YOUDEN_J
    )
    assert youden.policy == POLICY_YOUDEN_J
    # t=0.4 and t=0.8 tie on J=0.5; the declared tie-break (higher sensitivity
    # first) selects t=0.4.
    assert youden.threshold == pytest.approx(0.4)
    assert youden.achieved["sensitivity"] == pytest.approx(1.0)

    sens = select_threshold(
        y,
        p,
        provenance=provenance_for(y),
        policy=POLICY_SENSITIVITY_AT_SPECIFICITY,
        target_specificity=0.5,
    )
    assert sens.policy == POLICY_SENSITIVITY_AT_SPECIFICITY
    assert sens.threshold == pytest.approx(0.4)
    assert sens.achieved["sensitivity"] == pytest.approx(1.0)
    assert sens.achieved["specificity"] >= 0.5 - 1e-12


def test_reference_threshold_is_recorded_separately_and_not_tuned():
    reference = reference_threshold()
    assert reference.threshold == 0.5
    assert reference.policy == "fixed"
    assert reference.is_selected_from_data is False
    assert reference.provenance is None


def test_threshold_selection_requires_fit_safe_provenance():
    y = [1, 0, 1, 0]
    p = [0.7, 0.2, 0.6, 0.3]
    with pytest.raises(ProvenanceError, match="provenance"):
        select_threshold(y, p, provenance=None)  # type: ignore[arg-type]

    locked = provenance_for(y, partition=PARTITION_LOCKED_TEST)
    with pytest.raises(ProvenanceError, match="forbidden partition"):
        select_threshold(y, p, provenance=locked)


def test_threshold_selection_rejects_single_class_input():
    with pytest.raises(ThresholdError, match="single-class"):
        select_threshold(
            [0, 0, 0], [0.1, 0.2, 0.3], provenance=provenance_for([0, 0, 0])
        )


def test_unattainable_specificity_target_raises_instead_of_returning_a_number():
    y = [1, 0]
    p = [0.9, 0.9]
    with pytest.raises(ThresholdError, match="cannot be satisfied"):
        select_threshold(
            y,
            p,
            provenance=provenance_for(y),
            policy=POLICY_SENSITIVITY_AT_SPECIFICITY,
            target_specificity=0.5,
        )


def test_threshold_selection_is_deterministic():
    rng = np.random.default_rng(7)
    y = (rng.random(60) < 0.4).astype(int).tolist()
    p = np.clip(rng.random(60) * 0.6 + 0.2 * np.array(y), 0, 1).tolist()
    first = select_threshold(y, p, provenance=provenance_for(y))
    second = select_threshold(y, p, provenance=provenance_for(y))
    assert first.threshold == second.threshold
    assert first.to_dict() == second.to_dict()
