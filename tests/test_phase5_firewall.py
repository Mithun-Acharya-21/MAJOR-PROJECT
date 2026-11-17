"""Phase 5B — locked-test firewall behaviour (Task H).

The firewall knows locked participant keys, never locked labels. These tests
verify what it blocks, what it records, and that the single authorised access
cannot be repeated or taken with an incomplete pre-registration.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.firewall import (  # noqa: E402
    REQUIRED_PREREGISTRATION_FIELDS,
    LockedTestAccessError,
    LockedTestFirewall,
)
from ml.evaluation.nested_cv import generate_oof_predictions  # noqa: E402
from ml.evaluation.splits import (  # noqa: E402
    Unit,
    repeated_stratified_group_folds,
    unit_index,
)


def firewall(tmp_path=None, locked=("lock1", "lock2")) -> LockedTestFirewall:
    ledger = None if tmp_path is None else tmp_path / "locked_access.jsonl"
    return LockedTestFirewall(locked, ledger_path=ledger, code_revision="deadbeef")


def preregistration(experiment_id="E-T1", script_sha256="0" * 64) -> dict:
    return {
        "experiment_id": experiment_id,
        "model_configuration": {"family": "tfidf_logreg", "version": "frozen"},
        "threshold_rule": "balanced_accuracy, frozen at 0.5",
        "primary_metric": "roc_auc",
        "evaluation_script": "scripts/phase5/final_evaluation.py",
        "script_sha256": script_sha256,
        "registered_at": "2026-01-01T00:00:00+00:00",
    }


def test_firewall_from_eatd_splits_holds_keys_only():
    guard = LockedTestFirewall.from_eatd_splits()
    assert guard.n_locked_participants == 24
    assert guard.labels_available is False
    report = guard.to_dict()
    assert report["labels_available"] is False
    assert report["n_locked_participants"] == 24
    assert set(report) == {
        "dataset",
        "n_locked_participants",
        "locked_group_ids_recorded",
        "labels_available",
        "authorized_accesses",
        "scope",
    }
    assert "eatd_" not in json.dumps(report), (
        "locked participant keys must not be serialised into reports"
    )


def test_empty_firewall_is_refused():
    with pytest.raises(LockedTestAccessError, match="at least one locked"):
        LockedTestFirewall([])


def test_locked_participants_are_blocked_from_any_operation():
    guard = firewall()
    assert guard.is_locked("lock1") is True
    assert guard.is_locked("other") is False
    assert guard.locked_overlap(["lock2", "x", "lock1"]) == ["lock1", "lock2"]

    with pytest.raises(LockedTestAccessError, match="locked-test"):
        guard.assert_not_locked(["lock1", "p1"], operation="model_fit")
    with pytest.raises(LockedTestAccessError, match="selection"):
        guard.assert_selection_safe(["lock2"], operation="threshold_selection")


def test_development_filter_refuses_contaminated_input():
    guard = firewall()
    clean = [Unit(unit_id="u1", group_id="p1", label=0)]
    assert guard.filter_development(clean) == tuple(clean)

    contaminated = [
        Unit(unit_id="u1", group_id="p1", label=0),
        Unit(unit_id="u2", group_id="lock1", label=1),
    ]
    with pytest.raises(LockedTestAccessError, match="locked-test participant"):
        guard.filter_development(contaminated)


def test_locked_participants_cannot_reach_inner_cv_through_the_runner():
    units = [
        Unit(unit_id=f"p{g}/s{s}", group_id=f"p{g:02d}", label=g % 2)
        for g in range(10)
        for s in range(2)
    ]
    index = unit_index(units)
    guard = LockedTestFirewall({"p00"})
    repetition = repeated_stratified_group_folds(units, n_splits=2, seeds=(42,))[0]
    with pytest.raises(LockedTestAccessError, match="locked-test"):
        generate_oof_predictions(
            index,
            repetition,
            lambda request: [0.5] * len(request.eval_unit_ids),
            partition="outer_evaluation",
            firewall=guard,
        )


def test_locked_test_partition_cannot_be_declared_by_model_selection():
    guard = firewall()
    guard.assert_partition_pristine(["inner_train", "inner_validation"])
    with pytest.raises(LockedTestAccessError, match="locked_test"):
        guard.assert_partition_pristine(["inner_train", "locked_test"])


def test_final_evaluation_requires_a_complete_preregistration():
    guard = firewall()
    incomplete = {k: v for k, v in preregistration().items() if k != "primary_metric"}
    with pytest.raises(LockedTestAccessError, match="incomplete"):
        with guard.authorize_final_evaluation(
            experiment_id="E-T1", reason="phase 5 confirmatory", preregistration=incomplete
        ):
            pass

    with pytest.raises(LockedTestAccessError, match="experiment_id"):
        with guard.authorize_final_evaluation(
            experiment_id="E-T9",
            reason="phase 5 confirmatory",
            preregistration=preregistration(),
        ):
            pass

    with pytest.raises(LockedTestAccessError, match="reason"):
        with guard.authorize_final_evaluation(
            experiment_id="E-T1", reason="", preregistration=preregistration()
        ):
            pass

    assert guard.access_log == (), "failed authorisations must not be logged"


def test_final_evaluation_is_authorised_once_and_written_to_the_ledger(tmp_path):
    guard = firewall(tmp_path)
    with guard.authorize_final_evaluation(
        experiment_id="E-T1",
        reason="single confirmatory run on the locked test set",
        preregistration=preregistration(),
    ) as record:
        assert record.status == "running"
        ledger = tmp_path / "locked_access.jsonl"
        assert ledger.is_file(), "ledger entry must exist before the body runs"
        entry = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        assert entry["experiment_id"] == "E-T1"
        assert entry["scope"] == "final_evaluation_only"
        assert entry["n_locked_participants"] == 2
        assert entry["code_revision"] == "deadbeef"

    assert record.status == "completed"
    assert record.completed_at is not None
    assert len(guard.access_log) == 1

    with pytest.raises(LockedTestAccessError, match="already been authorised"):
        with guard.authorize_final_evaluation(
            experiment_id="E-T2",
            reason="second look",
            preregistration=preregistration("E-T2"),
        ):
            pass


def test_preregistration_fields_are_all_required():
    guard = firewall()
    for field in REQUIRED_PREREGISTRATION_FIELDS:
        partial = {k: v for k, v in preregistration().items() if k != field}
        with pytest.raises(LockedTestAccessError, match="incomplete"):
            with guard.authorize_final_evaluation(
                experiment_id="E-T1",
                reason="probe",
                preregistration=partial,
            ):
                pass


def test_evaluation_script_hash_must_match_the_preregistration(tmp_path):
    script = tmp_path / "final_evaluation.py"
    script.write_text("print('frozen evaluation script')\n", encoding="utf-8")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()

    guard = firewall(tmp_path)
    with pytest.raises(LockedTestAccessError, match="hash does not match"):
        with guard.authorize_final_evaluation(
            experiment_id="E-T1",
            reason="probe",
            preregistration=preregistration(script_sha256="f" * 64),
            script_path=script,
        ):
            pass

    with guard.authorize_final_evaluation(
        experiment_id="E-T1",
        reason="frozen script, frozen rule",
        preregistration=preregistration(script_sha256=digest),
        script_path=script,
    ):
        pass
    assert guard.access_log[0]["status"] == "completed"


def test_failed_final_evaluation_is_recorded_as_failed(tmp_path):
    guard = firewall(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with guard.authorize_final_evaluation(
            experiment_id="E-T1",
            reason="probe",
            preregistration=preregistration(),
        ):
            raise RuntimeError("boom")
    assert guard.access_log[0]["status"] == "failed"
