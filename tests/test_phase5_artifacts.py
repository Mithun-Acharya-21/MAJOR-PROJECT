"""Phase 5B — artifact metadata contract and configuration (Task I).

No artifact in these tests carries a performance claim: the fixtures are
synthetic metadata, and the contract's job is to refuse anything that would
imply results that were not produced.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.artifacts import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    PROTECTED_OUTPUT_DIRS,
    REQUIRED_ARTIFACT_FIELDS,
    ArtifactValidationError,
    build_experiment_artifact,
    collect_environment,
    git_revision,
    load_artifact_schema,
    sha256_of_files,
    validate_experiment_artifact,
    write_experiment_artifact,
)
from ml.evaluation.config import DEFAULT_CONFIG_PATH  # noqa: E402


def artifact(**overrides) -> dict:
    kwargs = dict(
        experiment_id="E-T1",
        dataset={
            "name": "synthetic",
            "version": "v0",
            "provenance_reference": "tests/fixtures",
            "sha256": "0" * 64,
        },
        feature_configuration={"text": "tfidf", "max_features": 100},
        label_definition={
            "rule": "binary_label",
            "positive_class": 1,
            "definition": "synthetic fixture",
        },
        evaluation={
            "unit": "participant",
            "n_units": 12,
            "n_positive": 4,
            "split_strategy": "repeated_nested_group_cv",
            "seeds": [42, 43, 44, 45, 46],
            "fold_assignments_reference": "data/reports/folds.json",
            "fold_assignments_sha256": "1" * 64,
        },
        preprocessing={
            "configuration": {"lowercase": True},
            "fit_partition": "inner_train",
        },
        model={"configuration": {"family": "logreg"}, "hyperparameters": {"C": 1.0}},
        selection={
            "threshold_method": "balanced_accuracy",
            "threshold_partition": "inner_validation",
            "calibration_method": "none",
            "calibration_partition": "inner_validation",
        },
        known_limitations=["synthetic metadata; no model was trained"],
    )
    kwargs.update(overrides)
    return build_experiment_artifact(**kwargs)


def test_valid_artifact_passes_the_contract_and_the_published_schema():
    report = artifact()
    assert validate_experiment_artifact(report) == []

    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(instance=report, schema=load_artifact_schema())


def test_required_fields_match_the_published_schema():
    schema = load_artifact_schema()
    assert ARTIFACT_SCHEMA_VERSION in schema["$id"]
    assert tuple(schema["required"]) == tuple(REQUIRED_ARTIFACT_FIELDS)
    assert all(field in schema["properties"] for field in REQUIRED_ARTIFACT_FIELDS)


def test_missing_field_is_reported_by_name():
    report = artifact()
    del report["evaluation"]
    with pytest.raises(ArtifactValidationError, match="evaluation"):
        validate_experiment_artifact(report)
    errors = validate_experiment_artifact(report, raise_on_error=False)
    assert any("evaluation" in error for error in errors)


def test_unavailable_versions_are_recorded_as_unknown_not_invented():
    report = artifact()
    environment = report["environment"]
    assert environment["software_versions"]["python"]
    assert isinstance(environment["hardware"], dict)
    assert git_revision()
    assert set(collect_environment()["software_versions"]) >= {"python", "numpy"}


@pytest.mark.parametrize(
    "field, message",
    [("clinical_validation", "clinical_validation must be false"),
     ("serving", "serving must be false")],
)
def test_claims_of_clinical_use_or_serving_are_refused(field, message):
    report = artifact()
    report[field] = True
    with pytest.raises(ArtifactValidationError, match=message):
        validate_experiment_artifact(report)


def test_limitations_must_be_stated():
    report = artifact(known_limitations=[])
    with pytest.raises(ArtifactValidationError, match="known_limitations"):
        validate_experiment_artifact(report)


def test_schema_version_is_pinned():
    report = copy.deepcopy(artifact())
    report["schema_version"] = "phase5_experiment_artifact_v2"
    with pytest.raises(ArtifactValidationError, match="schema_version"):
        validate_experiment_artifact(report)
    assert ARTIFACT_SCHEMA_VERSION == "phase5_experiment_artifact_v1"


def test_not_measured_results_cannot_carry_numbers():
    report = artifact()
    report["results"]["metrics"] = {"roc_auc": {"value": 0.91, "unit": "participant"}}
    with pytest.raises(ArtifactValidationError, match="not_measured"):
        validate_experiment_artifact(report)


@pytest.mark.parametrize("status", ["measured", "partial"])
def test_measured_results_require_a_named_source(status):
    report = artifact()
    report["results"] = {
        "status": status,
        "metrics": {"roc_auc": {"value": 0.5, "unit": "participant"}},
        "confidence_intervals": {},
        "source": "",
    }
    with pytest.raises(ArtifactValidationError, match="results.source"):
        validate_experiment_artifact(report)

    report["results"]["source"] = "results/phase5/E-T1/report.json"
    assert validate_experiment_artifact(report) == []


def test_null_metric_must_state_why_it_is_undefined():
    report = artifact()
    report["results"] = {
        "status": "measured",
        "metrics": {"pr_auc": {"value": None, "unit": "participant"}},
        "confidence_intervals": {},
        "source": "results/phase5/E-T1/report.json",
    }
    with pytest.raises(ArtifactValidationError, match="null without a reason"):
        validate_experiment_artifact(report)

    report["results"]["metrics"]["pr_auc"]["reason"] = "single_class_in_fold"
    assert validate_experiment_artifact(report) == []


def test_metric_units_must_be_declared():
    report = artifact()
    report["results"] = {
        "status": "measured",
        "metrics": {"roc_auc": {"value": 0.5, "unit": "patient-day"}},
        "confidence_intervals": {},
        "source": "results/phase5/E-T1/report.json",
    }
    with pytest.raises(ArtifactValidationError, match="unit must be one of"):
        validate_experiment_artifact(report)


def test_selection_on_outer_evaluation_or_locked_test_is_forbidden():
    for field in ("threshold_partition", "calibration_partition"):
        for partition in ("outer_evaluation", "locked_test"):
            report = artifact()
            report["selection"][field] = partition
            report["selection"]["threshold_method"] = "balanced_accuracy"
            with pytest.raises(ArtifactValidationError, match="is forbidden"):
                validate_experiment_artifact(report)


def test_preprocessing_may_not_be_fitted_on_evaluation_data():
    report = artifact()
    report["preprocessing"]["fit_partition"] = "outer_evaluation"
    with pytest.raises(ArtifactValidationError, match="fit on training data only"):
        validate_experiment_artifact(report)


def test_evaluation_counts_and_seeds_are_checked():
    report = artifact()
    report["evaluation"]["n_positive"] = report["evaluation"]["n_units"] + 1
    with pytest.raises(ArtifactValidationError, match="cannot exceed"):
        validate_experiment_artifact(report)

    report = artifact()
    report["evaluation"]["seeds"] = []
    with pytest.raises(ArtifactValidationError, match="seeds"):
        validate_experiment_artifact(report)

    report = artifact()
    report["evaluation"]["unit"] = "fold"
    with pytest.raises(ArtifactValidationError, match="evaluation.unit"):
        validate_experiment_artifact(report)


def test_writer_round_trips_and_refuses_to_overwrite(tmp_path):
    target = tmp_path / "E-T1.json"
    written = write_experiment_artifact(artifact(), target)
    assert written == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "E-T1"
    assert payload["results"]["status"] == "not_measured"

    with pytest.raises(FileExistsError, match="already exists"):
        write_experiment_artifact(artifact(), target)

    write_experiment_artifact(artifact(), target, overwrite=True)
    assert target.is_file()


def test_writer_creates_missing_directories(tmp_path):
    target = tmp_path / "nested" / "deeper" / "E-T1.json"
    write_experiment_artifact(artifact(), target)
    assert target.is_file()


@pytest.mark.parametrize("protected", PROTECTED_OUTPUT_DIRS)
def test_writer_refuses_frozen_output_directories(protected):
    target = Path(protected) / "phase5b_should_not_exist.json"
    with pytest.raises(PermissionError, match="refusing to write into"):
        write_experiment_artifact(artifact(), target)
    assert not target.exists(), "a refused write must not create or touch the file"


def test_writer_validates_before_writing(tmp_path):
    bad = artifact()
    bad["serving"] = True
    target = tmp_path / "bad.json"
    with pytest.raises(ArtifactValidationError):
        write_experiment_artifact(bad, target)
    assert not target.exists()


def test_file_hashing_is_stable_and_content_sensitive(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    digest = sha256_of_files([first, second])
    assert digest == sha256_of_files([first, second])
    assert digest != sha256_of_files([first])
    second.write_text("gamma\n", encoding="utf-8")
    assert sha256_of_files([first, second]) != digest
    missing = sha256_of_files([tmp_path / "absent.txt"])
    assert len(missing) == 64


def test_dataset_snapshot_must_be_identifiable():
    report = artifact()
    report["dataset"]["sha256"] = ""
    with pytest.raises(ArtifactValidationError, match="dataset.sha256"):
        validate_experiment_artifact(report)

    report = artifact()
    report["dataset"]["name"] = ""
    with pytest.raises(ArtifactValidationError, match="dataset.name"):
        validate_experiment_artifact(report)


# ---------------------------------------------------------------------------
# Declared evaluation configuration
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, mutate) -> Path:
    yaml = pytest.importorskip("yaml")
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    target = tmp_path / "evaluation.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return target


def test_declared_configuration_matches_the_harness_constants():
    from ml.evaluation import bootstrap, splits, thresholds
    from ml.evaluation.config import load_evaluation_config

    config = load_evaluation_config()
    assert config.phase == "5B"
    assert config.dataset == "eatd"
    assert config.evaluation_unit == "participant"
    assert config.seeds == splits.REPEAT_SEEDS == (42, 43, 44, 45, 46)
    assert config.outer_n_splits == splits.OUTER_FOLDS
    assert config.inner_n_splits == splits.INNER_FOLDS
    assert config.splitter == "StratifiedGroupKFold"
    assert config.threshold_policy == thresholds.PRIMARY_POLICY
    assert config.reference_threshold == thresholds.REFERENCE_THRESHOLD
    assert config.calibration_fit_in_this_phase is False
    assert config.locked_test["read_in_phase_5b"] is False
    assert config.locked_test["authorized_accesses_permitted"] == 1
    assert config.bootstrap["participant_cluster"]["n_resamples"] == (
        bootstrap.DEFAULT_N_RESAMPLES
    )
    assert config.bootstrap["participant_cluster"]["seed"] == bootstrap.DEFAULT_SEED
    assert config.bootstrap["sample_level"]["participant_clustering_respected"] is False
    assert config.artifact["clinical_validation"] is False
    assert config.artifact["serving"] is False
    assert tuple(config.artifact["forbidden_output_dirs"]) == PROTECTED_OUTPUT_DIRS
    assert config.guards["ece_min_bin_n"] == 2


def test_configuration_refuses_an_ungrouped_or_mislabelled_protocol(tmp_path):
    from ml.evaluation.config import EvaluationConfigError, load_evaluation_config

    cases = [
        (lambda raw: raw["outer_cv"].__setitem__("splitter", "StratifiedKFold"),
         "StratifiedGroupKFold"),
        (lambda raw: raw.__setitem__("evaluation_unit", "session"),
         "evaluation unit must be participant"),
        (lambda raw: raw["calibration"].__setitem__("fit_in_phase_5b", True),
         "fit_in_phase_5b must be false"),
        (lambda raw: raw["locked_test"].__setitem__("read_in_phase_5b", True),
         "read_in_phase_5b must be false"),
        (lambda raw: raw.__setitem__("seeds", []), "seeds must not be empty"),
        (lambda raw: raw.pop("guards"), "missing section"),
    ]
    for mutate, message in cases:
        path = _write_config(tmp_path, mutate)
        with pytest.raises(EvaluationConfigError, match=message):
            load_evaluation_config(path)


def test_configuration_file_is_the_documented_one():
    from ml.evaluation.config import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "ml" / "configs" / "phase5" / "evaluation.yaml"
    assert DEFAULT_CONFIG_PATH.is_file()
