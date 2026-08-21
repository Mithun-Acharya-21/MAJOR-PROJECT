import csv

from training.validate_dataset_manifest import validate_manifest


def test_valid_aligned_manifest_is_accepted(tmp_path) -> None:
    manifest = tmp_path / "valid.csv"
    manifest.write_text(
        "record_id,participant_id,session_id,split,purpose,alignment\n"
        "r1,p1,s1,train,multimodal_depression_finetuning,participant_session_aligned\n"
        "r2,p2,s2,test,multimodal_depression_evaluation,participant_session_aligned\n",
        encoding="utf-8",
    )

    assert validate_manifest(manifest) == []


def test_cross_split_participant_and_misalignment_are_rejected(tmp_path) -> None:
    manifest = tmp_path / "invalid.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "participant_id", "session_id", "split", "purpose", "alignment"])
        writer.writerow(["r1", "p1", "s1", "train", "multimodal_depression_finetuning", "emotion_only"])
        writer.writerow(["r2", "p1", "s2", "test", "multimodal_depression_evaluation", "participant_session_aligned"])

    errors = validate_manifest(manifest)
    assert any("participant_session_aligned" in error for error in errors)
    assert any("multiple splits" in error for error in errors)

