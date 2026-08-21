import pytest

from training.prepare_dataset import SplitRatios, allocate_participants, read_index, write_manifests


def _row(record_id: str, participant_id: str, label: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "participant_id": participant_id,
        "session_id": f"s-{participant_id}",
        "dataset_id": "approved-demo",
        "dataset_version": "v1",
        "label": label,
        "purpose": "multimodal_depression_finetuning",
        "alignment": "participant_session_aligned",
        "content_hash": f"hash-{record_id}",
    }


def test_participant_allocation_is_disjoint_and_writes_all_splits(tmp_path) -> None:
    rows = [_row(f"c{i}", f"c{i}", "control") for i in range(3)]
    rows += [_row(f"e{i}", f"e{i}", "elevated") for i in range(3)]

    allocation = allocate_participants(rows, SplitRatios(), seed=7)
    assert set(allocation.values()) == {"train", "validation", "test"}
    write_manifests(rows, allocation, tmp_path)
    assert {(tmp_path / f"{split}.csv").exists() for split in ("train", "validation", "test")} == {True}
    assert (tmp_path / "split-summary.txt").exists()


def test_misaligned_fusion_row_is_rejected(tmp_path) -> None:
    index = tmp_path / "bad.csv"
    index.write_text(
        "record_id,participant_id,session_id,dataset_id,dataset_version,label,purpose,alignment\n"
        "r1,p1,s1,approved,v1,control,multimodal_depression_finetuning,emotion_only\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="participant_session_aligned"):
        read_index(index)


def test_conflicting_participant_label_is_rejected() -> None:
    rows = [_row("r1", "p1", "control"), _row("r2", "p1", "elevated")]

    with pytest.raises(ValueError, match="one stable label"):
        allocate_participants(rows, SplitRatios(), seed=42)

