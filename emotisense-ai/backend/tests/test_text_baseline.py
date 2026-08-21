import csv

import pytest

from training.text.baseline import assert_no_participant_overlap, read_text_split


def test_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_no_participant_overlap(["p1", "p2"], ["p2", "p3"])


def test_test_named_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "final-test.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "participant_id", "label", "text"])
        writer.writeheader()
        writer.writerow({"record_id": "r1", "participant_id": "p1", "label": "a", "text": "hello"})
        writer.writerow({"record_id": "r2", "participant_id": "p2", "label": "b", "text": "world"})

    with pytest.raises(ValueError, match="final test"):
        read_text_split(path, "validation")

