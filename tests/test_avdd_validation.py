"""Phase 4 — AVDD-TCMI validation utility tests.

Covers the utilities added for the feature/alignment/extraction audit. Every
test builds its own temporary fixture, so no real dataset is read or modified.

Invariants asserted here are the ones the audit claims rest on:
  * a ragged row, an empty field and a non-numeric token are each detected and
    counted separately rather than being silently parsed or skipped;
  * NaN/inf are counted, not dropped;
  * constant columns and duplicate rows are measured, not assumed absent;
  * a missing extraction is reported as ``not_extracted`` (exit 4), never as a
    pass;
  * an extracted tree that is missing files, contains duplicates, or contains a
    JPG is reported as a failure;
  * label/modality misalignment is an issue, and unequal frame rates are a
    warning rather than an error.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "datasets"))

import validate_avdd_features as vaf  # noqa: E402
import verify_avdd_extraction as vax  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _csv(rows: int, cols: int, value: float = 0.5, header: list[str] | None = None) -> bytes:
    lines = []
    if header:
        lines.append(",".join(header))
    for r in range(rows):
        lines.append(",".join(str(value + r + c * 0.01) for c in range(cols)))
    return ("\n".join(lines) + "\n").encode()


def _sample(subset: str, sid: str, rows: int, kind: str = "acoustic") -> dict:
    return {"member": f"AVDD-TCMI-dataset/{subset}/{sid}/{sid}_{kind}.csv", "rows": rows}


def _labels() -> dict[str, list[dict]]:
    """1,230 label rows: HDS 282 (123 positive), SDS 948 (138 positive)."""
    out = {}
    for subset, n, pos in (("HDS", 282, 123), ("SDS", 948, 138)):
        out[subset] = [
            {"id": str(i), "label": "1" if i < pos else "0", "gender": "0"}
            for i in range(n)
        ]
    return out


def _populations() -> tuple[list[dict], list[dict]]:
    acc, vis = [], []
    for subset, n in (("HDS", 282), ("SDS", 948)):
        for i in range(n):
            acc.append(_sample(subset, str(i), 768))
            vis.append(_sample(subset, str(i), 1538, "visual"))
    return acc, vis


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
def test_acoustic_csv_parses_with_expected_shape():
    info = vaf.inspect_csv(_csv(4, 5), "acoustic")
    assert info["rows"] == 4 and info["cols"] == 5
    assert info["errors"] == []
    assert info["malformed_rows"] == 0
    assert info["nan"] == 0 and info["inf"] == 0
    assert info["columns"] is None  # acoustic files are header-less


def test_visual_csv_keeps_column_names():
    header = ["frame", "confidence", "gaze_0_x"]
    info = vaf.inspect_csv(_csv(3, 3, header=header), "visual")
    assert info["columns"] == header
    assert info["rows"] == 3 and info["cols"] == 3
    assert info["errors"] == []


def test_empty_file_is_flagged_not_parsed_as_zero_rows():
    info = vaf.inspect_csv(b"   \n\n", "acoustic")
    assert info["empty_file"] is True
    assert info["rows"] == 0
    assert "empty file" in info["errors"]


def test_ragged_row_is_counted_as_malformed():
    data = b"1.0,2.0,3.0\n1.0,2.0\n1.0,2.0,3.0\n"
    info = vaf.inspect_csv(data, "acoustic")
    assert info["malformed_rows"] == 1
    assert info["rows"] == 2
    assert info["bad_row_examples"]


def test_missing_field_is_counted_separately_from_malformed():
    data = b"1.0,2.0,3.0\n1.0,,3.0\n"
    info = vaf.inspect_csv(data, "acoustic")
    assert info["missing_tokens"] == 1
    assert info["malformed_rows"] == 0
    assert info["rows"] == 1


def test_non_numeric_token_is_counted():
    data = b"1.0,2.0,3.0\n1.0,abc,3.0\n"
    info = vaf.inspect_csv(data, "acoustic")
    assert info["nonnumeric_tokens"] == 1


def test_nan_and_inf_are_counted_not_dropped():
    data = b"1.0,nan,3.0\n1.0,inf,3.0\n2.0,4.0,6.0\n"
    info = vaf.inspect_csv(data, "acoustic")
    assert info["rows"] == 3
    assert info["nan"] == 1
    assert info["inf"] == 1


def test_constant_and_duplicate_rows_detected():
    data = b"1.0,5.0\n1.0,5.0\n1.0,7.0\n"
    info = vaf.inspect_csv(data, "acoustic")
    assert info["duplicate_rows"] == 1
    # column 0 is 1.0 in every row
    assert info["constant_columns_in_file"] == 1


def test_header_body_width_mismatch_is_an_error():
    data = b"a,b,c\n1.0,2.0\n"
    info = vaf.inspect_csv(data, "visual")
    assert info["errors"]
    assert "header" in info["errors"][0]


def test_member_to_subset_and_id_mapping():
    assert vaf._split_member("AVDD-TCMI-dataset/HDS/0/0_acoustic.csv") == ("HDS", "0")
    assert vaf._split_member("AVDD-TCMI-dataset/SDS/123/123_visual.csv") == ("SDS", "123")


# --------------------------------------------------------------------------- #
# corpus aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_reports_consistency_and_finiteness():
    results = []
    for sid in range(3):
        info = vaf.inspect_csv(_csv(4, 6), "acoustic")
        info["member"] = f"AVDD-TCMI-dataset/HDS/{sid}/{sid}_acoustic.csv"
        info["uncompressed_bytes"] = 100
        results.append(info)
    rep = vaf.aggregate(results, "acoustic")
    assert rep["files"] == 3
    assert rep["columns"] == 6
    assert rep["rows"]["total"] == 12
    assert rep["values"]["non_finite"] == 0
    # every value differs, so no column is constant
    assert rep["values"]["constant_columns"] == 0
    assert rep["ok"] is True


def test_aggregate_flags_inconsistent_column_counts():
    a = vaf.inspect_csv(_csv(2, 4), "acoustic")
    a["member"] = "AVDD-TCMI-dataset/HDS/1/1_acoustic.csv"
    a["uncompressed_bytes"] = 10
    b = vaf.inspect_csv(_csv(2, 5), "acoustic")
    b["member"] = "AVDD-TCMI-dataset/HDS/2/2_acoustic.csv"
    b["uncompressed_bytes"] = 10
    rep = vaf.aggregate([a, b], "acoustic")
    assert rep["ok"] is False
    assert any("column counts" in i for i in rep["issues"])


def test_aggregate_flags_empty_files():
    good = vaf.inspect_csv(_csv(2, 3), "acoustic")
    good["member"] = "AVDD-TCMI-dataset/HDS/1/1_acoustic.csv"
    good["uncompressed_bytes"] = 10
    empty = vaf.inspect_csv(b"", "acoustic")
    empty["member"] = "AVDD-TCMI-dataset/HDS/2/2_acoustic.csv"
    empty["uncompressed_bytes"] = 0
    rep = vaf.aggregate([good, empty], "acoustic")
    assert rep["ok"] is False
    assert rep["empty_files"] == 1


# --------------------------------------------------------------------------- #
# label / modality alignment
# --------------------------------------------------------------------------- #
def test_alignment_clean_population_passes():
    acc, vis = _populations()
    rep = vaf.alignment_report(acc, vis, _labels())
    assert rep["ok"] is True
    assert rep["samples_total"] == 1230
    assert rep["label_totals"] == {"depressed": 261, "non_depressed": 969, "total": 1230}


def test_alignment_missing_label_row_is_an_issue():
    acc, vis = _populations()
    labels = _labels()
    labels["HDS"].pop()
    rep = vaf.alignment_report(acc, vis, labels)
    assert rep["ok"] is False
    assert any("without a label row" in i for i in rep["issues"])


def test_alignment_sample_without_visual_counterpart_is_an_issue():
    acc, vis = _populations()
    vis.pop()
    labels = _labels()
    rep = vaf.alignment_report(acc, vis, labels)
    assert rep["ok"] is False
    assert any("no visual CSV" in i for i in rep["issues"])


def test_alignment_unequal_frame_counts_warns_but_does_not_fail():
    acc, vis = _populations()
    rep = vaf.alignment_report(acc, vis, _labels())
    assert rep["ok"] is True
    assert rep["frame_alignment"]["equal_counts"] == 0
    assert any("not frame-aligned" in w or "resampling" in w for w in rep["warnings"])


# --------------------------------------------------------------------------- #
# selective extraction integrity
# --------------------------------------------------------------------------- #
def _make_archive(tmp_path: Path, n_hds: int = 2, n_sds: int = 1) -> Path:
    archive = tmp_path / "AVDD-TCMI-dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for subset, n in (("HDS", n_hds), ("SDS", n_sds)):
            for i in range(n):
                zf.writestr(f"AVDD-TCMI-dataset/{subset}/{i}/{i}_acoustic.csv", f"{i}.0,1.0\n")
                zf.writestr(f"AVDD-TCMI-dataset/{subset}/{i}/{i}_visual.csv", "f,a\n1,2\n")
                zf.writestr(f"AVDD-TCMI-dataset/{subset}/{i}/{i}_tcm/0.jpg", b"\xff\xd8")
    return archive


def test_extraction_absent_is_reported_as_not_extracted(tmp_path):
    archive = _make_archive(tmp_path)
    rep = vax.verify(tmp_path / "missing_root", archive)
    assert rep["status"] == "not_extracted"
    assert rep["extracted"] is False
    assert rep["ok"] is False
    assert rep["missing"] == {"total": 6, "acoustic": 3, "visual": 3}
    assert rep["expected"]["bytes_total"] > 0


def test_extraction_verified_when_tree_matches_archive(tmp_path):
    archive = _make_archive(tmp_path)
    root = tmp_path / "interim" / "avdd_tcmi"
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            target = root / name.split("/", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    rep = vax.verify(root, archive, compare_bytes=True)
    assert rep["status"] == "verified", rep["issues"]
    assert rep["counts"] == {"acoustic": 3, "visual": 3, "total": 6}
    assert rep["byte_comparison"]["mismatch_count"] == 0
    assert rep["byte_comparison"]["files_crc_checked"] == 6


def test_extraction_missing_files_fails(tmp_path):
    archive = _make_archive(tmp_path)
    root = tmp_path / "interim"
    (root / "HDS" / "0").mkdir(parents=True)
    (root / "HDS" / "0" / "0_acoustic.csv").write_bytes(b"1.0,1.0\n")
    rep = vax.verify(root, archive)
    assert rep["status"] == "failed"
    assert rep["missing"]["total"] == 5


def test_extraction_duplicate_ids_fail(tmp_path):
    archive = _make_archive(tmp_path)
    root = tmp_path / "interim"
    for sub in ("a", "b"):
        d = root / sub / "HDS" / "0"
        d.mkdir(parents=True)
        (d / "0_acoustic.csv").write_bytes(b"1.0,1.0\n")
    rep = vax.verify(root, archive)
    assert rep["duplicate_ids"] == 1
    assert rep["ok"] is False


def test_extraction_unexpected_jpg_fails(tmp_path):
    archive = _make_archive(tmp_path)
    root = tmp_path / "interim"
    (root / "HDS" / "0" / "0_tcm").mkdir(parents=True)
    (root / "HDS" / "0" / "0_tcm" / "0.jpg").write_bytes(b"\xff\xd8")
    rep = vax.verify(root, archive)
    assert rep["images_extracted"] == 1
    assert rep["ok"] is False


def test_extraction_byte_mismatch_is_detected(tmp_path):
    archive = _make_archive(tmp_path)
    root = tmp_path / "interim"
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            target = root / name.split("/", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    (root / "HDS" / "0" / "0_acoustic.csv").write_bytes(b"9.0,9.0\n")
    rep = vax.verify(root, archive, compare_bytes=True)
    assert rep["status"] == "failed"
    assert rep["byte_comparison"]["mismatch_count"] >= 1


# --------------------------------------------------------------------------- #
# report artifacts (skipped when not generated)
# --------------------------------------------------------------------------- #
HARVARD_REPORT = PROJECT_ROOT / "data" / "reports" / "harvard_tweets_validation.json"
INVENTORY = PROJECT_ROOT / "data" / "reports" / "dataset_inventory.json"


@pytest.mark.skipif(not HARVARD_REPORT.exists(), reason="Harvard report not generated")
def test_harvard_label_ambiguity_isrecorded_not_resolved():
    rep = json.loads(HARVARD_REPORT.read_text(encoding="utf-8"))
    mapping = rep["label_code_mapping"]
    assert mapping["ambiguous"] is True
    assert "preserved" in mapping["policy"]
    # both codes are still present, unchanged
    assert rep["statistics"]["label_distribution"] == {"0": 58405, "1": 41185}
    assert rep["label_semantics"]["status"] == "unresolved"
    assert rep["label_semantics"]["supervised_use_allowed"] is False
    assert rep["label_semantics"]["depositor_mapping_found"] is False


@pytest.mark.skipif(not INVENTORY.exists(), reason="inventory not generated")
def test_inventory_consistency():
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by = {d["dataset_name"]: d for d in inv["datasets"]}
    assert inv["dataset_count"] == 6
    assert set(inv["available"]) == {"eatd", "harvard_tweets", "avdd_tcmi"}
    assert set(inv["access_pending"]) == {"daic_woz", "rsdd", "smhd"}
    # a sample count is never promoted to a participant count
    assert by["avdd_tcmi"]["participants"] is None
    assert by["avdd_tcmi"]["records"] == 1230
    # Harvard must not be advertised as label-verified
    assert by["harvard_tweets"]["label_mapping_verified"] is False
    assert by["eatd"]["label_mapping_verified"] is True
    # access-pending datasets carry no record counts
    for name in ("daic_woz", "rsdd", "smhd"):
        assert by[name]["status"] == "ACCESS_PENDING"
        assert by[name]["records"] is None
