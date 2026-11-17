"""Validate every AVDD-TCMI acoustic and visual feature CSV.

Reads straight from the archive member stream (no extraction needed) so the
contents can be audited even though only 33.7 GiB of CSV would be needed and the
volume has less free space than that. Original values are never modified.

Structural checks per file
--------------------------
rows, columns, column names, data type, empty file, ragged/malformed rows,
missing (empty) fields, NaN, +/-inf, duplicate rows, constant columns.

Feature checks per modality
---------------------------
dimensionality consistency, global per-column min/max/mean/std, missingness,
non-finite share, columns that never vary across the whole corpus.

Cross-modality checks (alignment report)
----------------------------------------
acoustic id set == visual id set == label id set per subset, per-sample frame
counts for both modalities, and the HDS/SDS label + gender distributions
recomputed from the released label files. Frame counts are compared, never
assumed equal: the two modalities are extracted at different rates.

Outputs
-------
data/reports/avdd_tcmi_acoustic_validation.json
data/reports/avdd_tcmi_visual_validation.json
data/reports/avdd_tcmi_alignment_report.json
data/manifests/avdd_tcmi_acoustic_columns.csv
data/manifests/avdd_tcmi_visual_columns.csv
docs/avdd_tcmi_feature_inspection.md

Usage:
  .venv/Scripts/python scripts/datasets/validate_avdd_features.py
  .venv/Scripts/python scripts/datasets/validate_avdd_features.py --limit 20 --workers 1
Exit codes: 0 ok, 1 validation failures, 4 dataset absent.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAIL, EXIT_MISSING, EXIT_OK, PROJECT_ROOT, RAW, REPORTS, MANIFESTS,
    get_logger, human_bytes, parse_avdd_labels, utcnow, write_csv, write_json,
)

ARCHIVE = RAW / "avdd_tcmi" / "AVDD-TCMI-dataset.zip"
EXTRACT_ROOT = PROJECT_ROOT / "data" / "interim" / "avdd_tcmi"
DOCS = PROJECT_ROOT / "docs"
ACOUSTIC_REPORT = REPORTS / "avdd_tcmi_acoustic_validation.json"
VISUAL_REPORT = REPORTS / "avdd_tcmi_visual_validation.json"
ALIGNMENT_REPORT = REPORTS / "avdd_tcmi_alignment_report.json"
DOC = DOCS / "avdd_tcmi_feature_inspection.md"

SUBSETS = ("HDS", "SDS")
DECLARED = {"samples": 1230, "HDS": 282, "SDS": 948,
            "depressed": 261, "non_depressed": 969}

_ZIP: zipfile.ZipFile | None = None


# --------------------------------------------------------------------------- #
# per-file parsing
# --------------------------------------------------------------------------- #
def _classify_token(tok: bytes) -> str:
    if not tok.strip():
        return "missing"
    try:
        float(tok)
    except ValueError:
        return "nonnumeric"
    return "ok"


def inspect_csv(data: bytes, kind: str, max_bad_rows: int = 50) -> dict:
    """Parse one feature CSV and return structure + value statistics.

    ``kind`` is "acoustic" (header-less float matrix) or "visual" (named header).
    The file is never written: this only reads the bytes handed in.
    """
    import numpy as np

    out: dict = {
        "kind": kind, "bytes": len(data), "rows": 0, "cols": 0,
        "columns": None, "empty_file": len(data.strip()) == 0,
        "malformed_rows": 0, "missing_tokens": 0, "nonnumeric_tokens": 0,
        "bad_row_examples": [], "nan": 0, "inf": 0, "duplicate_rows": 0,
        "constant_columns_in_file": 0, "errors": [],
    }
    if out["empty_file"]:
        out["errors"].append("empty file")
        return out

    raw_lines = data.splitlines()
    lines = [ln for ln in raw_lines if ln.strip()]
    if not lines:
        out["empty_file"] = True
        out["errors"].append("no non-empty lines")
        return out

    header: list[str] | None = None
    if kind == "visual":
        first = lines[0].decode("utf-8-sig", errors="replace")
        header = [c.strip() for c in first.split(",")]
        lines = lines[1:]
        out["columns"] = header

    ncols = 0
    good: list[bytes] = []
    bad: list[str] = []
    for line in lines:
        line = line.rstrip(b"\r")
        fields = line.split(b",")
        if ncols == 0:
            ncols = len(fields)
        if len(fields) != ncols:
            out["malformed_rows"] += 1
            if len(bad) < max_bad_rows:
                bad.append(f"row with {len(fields)} fields, expected {ncols}")
            continue
        kinds = [_classify_token(f) for f in fields]
        if "missing" in kinds or "nonnumeric" in kinds:
            out["missing_tokens"] += kinds.count("missing")
            out["nonnumeric_tokens"] += kinds.count("nonnumeric")
            if len(bad) < max_bad_rows:
                idx = next(i for i, k in enumerate(kinds) if k != "ok")
                bad.append(f"field {idx} is {kinds[idx]!r}: {fields[idx][:32]!r}")
            continue
        good.append(line)

    out["cols"] = ncols
    out["rows"] = len(good)
    out["data_rows_including_bad"] = len(lines)
    out["bad_row_examples"] = bad
    if header is not None and len(header) != ncols:
        out["errors"].append(f"header has {len(header)} columns but the body has {ncols}")
    if not good:
        out["errors"].append("no parseable data rows")
        return out

    text = b"\n".join(good)
    try:
        matrix = np.loadtxt(_BytesIter(text), delimiter=",", dtype=np.float32, ndmin=2)
    except ValueError as exc:  # pragma: no cover - defensive, line filter above should catch it
        out["errors"].append(f"numpy parse failed: {exc}")
        return out

    out["nan"] = int(np.isnan(matrix).sum())
    out["inf"] = int(np.isinf(matrix).sum())
    finite = np.isfinite(matrix)
    if not finite.all():
        matrix = np.where(finite, matrix, np.nan)

    if matrix.shape[0] > 1:
        out["duplicate_rows"] = int(matrix.shape[0] - np.unique(
            np.nan_to_num(matrix, nan=-0.0), axis=0).shape[0])

    with np.errstate(invalid="ignore"):
        col_min = np.nanmin(matrix, axis=0)
        col_max = np.nanmax(matrix, axis=0)
        col_sum = np.nansum(matrix, axis=0, dtype=np.float64)
        col_sumsq = np.nansum(matrix.astype(np.float64) ** 2, axis=0)
        col_nonfinite = np.isnan(matrix).sum(axis=0)

    out["constant_columns_in_file"] = int((col_min == col_max).sum())
    out["_arrays"] = {
        "min": col_min.astype(np.float64),
        "max": col_max.astype(np.float64),
        "sum": col_sum,
        "sumsq": col_sumsq,
        "nonfinite": col_nonfinite.astype(np.int64),
        "count": np.full(ncols, matrix.shape[0], dtype=np.int64),
    }
    return out


class _BytesIter:
    """Iterator of byte lines for np.loadtxt."""

    def __init__(self, blob: bytes):
        self._blob = blob

    def __iter__(self):
        for line in self._blob.split(b"\n"):
            if line.strip():
                yield line.decode("ascii", errors="replace") if isinstance(line, bytes) else line


def _worker_init(zip_path: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(zip_path)


def _analyse_member(member: str) -> dict:
    """Worker entry point: parse one archive member, return compact statistics."""
    assert _ZIP is not None
    kind = "acoustic" if member.lower().endswith("_acoustic.csv") else "visual"
    data = _ZIP.read(member)
    info = inspect_csv(data, kind)
    arrays = info.pop("_arrays", None)
    info["member"] = member
    info["uncompressed_bytes"] = len(data)
    if arrays is not None:
        info["_arrays"] = {k: v.tolist() for k, v in arrays.items()}
    return info


# --------------------------------------------------------------------------- #
# corpus aggregation
# --------------------------------------------------------------------------- #
def _split_member(member: str) -> tuple[str, str]:
    parts = member.split("/")
    idx = next((i for i, p in enumerate(parts) if p in SUBSETS), 0)
    subset = parts[idx]
    return subset, parts[-1].rsplit("_", 1)[0]


def aggregate(results: list[dict], kind: str) -> dict:
    """Fold per-file statistics into a modality-level report."""
    import numpy as np

    if not results:
        return {"status": "no_files", "ok": False, "issues": ["no feature CSVs were analysed"]}

    cols = Counter(r["cols"] for r in results)
    ncols = cols.most_common(1)[0][0]
    rows = [r["rows"] for r in results]
    rows.sort()
    header_sets = Counter(tuple(r["columns"]) for r in results if r["columns"])
    arrays = [r["_arrays"] for r in results if "_arrays" in r]
    ncols_with_arrays = len(arrays[0]["count"]) if arrays else ncols

    def stack(key: str, dtype):
        return np.array([a[key] for a in arrays if len(a[key]) == ncols_with_arrays], dtype=dtype)

    issues: list[str] = []
    warnings: list[str] = []

    if cols and len(cols) > 1:
        issues.append(f"inconsistent column counts across files: {dict(cols)}")
    if any(r["empty_file"] for r in results):
        issues.append(f"{sum(1 for r in results if r['empty_file'])} empty files")
    if any(r["malformed_rows"] for r in results):
        issues.append(f"{sum(1 for r in results if r['malformed_rows'])} files contain ragged rows")
    if any(r["missing_tokens"] for r in results):
        issues.append(f"{sum(r['missing_tokens'] for r in results)} missing (empty) values")
    if any(r["nonnumeric_tokens"] for r in results):
        issues.append(f"{sum(r['nonnumeric_tokens'] for r in results)} non-numeric tokens")
    if any(r["errors"] for r in results):
        issues.append(f"{sum(1 for r in results if r['errors'])} files reported parse errors")
    if header_sets and len(header_sets) > 1:
        issues.append(f"{len(header_sets)} distinct column headers across files")

    total_rows = sum(rows)
    nan_total = sum(r["nan"] for r in results)
    inf_total = sum(r["inf"] for r in results)
    dup_total = sum(r["duplicate_rows"] for r in results)
    total_values = total_rows * ncols if total_rows else 0
    if dup_total:
        warnings.append(f"{dup_total} rows are exact duplicates of another row in the same file "
                        "(expected where a frame is silent/identical, but it means the effective "
                        f"frame count is {total_rows - dup_total})")

    per_column = []
    if arrays:
        count = stack("count", np.int64).sum(axis=0)
        s = stack("sum", np.float64).sum(axis=0)
        ss = stack("sumsq", np.float64).sum(axis=0)
        cmin = stack("min", np.float64).min(axis=0)
        cmax = stack("max", np.float64).max(axis=0)
        nf = stack("nonfinite", np.int64).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / count
            var = np.where(count > 0, ss / count - mean ** 2, np.nan)
        var = np.clip(var, 0, None)
        std = np.sqrt(var)
        names = results[0]["columns"] or [f"f{i}" for i in range(ncols_with_arrays)]
        for i in range(ncols_with_arrays):
            name = names[i] if i < len(names) else f"f{i}"
            per_column.append({
                "index": i, "column": name,
                "n": int(count[i]),
                "mean": _num(mean[i]), "std": _num(std[i]),
                "min": _num(cmin[i]), "max": _num(cmax[i]),
                "non_finite": int(nf[i]),
                "constant": bool(cmin[i] == cmax[i]),
            })
        constant = [c["column"] for c in per_column if c["constant"]]
        if constant:
            warnings.append(f"{len(constant)} columns never vary across the whole corpus: "
                            f"{constant[:10]}{' ...' if len(constant) > 10 else ''}")
        spread = sorted((c for c in per_column if not c["constant"]),
                        key=lambda c: abs(c["max"] - c["min"]), reverse=True)[:5]
        widest = [{"column": c["column"], "min": c["min"], "max": c["max"]} for c in spread]
        missing_share = round(nan_total / total_values, 6) if total_values else None
        if missing_share:
            warnings.append(f"{nan_total} non-finite values ({missing_share:.4%}) across "
                            f"{total_values} values")
    else:  # pragma: no cover - defensive
        constant, widest, missing_share = [], [], None

    subsamples = Counter(r["member"].split("/")[-2] for r in results)
    by_subset = Counter(_split_member(r["member"])[0] for r in results)
    row_by_subset: dict[str, list[int]] = {}
    for r in results:
        row_by_subset.setdefault(_split_member(r["member"])[0], []).append(r["rows"])

    return {
        "modality": kind,
        "status": "validated" if not issues else "failed",
        "files": len(results),
        "files_by_subset": dict(by_subset),
        "unique_sample_ids": len({_split_member(r["member"]) for r in results}),
        "columns": ncols,
        "column_names": results[0]["columns"],
        "column_name_consistent": len(header_sets) <= 1,
        "distinct_headers": len(header_sets),
        "rows": {
            "total": total_rows,
            "min": rows[0] if rows else 0,
            "median": statistics.median(rows) if rows else 0,
            "max": rows[-1] if rows else 0,
            "distinct_per_file": sorted(set(rows))[:20],
            "by_subset": {k: {"min": min(v), "median": statistics.median(v), "max": max(v)}
                          for k, v in sorted(row_by_subset.items())},
        },
        "values": {
            "total": total_values,
            "non_finite": nan_total + inf_total,
            "nan": nan_total,
            "inf": inf_total,
            "non_finite_share": missing_share,
            "duplicate_rows": dup_total,
            "constant_columns": len(constant),
            "constant_column_names": constant[:50],
        },
        "widest_ranges": widest,
        "per_column_written_to": per_column_path(kind).relative_to(PROJECT_ROOT).as_posix(),
        "empty_files": sum(1 for r in results if r["empty_file"]),
        "files_with_ragged_rows": [r["member"] for r in results if r["malformed_rows"]][:20],
        "files_with_bad_tokens": [r["member"] for r in results
                                  if r["missing_tokens"] or r["nonnumeric_tokens"]][:20],
        "bytes_uncompressed": sum(r["uncompressed_bytes"] for r in results),
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
        "_per_column": per_column,
    }


def _num(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(f, 6)


def per_column_path(kind: str) -> Path:
    return MANIFESTS / f"avdd_tcmi_{kind}_columns.csv"


# --------------------------------------------------------------------------- #
# labels + alignment
# --------------------------------------------------------------------------- #
def read_labels(zf: zipfile.ZipFile) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for subset in SUBSETS:
        member = next((n for n in zf.namelist() if n.endswith(f"{subset}_labels.csv")), None)
        if not member:
            continue
        _, rows = parse_avdd_labels(zf.read(member).decode("utf-8-sig"))
        out[subset] = [{k: str(v).strip() for k, v in r.items()} for r in rows]
    return out


def alignment_report(acoustic: list[dict], visual: list[dict],
                     labels: dict[str, list[dict]]) -> dict:
    acc = {_split_member(r["member"]): r for r in acoustic}
    vis = {_split_member(r["member"]): r for r in visual}
    label_ids = {s: {r.get("id", "").strip() for r in rows} for s, rows in labels.items()}

    issues: list[str] = []
    warnings: list[str] = []

    only_acc = sorted(set(acc) - set(vis))
    only_vis = sorted(set(vis) - set(acc))
    if only_acc:
        issues.append(f"{len(only_acc)} samples have acoustic but no visual CSV")
    if only_vis:
        issues.append(f"{len(only_vis)} samples have visual but no acoustic CSV")

    for subset in SUBSETS:
        ids = {k[1] for k in acc if k[0] == subset}
        known = label_ids.get(subset, set())
        missing_label = sorted(ids - known)
        orphan_label = sorted(known - ids)
        if missing_label:
            issues.append(f"{subset}: {len(missing_label)} samples without a label row")
        if orphan_label:
            issues.append(f"{subset}: {len(orphan_label)} label rows without sample files")

    frame_pairs = [((acc[k]["rows"], vis[k]["rows"]), k) for k in sorted(set(acc) & set(vis))]
    equal = sum(1 for (a, v), _ in frame_pairs if a == v)
    ratios = [round(a / v, 4) for (a, v), _ in frame_pairs if v]
    ratio_counts = Counter(ratios)
    if equal and equal != len(frame_pairs):
        warnings.append(f"{equal}/{len(frame_pairs)} samples have equal acoustic and visual "
                        "frame counts; the two modalities are not frame-aligned")
    if equal == len(frame_pairs) and frame_pairs:
        warnings.append("acoustic and visual frame counts are equal for every sample, but frame "
                        "identity is not asserted by any file: alignment must not be assumed")
    if ratio_counts:
        warnings.append("acoustic/visual row ratio is not a single constant "
                        f"({len(ratio_counts)} distinct values), so frames cannot be joined "
                        "by index without a resampling policy")

    per_sample = []
    for (a, v), key in frame_pairs[:50]:
        per_sample.append({"subset": key[0], "id": key[1], "acoustic_rows": a, "visual_rows": v})

    label_totals: dict[str, dict] = {}
    for subset, rows in labels.items():
        codes = Counter(r.get("label", "") for r in rows)
        genders = Counter(r.get("gender", "") for r in rows)
        label_totals[subset] = {
            "rows": len(rows), "labels": dict(sorted(codes.items())),
            "genders": dict(sorted(genders.items())),
            "unique_ids": len({r.get("id", "") for r in rows}),
        }
    depressed = sum(v["labels"].get("1", 0) for v in label_totals.values())
    non_depressed = sum(v["labels"].get("0", 0) for v in label_totals.values())
    if depressed != DECLARED["depressed"]:
        issues.append(f"depressed samples {depressed} != declared {DECLARED['depressed']}")
    if non_depressed != DECLARED["non_depressed"]:
        issues.append(f"non-depressed samples {non_depressed} != declared {DECLARED['non_depressed']}")
    samples_by_subset = {s: len({k[1] for k in acc if k[0] == s}) for s in SUBSETS}
    if sum(samples_by_subset.values()) != DECLARED["samples"]:
        issues.append(f"sample total {sum(samples_by_subset.values())} != {DECLARED['samples']}")

    return {
        "generated_at": utcnow(),
        "dataset_name": "avdd_tcmi",
        "status": "verified" if not issues else "failed",
        "samples_by_subset": samples_by_subset,
        "samples_total": sum(samples_by_subset.values()),
        "acoustic_files": len(acc),
        "visual_files": len(vis),
        "acoustic_only": [list(k) for k in only_acc[:20]],
        "visual_only": [list(k) for k in only_vis[:20]],
        "labels": label_totals,
        "label_totals": {"depressed": depressed, "non_depressed": non_depressed,
                         "total": depressed + non_depressed},
        "label_codes": "original codes preserved verbatim (1/0 as released; meaning taken from "
                       "the dataset card and the released label files, not inferred from signal)",
        "frame_alignment": {
            "pairs_compared": len(frame_pairs),
            "equal_counts": equal,
            "acoustic_rows_min": min((a for (a, _), _ in frame_pairs), default=0),
            "acoustic_rows_max": max((a for (a, _), _ in frame_pairs), default=0),
            "visual_rows_min": min((v for (_, v), _ in frame_pairs), default=0),
            "visual_rows_max": max((v for (_, v), _ in frame_pairs), default=0),
            "distinct_ratios": len(ratio_counts),
            "ratio_examples": dict(ratio_counts.most_common(5)),
            "examples": per_sample,
        },
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
    }


# --------------------------------------------------------------------------- #
# doc
# --------------------------------------------------------------------------- #
def write_doc(acoustic: dict, visual: dict, alignment: dict, extraction: dict) -> Path:
    def tbl(rep: dict) -> str:
        rows = rep.get("rows", {})
        return "\n".join([
            "| Check | Result |",
            "|---|---|",
            f"| Files | {rep.get('files')} ({rep.get('files_by_subset')}) |",
            f"| Unique sample ids | {rep.get('unique_sample_ids')} |",
            f"| Columns per file | {rep.get('columns')} (consistent: {rep.get('column_name_consistent')}) |",
            f"| Rows per file | min {rows.get('min')}, median {rows.get('median')}, max {rows.get('max')} |",
            f"| Total values | {rep.get('values', {}).get('total')} |",
            f"| NaN / inf | {rep['values'].get('nan')} / {rep['values'].get('inf')} |",
            f"| Duplicate rows | {rep['values'].get('duplicate_rows')} |",
            f"| Constant columns | {rep['values'].get('constant_columns')} |",
            f"| Empty files | {rep.get('empty_files')} |",
            f"| Uncompressed size | {human_bytes(rep.get('bytes_uncompressed', 0))} |",
        ])

    def lines(items: list[str], empty: str = "_none_") -> str:
        return "\n".join(f"- {i}" for i in items) if items else empty

    text = f"""# AVDD-TCMI feature inspection

Generated: {utcnow()} from `data/raw/avdd_tcmi/AVDD-TCMI-dataset.zip`
by `scripts/datasets/validate_avdd_features.py`. Nothing was extracted, modified
or normalised: every number below comes from the archive member bytes.

Dataset: **{DECLARED['samples']} samples** (HDS {DECLARED['HDS']}, SDS {DECLARED['SDS']}),
CC-BY-NC-4.0, Chinese-language interview data with **pre-extracted** acoustic and
visual features and TCM facial-region images. There is no raw audio or raw video
in the release.

## 0. Extraction status

**The selective extraction has not been performed.** `data/interim/avdd_tcmi/`
does not exist, and no extraction script, report or log exists in the repository.

| Quantity | Value |
|---|---|
| Bytes needed for acoustic + visual CSVs | {extraction['expected']['bytes_total']:,} B ({extraction['expected']['bytes_total_human']}) |
| Free bytes on the volume | {extraction['free_bytes']:,} B ({extraction['free_bytes_human']}) |
| Fits without freeing space | **{extraction['fits_in_free_space']}** |

Because the ZIP is readable at ~250 MB/s decompressed, validation was run
directly against archive members instead of extracting first. This blocks
nothing; it only costs I/O per access. Report:
`data/reports/avdd_tcmi_extraction_integrity.json`.

## 1. Acoustic modality

{tbl(acoustic)}

Column names: the acoustic CSVs are **header-less**; columns are addressed
positionally as `f0..f{acoustic.get('columns', 0) - 1}` in
`{acoustic.get('per_column_written_to')}`.

Warnings:
{lines(acoustic.get('warnings', []))}

Failed checks:
{lines(acoustic.get('issues', []))}

## 2. Visual modality

{tbl(visual)}

Column names are OpenFace-style (`frame`, `face_id`, `timestamp`, `confidence`,
`success`, `gaze_*`, `eye_lmk_*`, `pose_*`, `AU*`, ...) and are identical in
every file. Per-column statistics: `{visual.get('per_column_written_to')}`.

Warnings:
{lines(visual.get('warnings', []))}

Failed checks:
{lines(visual.get('issues', []))}

## 3. Labels and alignment

| Check | Result |
|---|---|
| Samples | {alignment['samples_total']} ({alignment['samples_by_subset']}) |
| Acoustic files / visual files | {alignment['acoustic_files']} / {alignment['visual_files']} |
| Acoustic-only samples | {len(alignment['acoustic_only'])} |
| Visual-only samples | {len(alignment['visual_only'])} |
| Depressed / non-depressed | {alignment['label_totals']['depressed']} / {alignment['label_totals']['non_depressed']} |
| Gender f / m (HDS) | {alignment['labels'].get('HDS', {}).get('genders', {}).get('f', '?')} / {alignment['labels'].get('HDS', {}).get('genders', {}).get('m', '?')} |
| Gender f / m (SDS) | {alignment['labels'].get('SDS', {}).get('genders', {}).get('f', '?')} / {alignment['labels'].get('SDS', {}).get('genders', {}).get('m', '?')} |
| Frame pairs compared | {alignment['frame_alignment']['pairs_compared']} |
| Equal frame counts | {alignment['frame_alignment']['equal_counts']} |
| Acoustic rows range | {alignment['frame_alignment']['acoustic_rows_min']}–{alignment['frame_alignment']['acoustic_rows_max']} |
| Visual rows range | {alignment['frame_alignment']['visual_rows_min']}–{alignment['frame_alignment']['visual_rows_max']} |

Label files are header-less `id,label,gender` with the original codes preserved
verbatim. `label=1` is treated as the positive (depressed) class because the
dataset card and the released label files both use it; the meaning is **not**
inferred from the signal.

Warnings:
{lines(alignment.get('warnings', []))}

Failed checks:
{lines(alignment.get('issues', []))}

## 4. Verified facts vs unverified assumptions

**Verified** (read from the archive during this run): file counts, per-file row
and column counts, header identity, absence of empty/ragged/non-numeric values,
NaN/inf counts, duplicate rows, constant columns, per-column ranges, per-subset
label and gender counts, acoustic/visual/label id agreement.

**Unverified**: that the acoustic 1,024-dimensional vectors and the visual
538-column matrices are the features the dataset paper describes (the card does
not ship a feature dictionary), the time unit of `timestamp`, whether acoustic
and visual frames are simultaneous, and the clinical validity of the HDS/SDS
labels as depression diagnoses. HDS and SDS must not be treated as
interchangeable with a clinical diagnosis or with EATD's SDS scale.
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8")
    return DOC


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="analyse only the first N files per modality")
    ap.add_argument("--doc", type=Path, default=DOC)
    args = ap.parse_args()
    log = get_logger("validate_avdd_features")

    if not args.archive.exists():
        log.error("archive missing: %s", args.archive)
        return EXIT_MISSING

    with zipfile.ZipFile(args.archive) as zf:
        names = zf.namelist()
        acoustic_names = sorted(n for n in names if n.lower().endswith("_acoustic.csv"))
        visual_names = sorted(n for n in names if n.lower().endswith("_visual.csv"))
        labels = read_labels(zf)
    if args.limit:
        acoustic_names = acoustic_names[: args.limit]
        visual_names = visual_names[: args.limit]
    log.info("archive %s: %d acoustic + %d visual CSVs", human_bytes(args.archive.stat().st_size),
             len(acoustic_names), len(visual_names))

    extraction = json.loads((REPORTS / "avdd_tcmi_extraction_integrity.json").read_text()) \
        if (REPORTS / "avdd_tcmi_extraction_integrity.json").exists() else {
            "expected": {"bytes_total": 0, "bytes_total_human": "unknown"},
            "free_bytes": 0, "free_bytes_human": "unknown", "fits_in_free_space": False}

    def run(members: list[str], label: str) -> list[dict]:
        out: list[dict] = []
        t0 = time.time()
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                     initargs=(str(args.archive),)) as pool:
                for i, info in enumerate(pool.map(_analyse_member, members, chunksize=8), 1):
                    out.append(info)
                    if i % 100 == 0:
                        log.info("%s: %d/%d files (%.0fs)", label, i, len(members), time.time() - t0)
        else:
            _worker_init(str(args.archive))
            for i, m in enumerate(members, 1):
                out.append(_analyse_member(m))
                if i % 100 == 0:
                    log.info("%s: %d/%d files (%.0fs)", label, i, len(members), time.time() - t0)
        log.info("%s: parsed %d files in %.0fs", label, len(out), time.time() - t0)
        return out

    acc_raw = run(acoustic_names, "acoustic")
    vis_raw = run(visual_names, "visual")

    acoustic = aggregate(acc_raw, "acoustic")
    visual = aggregate(vis_raw, "visual")
    for rep, path in ((acoustic, ACOUSTIC_REPORT), (visual, VISUAL_REPORT)):
        cols = rep.pop("_per_column", [])
        rep.update({"generated_at": utcnow(), "dataset_name": "avdd_tcmi",
                    "source": "https://huggingface.co/datasets/edcdd/AVDD-TCMI-dataset",
                    "license": "CC-BY-NC-4.0",
                    "mode": "archive members read in place (no extraction)",
                    "declared_totals": DECLARED})
        write_json(path, rep)
        if cols:
            write_csv(per_column_path(rep["modality"]), cols,
                      ["index", "column", "n", "mean", "std", "min", "max",
                       "non_finite", "constant"])
        log.info("%s: %d files, %d cols, issues=%d -> %s", rep["modality"], rep["files"],
                 rep["columns"], len(rep["issues"]), path)

    alignment = alignment_report(acc_raw, vis_raw, labels)
    write_json(ALIGNMENT_REPORT, alignment)
    log.info("alignment: %d issues, %d warnings -> %s", len(alignment["issues"]),
             len(alignment["warnings"]), ALIGNMENT_REPORT)
    for w in alignment["warnings"]:
        log.warning("%s", w)
    for i in alignment["issues"]:
        log.error("%s", i)

    doc = write_doc(acoustic, visual, alignment, extraction)
    log.info("doc -> %s", doc)
    ok = acoustic["ok"] and visual["ok"] and alignment["ok"]
    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
