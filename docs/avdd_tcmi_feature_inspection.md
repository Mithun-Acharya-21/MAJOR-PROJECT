# AVDD-TCMI feature inspection

Generated: 2026-11-14T08:42:41+00:00 from `data/raw/avdd_tcmi/AVDD-TCMI-dataset.zip`
by `scripts/datasets/validate_avdd_features.py`. Nothing was extracted, modified
or normalised: every number below comes from the archive member bytes.

Dataset: **1230 samples** (HDS 282, SDS 948),
CC-BY-NC-4.0, Chinese-language interview data with **pre-extracted** acoustic and
visual features and TCM facial-region images. There is no raw audio or raw video
in the release.

## 0. Extraction status

**The selective extraction has not been performed.** `data/interim/avdd_tcmi/`
does not exist, and no extraction script, report or log exists in the repository.

| Quantity | Value |
|---|---|
| Bytes needed for acoustic + visual CSVs | 36,183,442,074 B (33.7 GB) |
| Free bytes on the volume | 34,351,919,104 B (32.0 GB) |
| Fits without freeing space | **False** |

Because the ZIP is readable at ~250 MB/s decompressed, validation was run
directly against archive members instead of extracting first. This blocks
nothing; it only costs I/O per access. Report:
`data/reports/avdd_tcmi_extraction_integrity.json`.

## 1. Acoustic modality

| Check | Result |
|---|---|
| Files | 1230 ({'HDS': 282, 'SDS': 948}) |
| Unique sample ids | 1230 |
| Columns per file | 1024 (consistent: True) |
| Rows per file | min 136, median 820.5, max 1193 |
| Total values | 1037291520 |
| NaN / inf | 0 / 0 |
| Duplicate rows | 4718 |
| Constant columns | 0 |
| Empty files | 0 |
| Uncompressed size | 24.6 GB |

Column names: the acoustic CSVs are **header-less**; columns are addressed
positionally as `f0..f1023` in
`data/manifests/avdd_tcmi_acoustic_columns.csv`.

Warnings:
- 4718 rows are exact duplicates of another row in the same file (expected where a frame is silent/identical, but it means the effective frame count is 1008262)

Failed checks:
_none_

## 2. Visual modality

| Check | Result |
|---|---|
| Files | 1230 ({'HDS': 282, 'SDS': 948}) |
| Unique sample ids | 1230 |
| Columns per file | 538 (consistent: True) |
| Rows per file | min 608, median 1601.5, max 6860 |
| Total values | 1685123600 |
| NaN / inf | 0 / 0 |
| Duplicate rows | 0 |
| Constant columns | 2 |
| Empty files | 0 |
| Uncompressed size | 9.1 GB |

Column names are OpenFace-style (`frame`, `face_id`, `timestamp`, `confidence`,
`success`, `gaze_*`, `eye_lmk_*`, `pose_*`, `AU*`, ...) and are identical in
every file. Per-column statistics: `data/manifests/avdd_tcmi_visual_columns.csv`.

Warnings:
- 2 columns never vary across the whole corpus: ['face_id', 'success']

Failed checks:
_none_

## 3. Labels and alignment

| Check | Result |
|---|---|
| Samples | 1230 ({'HDS': 282, 'SDS': 948}) |
| Acoustic files / visual files | 1230 / 1230 |
| Acoustic-only samples | 0 |
| Visual-only samples | 0 |
| Depressed / non-depressed | 261 / 969 |
| Gender f / m (HDS) | 194 / 88 |
| Gender f / m (SDS) | 517 / 431 |
| Frame pairs compared | 1230 |
| Equal frame counts | 0 |
| Acoustic rows range | 136–1193 |
| Visual rows range | 608–6860 |

Label files are header-less `id,label,gender` with the original codes preserved
verbatim. `label=1` is treated as the positive (depressed) class because the
dataset card and the released label files both use it; the meaning is **not**
inferred from the signal.

Warnings:
- acoustic/visual row ratio is not a single constant (429 distinct values), so frames cannot be joined by index without a resampling policy

Failed checks:
_none_

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
