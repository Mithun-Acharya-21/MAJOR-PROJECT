"""Phase 5F — stream AVDD-TCMI features from the ZIP and build aggregated caches.

No source CSV is fully extracted. Each member is read in memory, aggregated to
one row per sample, and discarded. HDS and SDS are kept separate in the cache
via a ``subset`` column; no pooling is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT_ROOT / "data" / "raw" / "avdd_tcmi" / "AVDD-TCMI-dataset.zip"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "avdd_tcmi_phase5"
SEED = 42

ACOUSTIC_DIM = 1024
VISUAL_DROP = {"face_id", "success"}  # constant columns (verified)
VISUAL_EXCLUDE = {"frame", "timestamp", "confidence"}  # metadata/QC, not primary features
STATS = ["mean", "std", "median", "iqr", "p05", "p95", "slope"]
CONFIDENCE_STAT = "confidence_mean"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(zf: zipfile.ZipFile, subset: str) -> dict[int, tuple[int, str]]:
    member = f"AVDD-TCMI-dataset/{subset}_labels.csv"
    raw = zf.read(member).decode("utf-8")
    out: dict[int, tuple[int, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        sid = int(parts[0])
        label = int(parts[1])
        gender = parts[2].strip() if len(parts) > 2 else ""
        out[sid] = (label, gender)
    return out


def slope_vs_normalized_index(values: np.ndarray) -> np.ndarray:
    """Least-squares slope of each column against normalized frame index 0..1."""
    n = values.shape[0]
    if n < 2:
        return np.zeros(values.shape[1], dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    x = (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x
    # slope = cov(x, y) / var(x)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    y = values.astype(np.float64)
    yc = y - y.mean(axis=0, keepdims=True)
    return (xc @ yc) / denom


def aggregate_frame_matrix(frame_values: np.ndarray) -> np.ndarray:
    """Return one vector of STATS for each column (flattened column-major)."""
    n_frames, n_cols = frame_values.shape
    out: list[float] = []
    arr = frame_values.astype(np.float64)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0, ddof=0)
    medians = np.median(arr, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p05 = np.percentile(arr, 5, axis=0)
    p95 = np.percentile(arr, 95, axis=0)
    slopes = slope_vs_normalized_index(arr)
    # column-major: for each original column, append all stats
    for j in range(n_cols):
        out.extend(
            [
                float(means[j]),
                float(stds[j]),
                float(medians[j]),
                float(p75[j] - p25[j]),
                float(p05[j]),
                float(p95[j]),
                float(slopes[j]),
            ]
        )
    return np.asarray(out, dtype=np.float32)


def build_acoustic(zf: zipfile.ZipFile, labels: dict[int, tuple[int, str]], subset: str) -> tuple[np.ndarray, list[str], dict]:
    member_names = sorted(
        (i.filename for i in zf.infolist() if i.filename.endswith("_acoustic.csv") and f"/{subset}/" in i.filename),
        key=lambda p: int(p.split("/")[-2]),
    )
    sample_ids: list[int] = []
    labels_out: list[int] = []
    genders: list[str] = []
    n_frames_list: list[int] = []
    rows: list[np.ndarray] = []
    missing_total = 0
    nonfinite_total = 0
    duplicate_frames_total = 0
    ragged = 0
    for member in member_names:
        sid = int(member.split("/")[-2])
        if sid not in labels:
            raise ValueError(f"acoustic sample {sid} ({subset}) has no label entry")
        data = zf.read(member)
        df = pd.read_csv(io.BytesIO(data), header=None, dtype=np.float32)
        if df.shape[1] != ACOUSTIC_DIM:
            ragged += 1
            raise ValueError(f"{member}: expected {ACOUSTIC_DIM} columns, got {df.shape[1]}")
        # missing/nonfinite
        missing_total += int(df.isna().sum().sum())
        nonfinite_total += int((~np.isfinite(df.to_numpy(dtype=np.float64))).sum())
        # duplicate frames
        arr = df.to_numpy(dtype=np.float64)
        duplicate_frames_total += int(pd.DataFrame(arr).duplicated().sum())
        feature_vec = aggregate_frame_matrix(arr.astype(np.float32))
        rows.append(feature_vec)
        sample_ids.append(sid)
        if len(sample_ids) % 50 == 0:
            print(f'  visual {subset}: {len(sample_ids)}', flush=True)
        label, gender = labels[sid]
        labels_out.append(label)
        genders.append(gender)
        n_frames_list.append(int(arr.shape[0]))
    feature_names = [f"f{j}_{stat}" for j in range(ACOUSTIC_DIM) for stat in STATS]
    X = np.stack(rows, axis=0).astype(np.float32)
    audit = {
        "modality": "acoustic",
        "subset": subset,
        "n_samples": len(sample_ids),
        "n_features": int(X.shape[1]),
        "n_frames_min": int(min(n_frames_list)),
        "n_frames_median": float(np.median(n_frames_list)),
        "n_frames_max": int(max(n_frames_list)),
        "missing_values": int(missing_total),
        "nonfinite_values": int(nonfinite_total),
        "duplicate_frames": int(duplicate_frames_total),
        "ragged_rows_or_files": int(ragged),
        "feature_names_count": len(feature_names),
    }
    meta = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "labels": np.asarray(labels_out, dtype=np.int64),
        "genders": np.asarray(genders, dtype=str),
        "n_frames": np.asarray(n_frames_list, dtype=np.int64),
    }
    return X, feature_names, audit, meta


def build_visual(zf: zipfile.ZipFile, labels: dict[int, tuple[int, str]], subset: str) -> tuple[np.ndarray, list[str], dict, dict, list[str]]:
    member_names = sorted(
        (i.filename for i in zf.infolist() if i.filename.endswith("_visual.csv") and f"/{subset}/" in i.filename),
        key=lambda p: int(p.split("/")[-2]),
    )
    sample_ids: list[int] = []
    labels_out: list[int] = []
    genders: list[str] = []
    n_frames_list: list[int] = []
    confidence_means: list[float] = []
    rows: list[np.ndarray] = []
    missing_total = 0
    nonfinite_total = 0
    duplicate_frames_total = 0
    ragged = 0
    retained_columns: list[str] = []
    for idx, member in enumerate(member_names):
        sid = int(member.split("/")[-2])
        if sid not in labels:
            raise ValueError(f"visual sample {sid} ({subset}) has no label entry")
        data = zf.read(member)
        df = pd.read_csv(io.BytesIO(data))
        df.columns = [str(c).strip() for c in df.columns]
        if idx == 0:
            all_cols = list(df.columns)
            retained_columns = [c for c in all_cols if c not in VISUAL_DROP and c not in VISUAL_EXCLUDE]
            # ensure no constant metadata accidentally retained
        if list(df.columns) != all_cols if idx > 0 else False:
            pass
        if len(df.columns) != len(all_cols):
            ragged += 1
            raise ValueError(f"{member}: column count {len(df.columns)} != {len(all_cols)}")
        for col in VISUAL_DROP | VISUAL_EXCLUDE:
            if col not in df.columns:
                raise ValueError(f"{member}: expected column {col!r} missing")
        missing_total += int(df[retained_columns].isna().sum().sum())
        arr = df[retained_columns].to_numpy(dtype=np.float64)
        nonfinite_total += int((~np.isfinite(arr)).sum())
        duplicate_frames_total += int(pd.DataFrame(arr).duplicated().sum())
        feature_vec = aggregate_frame_matrix(arr.astype(np.float32))
        rows.append(feature_vec)
        sample_ids.append(sid)
        if len(sample_ids) % 50 == 0:
            print(f'  visual {subset}: {len(sample_ids)}', flush=True)
        label, gender = labels[sid]
        labels_out.append(label)
        genders.append(gender)
        n_frames_list.append(int(arr.shape[0]))
        confidence_means.append(float(df["confidence"].mean()) if "confidence" in df.columns else float("nan"))
    feature_names = [f"{col}_{stat}" for col in retained_columns for stat in STATS]
    X = np.stack(rows, axis=0).astype(np.float32)
    audit = {
        "modality": "visual",
        "subset": subset,
        "n_samples": len(sample_ids),
        "n_features": int(X.shape[1]),
        "n_retained_columns": len(retained_columns),
        "retained_columns": retained_columns,
        "dropped_constant_columns": sorted(VISUAL_DROP),
        "excluded_metadata_columns": sorted(VISUAL_EXCLUDE),
        "n_frames_min": int(min(n_frames_list)),
        "n_frames_median": float(np.median(n_frames_list)),
        "n_frames_max": int(max(n_frames_list)),
        "missing_values": int(missing_total),
        "nonfinite_values": int(nonfinite_total),
        "duplicate_frames": int(duplicate_frames_total),
        "ragged_rows_or_files": int(ragged),
        "feature_names_count": len(feature_names),
    }
    meta = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "labels": np.asarray(labels_out, dtype=np.int64),
        "genders": np.asarray(genders, dtype=str),
        "n_frames": np.asarray(n_frames_list, dtype=np.int64),
        "confidence_mean": np.asarray(confidence_means, dtype=np.float64),
    }
    return X, feature_names, audit, meta, retained_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_sha = sha256_file(args.archive)

    zf = zipfile.ZipFile(args.archive)
    try:
        all_labels: dict[str, dict[int, tuple[int, str]]] = {}
        for subset in ("HDS", "SDS"):
            all_labels[subset] = load_labels(zf, subset)

        # Acoustic: build per subset, then concatenate in a stable order (HDS then SDS)
        print('START acoustic', flush=True)
        acoustic_parts = []
        acoustic_audits = []
        acoustic_metas = []
        acoustic_names: list[str] | None = None
        for subset in ("HDS", "SDS"):
            X, names, audit, meta = build_acoustic(zf, all_labels[subset], subset)
            if acoustic_names is None:
                acoustic_names = names
            elif names != acoustic_names:
                raise RuntimeError("acoustic feature names differ across subsets")
            acoustic_parts.append((subset, X, meta))
            acoustic_audits.append(audit)
            acoustic_metas.append(meta)

        acoustic_sample_ids = np.concatenate([m["sample_ids"] for _, _, m in acoustic_parts])
        acoustic_subsets = np.concatenate([np.array([s] * len(m["sample_ids"]), dtype=str) for s, _, m in acoustic_parts])
        acoustic_labels = np.concatenate([m["labels"] for _, _, m in acoustic_parts])
        acoustic_genders = np.concatenate([m["genders"] for _, _, m in acoustic_parts])
        acoustic_nframes = np.concatenate([m["n_frames"] for _, _, m in acoustic_parts])
        acoustic_X = np.concatenate([X for _, X, _ in acoustic_parts], axis=0)

        print('START visual', flush=True)
        visual_parts = []
        visual_audits = []
        visual_metas = []
        visual_names: list[str] | None = None
        retained_cols: list[str] | None = None
        for subset in ("HDS", "SDS"):
            X, names, audit, meta, cols = build_visual(zf, all_labels[subset], subset)
            if visual_names is None:
                visual_names = names
                retained_cols = cols
            elif names != visual_names:
                raise RuntimeError("visual feature names differ across subsets")
            visual_parts.append((subset, X, meta))
            visual_audits.append(audit)
            visual_metas.append(meta)

        visual_sample_ids = np.concatenate([m["sample_ids"] for _, _, m in visual_parts])
        visual_subsets = np.concatenate([np.array([s] * len(m["sample_ids"]), dtype=str) for s, _, m in visual_parts])
        visual_labels = np.concatenate([m["labels"] for _, _, m in visual_parts])
        visual_genders = np.concatenate([m["genders"] for _, _, m in visual_parts])
        visual_nframes = np.concatenate([m["n_frames"] for _, _, m in visual_parts])
        visual_conf = np.concatenate([m["confidence_mean"] for _, _, m in visual_parts])
        visual_X = np.concatenate([X for _, X, _ in visual_parts], axis=0)
    finally:
        zf.close()

    # Write caches
    acoustic_cache = args.out_dir / "avdd_acoustic_cache.npz"
    visual_cache = args.out_dir / "avdd_visual_cache.npz"
    np.savez_compressed(
        acoustic_cache,
        X=acoustic_X,
        sample_ids=acoustic_sample_ids,
        subset=acoustic_subsets,
        labels=acoustic_labels,
        gender=acoustic_genders,
        n_frames=acoustic_nframes,
        feature_names=np.asarray(acoustic_names, dtype=str),
    )
    np.savez_compressed(
        visual_cache,
        X=visual_X,
        sample_ids=visual_sample_ids,
        subset=visual_subsets,
        labels=visual_labels,
        gender=visual_genders,
        n_frames=visual_nframes,
        confidence_mean=visual_conf,
        feature_names=np.asarray(visual_names, dtype=str),
    )

    # Per-subset manifests
    for subset in ("HDS", "SDS"):
        for modality, sample_ids_arr, labels_arr, nframes_arr in (
            ("acoustic", acoustic_sample_ids, acoustic_labels, acoustic_nframes),
            ("visual", visual_sample_ids, visual_labels, visual_nframes),
        ):
            pass

    manifest = {
        "phase": "5F",
        "dataset": "avdd_tcmi",
        "created_by": "scripts/phase5/build_avdd_feature_cache.py",
        "source_archive": str(args.archive.relative_to(PROJECT_ROOT)),
        "source_archive_sha256": archive_sha,
        "source_member_pattern": "AVDD-TCMI-dataset/{HDS,SDS}/{id}/{id}_{acoustic,visual}.csv",
        "labels_members": ["AVDD-TCMI-dataset/HDS_labels.csv", "AVDD-TCMI-dataset/SDS_labels.csv"],
        "aggregation_policy": {
            "unit": "sample (all frames within a sample)",
            "frame_order": "as stored in source CSV (no temporal alignment across modalities)",
            "normalized_index": "frame index normalized to 0..1 per sample for slope",
            "statistics": STATS,
            "acoustic_columns": f"all {ACOUSTIC_DIM} headerless columns",
            "visual_retained_columns": retained_cols,
            "visual_dropped_constant": sorted(VISUAL_DROP),
            "visual_excluded_metadata": sorted(VISUAL_EXCLUDE),
            "visual_confidence": "mean per sample retained as QC only, not a model feature",
            "note": "Acoustic and visual frames are never joined by row index; frame counts differ.",
        },
        "counts": {
            "HDS": int((acoustic_subsets == "HDS").sum()),
            "SDS": int((acoustic_subsets == "SDS").sum()),
            "total": int(len(acoustic_subsets)),
            "HDS_labels": {str(k): int(v) for k, v in sorted(Counter(acoustic_labels[acoustic_subsets == "HDS"]).items())},
            "SDS_labels": {str(k): int(v) for k, v in sorted(Counter(acoustic_labels[acoustic_subsets == "SDS"]).items())},
        },
        "audits": {
            "acoustic": acoustic_audits,
            "visual": visual_audits,
        },
        "outputs": {
            "acoustic_cache": str(acoustic_cache.relative_to(PROJECT_ROOT)),
            "acoustic_cache_sha256": sha256_file(acoustic_cache),
            "visual_cache": str(visual_cache.relative_to(PROJECT_ROOT)),
            "visual_cache_sha256": sha256_file(visual_cache),
        },
        "limitations": [
            "Sample-level unit; no participant/speaker identity is released.",
            "Acoustic columns have no feature dictionary; coefficients are not interpretable.",
            "Sample-level bootstrap CIs may be optimistic if multiple samples share a speaker.",
            "Visual retained set includes 484 raw landmark columns flagged as identity/pose/camera confounds.",
        ],
    }
    manifest_path = args.out_dir / "avdd_feature_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Per-subset manifest CSVs
    for subset in ("HDS", "SDS"):
        mask = acoustic_subsets == subset
        pd.DataFrame(
            {
                "sample_id": acoustic_sample_ids[mask],
                "subset": acoustic_subsets[mask],
                "label": acoustic_labels[mask],
                "gender": acoustic_genders[mask],
                "n_frames_acoustic": acoustic_nframes[mask],
            }
        ).to_csv(args.out_dir / f"manifest_{subset.lower()}_acoustic.csv", index=False)
        vmask = visual_subsets == subset
        pd.DataFrame(
            {
                "sample_id": visual_sample_ids[vmask],
                "subset": visual_subsets[vmask],
                "label": visual_labels[vmask],
                "gender": visual_genders[vmask],
                "n_frames_visual": visual_nframes[vmask],
                "confidence_mean": visual_conf[vmask],
            }
        ).to_csv(args.out_dir / f"manifest_{subset.lower()}_visual.csv", index=False)

    print(json.dumps(manifest["counts"], indent=2))
    print("acoustic:", acoustic_X.shape, "visual:", visual_X.shape)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
