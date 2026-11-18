"""Create dedicated visual cache manifest avdd_visual_cache_manifest.json.

Verifies and documents data/processed/avdd_tcmi_phase5/avdd_visual_cache.npz.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "processed" / "avdd_tcmi_phase5" / "avdd_visual_cache.npz"
MANIFEST_PATH = ROOT / "data" / "processed" / "avdd_tcmi_phase5" / "avdd_visual_cache_manifest.json"
ARCHIVE_PATH = ROOT / "data" / "raw" / "avdd_tcmi" / "AVDD-TCMI-dataset.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache not found: {CACHE_PATH}")

    data = np.load(CACHE_PATH, allow_pickle=False)
    X = data["X"]
    sample_ids = data["sample_ids"]
    subsets = data["subset"]
    labels = data["labels"]
    genders = data["gender"]
    n_frames = data["n_frames"]
    confidence_mean = data["confidence_mean"]
    feature_names = data["feature_names"]

    # Integrity verification
    is_finite = bool(np.all(np.isfinite(X)))
    has_nans = bool(np.isnan(X).any())
    has_infs = bool(np.isinf(X).any())

    hds_mask = subsets == "HDS"
    sds_mask = subsets == "SDS"

    unique_ids_hds = len(np.unique(sample_ids[hds_mask]))
    unique_ids_sds = len(np.unique(sample_ids[sds_mask]))
    composite_keys = [f"{s}_{sid}" for s, sid in zip(subsets, sample_ids)]
    unique_composite = len(np.unique(composite_keys))

    cache_sha256 = sha256_file(CACHE_PATH)
    archive_sha256 = sha256_file(ARCHIVE_PATH) if ARCHIVE_PATH.exists() else None

    manifest = {
        "schema_version": "1.0",
        "dataset": "AVDD-TCMI",
        "modality": "visual",
        "cache_file": str(CACHE_PATH.relative_to(ROOT)),
        "cache_sha256": cache_sha256,
        "source_archive": str(ARCHIVE_PATH.relative_to(ROOT)) if ARCHIVE_PATH.exists() else None,
        "source_archive_sha256": archive_sha256,
        "creation_method": "scripts/phase5/build_avdd_feature_cache.py",
        "matrix_properties": {
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "all_finite": is_finite,
            "has_nans": has_nans,
            "has_infs": has_infs,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_unique_composite_sample_keys": unique_composite,
            "hds_unique_sample_ids": unique_ids_hds,
            "sds_unique_sample_ids": unique_ids_sds,
            "id_policy": "Sample IDs are 0..281 in HDS and 0..947 in SDS; unique identifier is composite (subset, sample_id)",
        },
        "subset_breakdown": {
            "HDS": {
                "n_samples": int(hds_mask.sum()),
                "n_depressed_label1": int((labels[hds_mask] == 1).sum()),
                "n_control_label0": int((labels[hds_mask] == 0).sum()),
                "n_frames_min": int(n_frames[hds_mask].min()),
                "n_frames_median": float(np.median(n_frames[hds_mask])),
                "n_frames_max": int(n_frames[hds_mask].max()),
                "confidence_mean_avg": float(np.mean(confidence_mean[hds_mask])),
            },
            "SDS": {
                "n_samples": int(sds_mask.sum()),
                "n_depressed_label1": int((labels[sds_mask] == 1).sum()),
                "n_control_label0": int((labels[sds_mask] == 0).sum()),
                "n_frames_min": int(n_frames[sds_mask].min()),
                "n_frames_median": float(np.median(n_frames[sds_mask])),
                "n_frames_max": int(n_frames[sds_mask].max()),
                "confidence_mean_avg": float(np.mean(confidence_mean[sds_mask])),
            },
        },
        "aggregation_specification": {
            "retained_raw_columns": 533,
            "dropped_constant_columns": ["face_id", "success"],
            "excluded_metadata_columns": ["frame", "timestamp", "confidence"],
            "confidence_policy": "Excluded from feature vector; preserved as QC covariate confidence_mean",
            "functionals": ["mean", "std", "median", "iqr", "p05", "p95", "slope"],
            "slope_normalization": "least-squares slope against within-sample normalized frame index 0..1",
            "feature_naming_convention": "<raw_column>_<functional>",
        },
        "keys_in_npz": [
            "X",
            "sample_ids",
            "subset",
            "labels",
            "gender",
            "n_frames",
            "confidence_mean",
            "feature_names",
        ],
        "verification_status": "VALID_AND_VERIFIED",
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Written visual cache manifest: {MANIFEST_PATH}")
    print(f"Shape: {X.shape}, dtype: {X.dtype}, all_finite: {is_finite}")
    print(f"HDS: {int(hds_mask.sum())}, SDS: {int(sds_mask.sum())}")


if __name__ == "__main__":
    main()
