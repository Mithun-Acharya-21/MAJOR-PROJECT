# DAIC-WOZ Experimental Reproducibility Protocol

---

## 1. Deterministic Execution Parameters

- **Random Seed:** Fixed to `42` across NumPy, PyTorch, and scikit-learn.
- **Hardware Platform:** Windows 11 | Python 3.14.0 | PyTorch 2.13.0+cu130 | NVIDIA GeForce RTX 3050 6GB.
- **Data Partitions:** Stratified participant-level splits stored in `data/splits/daic_woz/` (immutable test set).
- **Automated Test Suite:** `tests/test_daic_woz_pipeline.py` verifies data structures, schemas, and quarantine boundaries.

---

## 2. Integrity Verification Checksums

All source archives placed in `data/raw/daic_woz/` are recorded in `data/manifests/daic_woz/download_manifest.csv` with full SHA-256 signatures before training initiation.
