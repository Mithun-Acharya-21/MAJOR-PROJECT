"""
Experiment V1 — openSMILE eGeMAPS + Logistic Regression voice baseline.

Feature set: openSMILE 2.6.0, ``eGeMAPSv02`` at the ``Functionals`` level
(88 standardised acoustic functionals: F0, jitter, shimmer, HNR, spectral
shape, MFCC1-4, loudness, formants, voiced-segment statistics).

Audio handling (documented, never silent):
  * WAV is read with soundfile at its native rate. All 486 EATD files are
    16 kHz / stereo per the Phase 1 audit; stereo is averaged to mono.
    A file whose rate is not 16 kHz is resampled with librosa and counted
    under ``resampled``.
  * Sessions the Phase 1 audit flagged ``audio_usable == False`` (the single
    0.48 s recording) are excluded up front and listed in
    ``excluded_sessions`` with a reason.
  * If openSMILE returns an all-NaN vector (segment too short for F0/variance
    statistics) the session is excluded and listed — not silently zero-filled.
  * Remaining sporadic NaNs are imputed with TRAIN medians only.

Preprocessing fitted on train only: median imputer + StandardScaler.
Model selection on VALIDATION participant-level ROC-AUC; test scored once.

Exports:
    FeatureExtractionConfig, extract_session_features, extract_corpus,
    run_voice_egemaps
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from ml.phase2.dataset import load_sessions, resolve_path
from ml.phase2.evaluation import (
    aggregate_by_participant,
    evaluate_predictions,
    experiment_metadata,
    metric_bundle,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_ID = "eatd_voice_egemaps_v1"
MODEL_VERSION = "voice-eatd-egemaps-2.0.0"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "experiments" / EXPERIMENT_ID
RESULT_PATH = PROJECT_ROOT / "results" / "phase2" / "voice" / f"{EXPERIMENT_ID}.json"

MIN_DURATION_SEC = 1.0
TARGET_SR = 16000


@dataclass
class FeatureExtractionConfig:
    """Reproducible openSMILE configuration record."""

    opensmile_version: str
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "Functionals"
    target_sample_rate: int = TARGET_SR
    min_duration_sec: float = MIN_DURATION_SEC
    mono: bool = True
    nan_policy: str = "impute_train_median"
    excluded_sessions: List[dict] = field(default_factory=list)
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "opensmile_version": self.opensmile_version,
            "feature_set": self.feature_set,
            "feature_level": self.feature_level,
            "target_sample_rate": self.target_sample_rate,
            "min_duration_sec": self.min_duration_sec,
            "mono": self.mono,
            "nan_policy": self.nan_policy,
            **self.extra,
            "excluded_sessions": self.excluded_sessions,
        }


def _load_audio(audio_path: str) -> Tuple[np.ndarray, int, Dict[str, object]]:
    import soundfile as sf

    data, sr = sf.read(resolve_path(audio_path), dtype="float32", always_2d=True)
    meta: Dict[str, object] = {"original_sample_rate": int(sr), "channels": int(data.shape[1])}
    y = data.mean(axis=1)  # stereo → mono
    if sr != TARGET_SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        meta["resampled"] = True
        sr = TARGET_SR
    else:
        meta["resampled"] = False
    return np.asarray(y, dtype=np.float32), sr, meta


def _make_smile():
    import opensmile

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_session_features(
    row: dict,
    smile=None,
    config: Optional[FeatureExtractionConfig] = None,
) -> Tuple[Optional[np.ndarray], List[str], Dict[str, object]]:
    """eGeMAPS functionals for one session, or (None, [], reason) if unusable."""
    import opensmile

    smile = smile or _make_smile()
    config = config or FeatureExtractionConfig(opensmile_version=opensmile.__version__)

    if not row.get("audio_usable", True):
        return None, [], {"excluded": True, "reason": "marked_audio_unusable_in_phase1_audit"}

    try:
        y, sr, meta = _load_audio(row["audio_path"])
    except Exception as e:  # corrupt/undecodable file
        return None, [], {"excluded": True, "reason": f"audio_read_error: {type(e).__name__}"}

    duration = len(y) / sr
    if duration < config.min_duration_sec:
        return None, [], {
            "excluded": True,
            "reason": f"duration_{duration:.2f}s_below_{config.min_duration_sec}s",
        }

    try:
        frame = smile.process_signal(y, sr)
    except Exception as e:
        return None, [], {"excluded": True, "reason": f"opensmile_error: {type(e).__name__}"}

    columns = list(frame.columns)
    values = frame.to_numpy(dtype=np.float64).reshape(-1)
    if values.size == 0 or np.all(np.isnan(values)):
        return None, [], {
            "excluded": True,
            "reason": "opensmile_all_nan (segment too short for functionals)",
        }
    info = {
        "excluded": False,
        "duration_sec": round(duration, 3),
        "n_nan_features": int(np.isnan(values).sum()),
        **meta,
    }
    return values, columns, info


def extract_corpus(
    rows: Sequence[dict],
    cache_path: Optional[Path] = None,
    reuse_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str], FeatureExtractionConfig, Dict[str, object]]:
    """Extract eGeMAPS for all rows; returns X, y, kept rows, feature names, config, stats."""
    import opensmile

    if reuse_cache and cache_path is not None and Path(cache_path).exists():
        import json

        data = np.load(cache_path, allow_pickle=False)
        meta = json.loads(Path(str(cache_path) + ".json").read_text(encoding="utf-8"))
        cfg_raw = meta["config"]
        config = FeatureExtractionConfig(
            opensmile_version=cfg_raw["opensmile_version"],
            feature_set=cfg_raw.get("feature_set", "eGeMAPSv02"),
            feature_level=cfg_raw.get("feature_level", "Functionals"),
            target_sample_rate=int(cfg_raw.get("target_sample_rate", TARGET_SR)),
            min_duration_sec=float(cfg_raw.get("min_duration_sec", MIN_DURATION_SEC)),
            mono=bool(cfg_raw.get("mono", True)),
            nan_policy=cfg_raw.get("nan_policy", "impute_train_median"),
            excluded_sessions=cfg_raw.get("excluded_sessions", []),
        )
        return (
            data["X"], data["y"], meta["rows"], meta["feature_names"], config,
            {"cached": True, **meta["stats"]},
        )

    smile = _make_smile()
    config = FeatureExtractionConfig(opensmile_version=opensmile.__version__)
    X: List[np.ndarray] = []
    kept: List[dict] = []
    feature_names: List[str] = []
    t0 = time.time()
    nan_features = 0

    for row in rows:
        values, columns, info = extract_session_features(row, smile=smile, config=config)
        if values is None:
            config.excluded_sessions.append(
                {
                    "participant_key": row["participant_key"],
                    "topic": row["topic"],
                    "audio_path": row["audio_path"],
                    "reason": info.get("reason"),
                }
            )
            continue
        if not feature_names:
            feature_names = columns
        elif columns != feature_names:
            raise AssertionError("openSMILE feature layout changed mid-corpus")
        nan_features += int(info["n_nan_features"])
        X.append(values)
        kept.append({k: row[k] for k in (
            "participant_key", "participant_label", "topic", "split",
            "audio_path", "audio_duration_sec",
        )})

    elapsed = time.time() - t0
    stats = {
        "sessions_requested": len(rows),
        "sessions_extracted": len(kept),
        "sessions_excluded": len(config.excluded_sessions),
        "nan_features_before_imputation": nan_features,
        "extraction_seconds": round(elapsed, 2),
        "seconds_per_session": round(elapsed / max(len(rows), 1), 3),
        "cached": False,
    }
    X_arr = np.vstack(X) if X else np.empty((0, 0))
    y_arr = np.asarray([r["participant_label"] for r in kept], dtype=int)

    if cache_path is not None and len(kept):
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X_arr, y=y_arr)
        write_json(
            Path(str(cache_path) + ".json"),
            {
                "rows": kept,
                "feature_names": feature_names,
                "config": config.to_dict(),
                "stats": stats,
            },
        )
    return X_arr, y_arr, kept, feature_names, config, stats


def _fit_preprocess(X_train: np.ndarray):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    ).fit(X_train)


def _grid() -> List[dict]:
    return [
        {"C": C, "class_weight": cw}
        for C, cw in itertools.product((0.01, 0.1, 1.0, 10.0), (None, "balanced"))
    ]


def _fit_lr(X: np.ndarray, y: Sequence[int], C: float, class_weight):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=C, class_weight=class_weight, max_iter=5000, solver="liblinear", random_state=0
    )
    clf.fit(X, list(y))
    return clf


def _participant_auc(rows: Sequence[dict], probs: Sequence[float]) -> float:
    agg = aggregate_by_participant(
        [r["participant_key"] for r in rows],
        [r["participant_label"] for r in rows],
        probs,
    )
    m = metric_bundle(agg["labels"], agg["probs"])
    return float(m["roc_auc"]) if m["roc_auc"] is not None else 0.0


def run_voice_egemaps(
    seed: int = 42,
    n_boot: int = 2000,
    write: bool = True,
    reuse_cache: bool = True,
) -> Dict[str, object]:
    """Full V1 experiment on the participant-level split."""
    t_start = time.time()

    def rows_for(split: str) -> List[dict]:
        # All 486 sessions are offered to the extractor; sessions the Phase 1
        # audit flagged unusable are excluded BY the extractor and recorded in
        # config.feature_extraction.excluded_sessions with a reason.
        return load_sessions(split, modality="audio", require_usable_audio=False)

    cache = ARTIFACT_DIR / "egemaps_features.npz"
    all_rows = rows_for("train") + rows_for("validation") + rows_for("test")
    X, _y, kept, feature_names, fe_config, ex_stats = extract_corpus(
        all_rows, cache_path=cache, reuse_cache=reuse_cache
    )
    if X.shape[0] != len(kept):
        raise AssertionError("feature matrix / row list mismatch")

    idx = {s: [] for s in ("train", "validation", "test")}
    for i, r in enumerate(kept):
        idx[r["split"]].append(i)
    for s, ids in idx.items():
        if not ids:
            raise RuntimeError(f"no usable audio sessions in split {s!r}")

    train_rows = [kept[i] for i in idx["train"]]
    val_rows = [kept[i] for i in idx["validation"]]
    test_rows = [kept[i] for i in idx["test"]]
    X_tr, X_va, X_te = X[idx["train"]], X[idx["validation"]], X[idx["test"]]

    pre = _fit_preprocess(X_tr)
    Z_tr, Z_va = pre.transform(X_tr), pre.transform(X_va)

    trials: List[dict] = []
    best, best_key = None, (-1.0, -1.0)
    for cfg in _grid():
        clf = _fit_lr(Z_tr, [r["participant_label"] for r in train_rows], cfg["C"], cfg["class_weight"])
        probs = clf.predict_proba(Z_va)[:, 1]
        agg = aggregate_by_participant(
            [r["participant_key"] for r in val_rows],
            [r["participant_label"] for r in val_rows],
            probs,
        )
        pm = metric_bundle(agg["labels"], agg["probs"])
        auc = float(pm["roc_auc"]) if pm["roc_auc"] is not None else 0.0
        f1 = float(pm["f1"]) if pm["f1"] is not None else 0.0
        trials.append({**cfg, "val_participant_roc_auc": auc, "val_participant_f1": f1})
        if (auc, f1) > best_key:
            best_key, best = (auc, f1), {**cfg, "val_participant_roc_auc": auc, "val_participant_f1": f1}
    if best is None:
        raise RuntimeError("empty candidate grid")

    clf_val = _fit_lr(Z_tr, [r["participant_label"] for r in train_rows], best["C"], best["class_weight"])
    val_probs = clf_val.predict_proba(Z_va)[:, 1]
    val_eval = evaluate_predictions(val_rows, val_probs, n_boot=n_boot, seed=seed)

    # Final refit: train + validation (scaler + imputer refitted on both).
    X_refit = np.vstack([X_tr, X_va])
    refit_rows = train_rows + val_rows
    pre_final = _fit_preprocess(X_refit)
    clf = _fit_lr(
        pre_final.transform(X_refit),
        [r["participant_label"] for r in refit_rows],
        best["C"],
        best["class_weight"],
    )
    t_infer = time.time()
    test_probs = clf.predict_proba(pre_final.transform(X_te))[:, 1]
    inference_seconds = round(time.time() - t_infer, 4)
    test_eval = evaluate_predictions(test_rows, test_probs, n_boot=n_boot, seed=seed)

    checkpoint_paths: List[Path] = []
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pre_final, ARTIFACT_DIR / "preprocessing.joblib")
        joblib.dump(clf, ARTIFACT_DIR / "classifier.joblib")
        checkpoint_paths = [
            ARTIFACT_DIR / "preprocessing.joblib",
            ARTIFACT_DIR / "classifier.joblib",
        ]
        write_json(
            ARTIFACT_DIR / "test_predictions.json",
            {
                "participant_key": [r["participant_key"] for r in test_rows],
                "topic": [r["topic"] for r in test_rows],
                "label": [r["participant_label"] for r in test_rows],
                "probability": [round(float(p), 6) for p in test_probs],
                "model_version": MODEL_VERSION,
            },
        )

    config = {
        "feature_extraction": fe_config.to_dict(),
        "n_features": len(feature_names),
        "feature_names_sha256_of_order": __import__("hashlib").sha256(
            ",".join(feature_names).encode()
        ).hexdigest(),
        "preprocessing": "SimpleImputer(median) + StandardScaler, fitted on train only (refit on train+validation for the final model)",
        "classifier": "LogisticRegression(liblinear)",
        "selected": best,
        "grid": trials,
        "split_sessions": {s: len(ids) for s, ids in idx.items()},
    }
    metadata = experiment_metadata(
        experiment_id=EXPERIMENT_ID,
        model_version=MODEL_VERSION,
        modality="voice",
        config=config,
        training={
            "seed": seed,
            "extraction": ex_stats,
            "fit_seconds": round(time.time() - t_start, 2),
            "inference_seconds_test": inference_seconds,
            "refit_on_train_plus_validation": True,
            "selection_metric": "validation participant-level ROC-AUC",
            "session_counts": {s: len(ids) for s, ids in idx.items()},
            "excluded_sessions": fe_config.excluded_sessions,
        },
        validation_metrics=val_eval,
        test_metrics=test_eval,
        checkpoint_paths=checkpoint_paths,
        notes=[
            "eGeMAPS functionals are utterance-level; no temporal modelling.",
            "Sessions excluded for duration/NaN reasons are listed explicitly in config.feature_extraction.excluded_sessions.",
            "Deterministic model: no stochastic component beyond the bootstrap seed.",
        ],
    )
    if write:
        write_json(ARTIFACT_DIR / "metadata.json", metadata)
        write_json(RESULT_PATH, metadata)
    return metadata
