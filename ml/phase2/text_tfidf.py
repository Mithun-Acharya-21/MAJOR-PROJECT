"""
Experiment T1 — TF-IDF + Logistic Regression text baseline (EATD, Chinese).

Why this baseline:
    The EATD transcripts are Chinese interview speech. Whitespace tokenization
    is meaningless, so the feature space is an explicit union of
      * CJK character unigrams/bigrams (custom tokenizer: one token per Chinese
        character, whole Latin words, numbers), and
      * character n-grams (2-4, ``char_wb``), which capture Chinese morphology
        without a word segmenter (no jieba dependency).
    Class-weighted L2 logistic regression is the standard, probabilistic
    linear baseline for imbalanced text screening.

Experimental discipline:
    * The vectorizer is fitted on TRAIN text only (then train+validation for
      the final refit — never test).
    * (feature set, C, class_weight) is selected on VALIDATION participant-level
      ROC-AUC. The test split is transformed once, after selection.
    * Session probabilities are aggregated to participant level by mean.

Exports:
    CJK_TOKEN_PATTERN, cjk_tokenize, build_features, candidate_grid,
    fit_model, predict_rows, select_on_validation, run_text_tfidf
"""

from __future__ import annotations

import itertools
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from backend.app.models.text_branch import preprocess_text
from ml.phase2.dataset import load_sessions
from ml.phase2.evaluation import (
    aggregate_by_participant,
    evaluate_predictions,
    experiment_metadata,
    metric_bundle,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_ID = "eatd_text_tfidf_v1"
MODEL_VERSION = "text-eatd-tfidf-2.0.0"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "experiments" / EXPERIMENT_ID
RESULT_PATH = PROJECT_ROOT / "results" / "phase2" / "text" / f"{EXPERIMENT_ID}.json"

CJK_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[\u4e00-\u9fff]")


def cjk_tokenize(text: str) -> List[str]:
    """One token per Chinese character / Latin word / number."""
    return CJK_TOKEN_PATTERN.findall(preprocess_text(text))


def build_features(kind: str = "word_char"):
    """TF-IDF feature space: 'word', 'char' or the union 'word_char'."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    word = TfidfVectorizer(
        tokenizer=cjk_tokenize,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    if kind == "word":
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=False,
        ngram_range=(2, 4),
        min_df=2,
        sublinear_tf=True,
    )
    if kind == "char":
        return char
    if kind == "word_char":
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"unknown feature kind {kind!r}")


def candidate_grid(kind: str = "word_char") -> List[dict]:
    """Small, documented grid — 8 configurations, no test data involved."""
    return [
        {"kind": kind, "C": C, "class_weight": cw}
        for C, cw in itertools.product((0.1, 0.3, 1.0, 3.0), (None, "balanced"))
    ]


def fit_model(
    train_rows: Sequence[dict],
    kind: str = "word_char",
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
):
    """Fit vectorizer + logistic regression on the given rows only."""
    from sklearn.linear_model import LogisticRegression

    features = build_features(kind)
    X = features.fit_transform([r["text"] for r in train_rows])
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=5000,
        solver="liblinear",
        random_state=0,
    )
    clf.fit(X, [r["label"] for r in train_rows])
    return features, clf


def predict_rows(features, clf, rows: Sequence[dict]) -> np.ndarray:
    X = features.transform([r["text"] for r in rows])
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=float)


def _participant_auc(rows: Sequence[dict], probs: Sequence[float]) -> float:
    agg = aggregate_by_participant(
        [r["participant_key"] for r in rows],
        [r["participant_label"] for r in rows],
        probs,
    )
    m = metric_bundle(agg["labels"], agg["probs"])
    return float(m["roc_auc"]) if m["roc_auc"] is not None else 0.0


def select_on_validation(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    grid: Optional[List[dict]] = None,
) -> Tuple[dict, List[dict]]:
    """Grid search on validation participant ROC-AUC (tie-break: val F1)."""
    trials: List[dict] = []
    best: Optional[dict] = None
    best_key = (-1.0, -1.0)
    for cfg in grid or candidate_grid():
        t0 = time.time()
        features, clf = fit_model(
            train_rows, kind=cfg["kind"], C=cfg["C"], class_weight=cfg["class_weight"]
        )
        probs = predict_rows(features, clf, val_rows)
        y_val = [r["participant_label"] for r in val_rows]
        session_m = metric_bundle(y_val, probs)
        agg = aggregate_by_participant(
            [r["participant_key"] for r in val_rows], y_val, probs
        )
        part_m = metric_bundle(agg["labels"], agg["probs"])
        trial = {
            **cfg,
            "n_features": int(features.transform([train_rows[0]["text"]]).shape[1]),
            "val_session_roc_auc": session_m["roc_auc"],
            "val_participant_roc_auc": part_m["roc_auc"],
            "val_participant_f1": part_m["f1"],
            "val_participant_accuracy": part_m["accuracy"],
            "fit_seconds": round(time.time() - t0, 2),
        }
        trials.append(trial)
        key = (
            float(part_m["roc_auc"]) if part_m["roc_auc"] is not None else 0.0,
            float(part_m["f1"]) if part_m["f1"] is not None else 0.0,
        )
        if key > best_key:
            best_key = key
            best = trial
    if best is None:
        raise RuntimeError("empty candidate grid")
    return best, trials


def top_features(features, clf, k: int = 15) -> Dict[str, List[dict]]:
    """Most positive / most negative coefficients (interpretability only)."""
    names = np.asarray(features.get_feature_names_out(), dtype=object)
    coef = clf.coef_.ravel()
    order = np.argsort(coef)
    top_neg = [
        {"feature": str(names[i]), "weight": round(float(coef[i]), 4)}
        for i in order[:k]
    ]
    top_pos = [
        {"feature": str(names[i]), "weight": round(float(coef[i]), 4)}
        for i in order[-k:][::-1]
    ]
    return {"risk_increasing": top_pos, "risk_decreasing": top_neg}


def run_text_tfidf(
    seed: int = 42,
    n_boot: int = 2000,
    write: bool = True,
) -> Dict[str, object]:
    """Full T1 experiment: select on validation, refit, score the locked test set."""
    t_start = time.time()
    train_rows = load_sessions("train", modality="text")
    val_rows = load_sessions("validation", modality="text")
    test_rows = load_sessions("test", modality="text")

    train_participants = {r["participant_key"] for r in train_rows}
    val_participants = {r["participant_key"] for r in val_rows}
    test_participants = {r["participant_key"] for r in test_rows}
    overlap = (train_participants & val_participants) | (train_participants & test_participants) | (
        val_participants & test_participants
    )
    if overlap:
        raise AssertionError(f"participant leakage across splits: {sorted(overlap)[:3]}")

    best, trials = select_on_validation(train_rows, val_rows)

    # Validation report from the train-only model used for selection.
    features_val, clf_val = fit_model(
        train_rows, kind=best["kind"], C=best["C"], class_weight=best["class_weight"]
    )
    val_probs = predict_rows(features_val, clf_val, val_rows)
    val_eval = evaluate_predictions(val_rows, val_probs, n_boot=n_boot, seed=seed)

    # Final model: refit on train + validation (test still untouched).
    refit_rows = list(train_rows) + list(val_rows)
    features, clf = fit_model(
        refit_rows, kind=best["kind"], C=best["C"], class_weight=best["class_weight"]
    )
    t_infer = time.time()
    test_probs = predict_rows(features, clf, test_rows)
    inference_seconds = round(time.time() - t_infer, 4)
    test_eval = evaluate_predictions(test_rows, test_probs, n_boot=n_boot, seed=seed)

    checkpoint_paths: List[Path] = []
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(features, ARTIFACT_DIR / "features.joblib")
        joblib.dump(clf, ARTIFACT_DIR / "classifier.joblib")
        checkpoint_paths = [
            ARTIFACT_DIR / "features.joblib",
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
        "features": "FeatureUnion(TF-IDF word 1-2gram on CJK chars/Latin words, TF-IDF char_wb 2-4gram min_df=2)",
        "vectorizer_fit_on": "train only for selection; train+validation for the final model",
        "classifier": "LogisticRegression(solver=liblinear, max_iter=5000)",
        "selected": best,
        "grid": trials,
        "n_features": int(features.transform([train_rows[0]["text"]]).shape[1]),
        "top_coefficients": top_features(features, clf),
        "class_balance_train": {
            "sessions": len(train_rows),
            "positive_sessions": int(sum(r["label"] for r in train_rows)),
            "participants": len(train_participants),
            "positive_participants": len(
                {r["participant_key"] for r in train_rows if r["participant_label"] == 1}
            ),
        },
    }
    metadata = experiment_metadata(
        experiment_id=EXPERIMENT_ID,
        model_version=MODEL_VERSION,
        modality="text",
        config=config,
        training={
            "seed": seed,
            "fit_seconds": round(time.time() - t_start, 2),
            "inference_seconds_test": inference_seconds,
            "test_sessions": len(test_rows),
            "refit_on_train_plus_validation": True,
            "selection_metric": "validation participant-level ROC-AUC",
            "n_boot": n_boot,
        },
        validation_metrics=val_eval,
        test_metrics=test_eval,
        checkpoint_paths=checkpoint_paths,
        notes=[
            "Deterministic model: liblinear logistic regression has no stochastic component (seed affects bootstrap only).",
            "Aggregation for participant-level metrics: mean of the participant's session probabilities.",
            "Test split (24 participants, 8 positive) scored once, after selection.",
        ],
    )
    if write:
        write_json(ARTIFACT_DIR / "metadata.json", metadata)
        write_json(RESULT_PATH, metadata)
    return metadata
