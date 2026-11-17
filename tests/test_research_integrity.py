"""
Research-integrity test suite (Phase 23).

Covers: DATA (subject-level split, overlap, class distribution, duplicates),
MODEL (classification head, non-constant logits, checkpoint load),
CALIBRATION (artifact, validation-only fitting, effect), ADAPTIVE (min/max,
sufficient-information stopping, ambiguity, bank exhaustion), EXPLAINABILITY
(LIME uses the actual model, no fabricated Grad-CAM), RECOMMENDATIONS,
SAFETY, FUSION (missing modality, honesty labels).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "eatd_splits" / "eatd_subject_splits.csv"
CALIBRATION_JSON = PROJECT_ROOT / "models" / "calibration.json"
TEXT_MODEL_DIR = PROJECT_ROOT / "models" / "final" / "text_model"
FER_REPORT = PROJECT_ROOT / "data" / "fer2013" / "validation_report.json"


# ===========================================================================
# DATA — subject-level splits (Phase 2)
# ===========================================================================

def _load_split_module():
    spec = importlib.util.spec_from_file_location(
        "create_eatd_splits", PROJECT_ROOT / "scripts" / "create_eatd_splits.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not SPLITS_CSV.exists(), reason="splits not generated")
def test_split_subject_level_zero_overlap():
    import csv
    rows = list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))
    per_split = {}
    for r in rows:
        per_split.setdefault(r["split"], set()).add(r["subject_uid"])
    assert per_split["train"] & per_split["validation"] == set()
    assert per_split["train"] & per_split["test"] == set()
    assert per_split["validation"] & per_split["test"] == set()


@pytest.mark.skipif(not SPLITS_CSV.exists(), reason="splits not generated")
def test_split_deterministic_same_seed():
    mod = _load_split_module()
    rows = mod.load_rows()
    subjects, _dropped = mod.build_subject_groups(rows)
    s1 = mod.make_split(subjects, seed=42, ratios=(0.7, 0.15, 0.15))
    s2 = mod.make_split(subjects, seed=42, ratios=(0.7, 0.15, 0.15))
    assert s1 == s2, "same seed must reproduce the same split"
    s3 = mod.make_split(subjects, seed=7, ratios=(0.7, 0.15, 0.15))
    assert s3 != s1 or True  # different seeds MAY differ; determinism is the hard requirement


@pytest.mark.skipif(not SPLITS_CSV.exists(), reason="splits not generated")
def test_split_class_distribution_both_classes_per_split():
    import csv
    rows = list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))
    for split in ("train", "validation", "test"):
        labels = {r["label"] for r in rows if r["split"] == split}
        assert labels == {"0", "1"}, f"{split} must contain both classes, got {labels}"


@pytest.mark.skipif(not SPLITS_CSV.exists(), reason="splits not generated")
def test_split_no_duplicate_texts_within_subject():
    import csv
    rows = list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject_uid"], set()).add(r["text"])
    for uid, texts in by_subject.items():
        assert len(texts) == len([r for r in rows if r["subject_uid"] == uid]), (
            f"subject {uid} contains duplicate texts"
        )


@pytest.mark.skipif(not SPLITS_CSV.exists(), reason="splits not generated")
def test_split_report_matches_csv():
    report = json.loads((SPLITS_CSV.parent / "split_report.json").read_text(encoding="utf-8"))
    assert report["overlap_verification"]["all_zero"] is True
    assert report["subject_level_split"] is True
    import csv
    rows = list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))
    assert report["totals"]["samples"] == len(rows)


# ===========================================================================
# DATA — FER-2013 honesty (Phase 7/8)
# ===========================================================================

def test_fer2013_report_marks_incompleteness():
    if not FER_REPORT.exists():
        pytest.skip("FER-2013 validation report not generated")
    rep = json.loads(FER_REPORT.read_text(encoding="utf-8"))
    missing = rep["splits"]["train"]["missing_classes"]
    if missing:
        assert rep["usable_for_seven_class_training"] is False
        assert "MUST NOT" in rep["integrity_statement"]
    else:
        assert rep["usable_for_seven_class_training"] is True


# ===========================================================================
# MODEL — classification head (Phase 4/5)
# ===========================================================================

def test_text_branch_has_classification_head():
    import torch
    from backend.app.models.text_branch import TextBranch

    model = TextBranch(use_classifier_head=True)
    assert hasattr(model, "classifier")
    final = model.classifier[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.out_features == 1, "head must output a single logit"
    # No constant/frozen output: the head has learnable parameters
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_trained_classifier_probabilities_vary_with_input():
    from backend.app.models.text_classifier import get_text_classifier

    clf = get_text_classifier()
    if clf is None:
        pytest.skip("no trained text model checkpoint")
    p1 = clf.predict("我最近情绪很低落，每天都很沮丧，无法集中精力工作")
    p2 = clf.predict("今天天气很好，我和朋友一起出去吃饭，心情不错")
    assert 0.0 <= p1["probability"] <= 1.0
    assert 0.0 <= p2["probability"] <= 1.0
    assert p1["probability"] != p2["probability"], (
        "trained model must produce different probabilities for different inputs"
    )


def test_trained_classifier_logits_not_hardcoded():
    import torch
    from backend.app.models.text_classifier import get_text_classifier

    clf = get_text_classifier()
    if clf is None:
        pytest.skip("no trained text model checkpoint")
    texts = ["我每天都觉得很累，睡不着觉", "我和家人度过了一个愉快的周末"]
    probs = clf.predict_proba_batch(texts)
    assert probs.shape == (2, 2)
    assert not torch.allclose(
        torch.tensor(probs[:, 1]), torch.ones(2) * probs[0, 1]
    ), "logits must not be constant across inputs"


def test_trained_model_metadata_exists():
    meta_path = TEXT_MODEL_DIR / "text_model_metadata.json"
    if not meta_path.exists():
        pytest.skip("no trained text model metadata")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("model_version", "dataset_version", "training_seed",
                "training_date", "validation_metrics", "architecture",
                "preprocessing_version"):
        assert key in meta, f"metadata missing required field: {key}"


# ===========================================================================
# CALIBRATION (Phase 14/15)
# ===========================================================================

def test_calibration_artifact_exists_and_complete():
    if not CALIBRATION_JSON.exists():
        pytest.skip("calibration artifact not fitted")
    art = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    for key in ("method", "fitted_at", "model_version", "dataset_version",
                "validation_metrics", "fit_split"):
        assert key in art, f"calibration artifact missing: {key}"
    assert art["fit_split"] == "validation"
    assert art["n_samples"] > 0


def test_calibration_fitted_on_validation_not_test():
    """The fit input must be the validation-predictions file, not test data."""
    if not CALIBRATION_JSON.exists():
        pytest.skip("calibration artifact not fitted")
    art = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    assert "validation_predictions" in art.get("fit_input", ""), (
        "calibration must be fitted on validation-split predictions"
    )


def test_calibrated_probabilities_differ_from_raw():
    from backend.app.services.calibration_service import load_calibration_model

    cal = load_calibration_model()
    if cal is None or cal.method == "identity":
        pytest.skip("no non-identity calibration fitted")
    changed = sum(
        cal.transform(p) != p for p in (0.1, 0.3, 0.5, 0.7, 0.9)
    )
    assert changed >= 1, "a fitted calibrator must change at least some probabilities"


def test_identity_calibration_passthrough():
    from backend.app.services.calibration_service import CalibrationModel

    cal = CalibrationModel("identity")
    assert cal.transform(0.42) == pytest.approx(0.42)


# ===========================================================================
# ADAPTIVE (Phase 13)
# ===========================================================================

def _interview(session, answers):
    from backend.app.services.adaptive_question_service import select_next_question
    from backend.app.services.response_analysis_service import analyze_response

    for a in answers:
        q = select_next_question(session)
        if q is None:
            return "bank_exhausted"
        session.record_question(q)
        session.record_response(a, analyze_response(a))
    return "answers_exhausted"


def test_adaptive_min_and_max_question_counts():
    from backend.app.session_manager import screening_session_manager
    from backend.app.services.adaptive_question_service import (
        MAX_QUESTIONS, MIN_QUESTIONS, should_stop,
    )

    s = screening_session_manager.create()
    stop, reason = should_stop(s)
    assert not stop and reason == "min_questions_not_reached"
    for _ in range(MAX_QUESTIONS):
        q_ans = "I feel low and tired most days for weeks now."
        from backend.app.services.adaptive_question_service import select_next_question
        q = select_next_question(s)
        if q is None:
            break
        s.record_question(q)
        s.record_response(q_ans, __import__(
            "backend.app.services.response_analysis_service", fromlist=["analyze_response"]
        ).analyze_response(q_ans))
    stop, reason = should_stop(s)
    assert stop and reason == "max_questions_reached"


def test_adaptive_sufficient_information_stops_before_max():
    """High-information user MUST stop via a sufficient-information reason before MAX."""
    from backend.app.session_manager import screening_session_manager
    from backend.app.services.adaptive_question_service import (
        MAX_QUESTIONS, should_stop,
    )

    s = screening_session_manager.create()
    answers = [
        "I've been feeling really low and hopeless most days.",
        "I've lost interest in almost everything I used to enjoy.",
        "I can't sleep — I'm awake all night, most days.",
        "I'm exhausted all the time, no energy at all.",
        "It has been like this for weeks now, happens every day.",
        "I can't concentrate on my work, I'm falling behind at everything.",
        "I've been avoiding people, I don't see anyone anymore.",
        "Everything piles up, daily tasks are impossible.",
        "My appetite is gone, I stopped eating.",
        "I feel worthless, like a complete failure.",
        "I can't get out of bed, physically slowed down.",
        "I feel guilty about everything, it's all my fault.",
        "My mood is empty and down, I feel sad all the time.",
        "I'm unmotivated and bored of everything lately.",
    ]
    outcome = None
    for i, a in enumerate(answers, start=1):
        outcome = _interview(s, [a])
        if outcome == "bank_exhausted":
            break
        stop, reason = should_stop(s)
        if stop:
            assert i < MAX_QUESTIONS, "must stop strictly before MAX"
            assert reason in ("sufficient_information", "low_uncertainty_gaps_resolved"), (
                f"expected a sufficient-information stop, got: {reason}"
            )
            assert len(s.responses) >= 8
            return
    pytest.fail(
        f"sufficient-information stopping never triggered (outcome={outcome}, "
        f"answers={len(s.responses)})"
    )


def test_adaptive_ambiguous_user_continues():
    """An ambiguous user must NOT receive a sufficient-information stop early."""
    from backend.app.session_manager import screening_session_manager
    from backend.app.services.adaptive_question_service import should_stop

    s = screening_session_manager.create()
    ambiguous = "I'm not sure really, maybe it's okay, I guess things are somewhat fine."
    for _ in range(8):
        _interview(s, [ambiguous])
        stop, reason = should_stop(s)
        if len(s.responses) >= 8:
            assert not stop or reason == "max_questions_reached", (
                f"ambiguous user must not trigger sufficient-information stop, got {reason}"
            )


def test_adaptive_bank_exhaustion_is_graceful():
    from backend.app.session_manager import screening_session_manager
    from backend.app.services.adaptive_question_service import select_next_question

    s = screening_session_manager.create()
    # Repeatedly answer informatively until the bank is exhausted
    exhausted = False
    for _ in range(30):
        q = select_next_question(s)
        if q is None:
            exhausted = True
            break
        s.record_question(q)
        s.record_response(
            "I feel low and tired most days for weeks now, and it is getting worse.",
            __import__(
                "backend.app.services.response_analysis_service", fromlist=["analyze_response"]
            ).analyze_response(
                "I feel low and tired most days for weeks now, and it is getting worse."
            ),
        )
    assert exhausted, "question bank must eventually exhaust gracefully (None, not crash)"


def test_single_coverage_source_of_truth():
    """session.dimension_coverage must equal compute_dimension_coverage(features)."""
    from backend.app.session_manager import screening_session_manager
    from backend.app.services.adaptive_question_service import compute_dimension_coverage

    s = screening_session_manager.create()
    _interview(s, ["I can't sleep most nights.", "I feel low most days."])
    assert s.dimension_coverage == compute_dimension_coverage(s.features)


# ===========================================================================
# EXPLAINABILITY (Phase 17/18)
# ===========================================================================

def test_lime_calls_the_actual_prediction_function():
    """explain_text_with_model must route ALL texts through the given predict_fn."""
    import numpy as np
    from backend.app.explainability.lime_text import explain_text_with_model

    calls = {"n": 0, "texts": []}

    def fake_predict_fn(texts):
        calls["n"] += 1
        calls["texts"].extend(texts)
        arr = np.array([[0.4, 0.6]] * len(texts))
        # Make the function genuinely text-dependent
        for i, t in enumerate(texts):
            arr[i, 1] = 0.9 if "hopeless" in t else 0.2
        return arr

    out = explain_text_with_model("I feel hopeless today", fake_predict_fn,
                                  num_samples=20)
    assert calls["n"] > 0, "LIME must call the provided prediction function"
    assert any("hopeless" in t for t in calls["texts"]), (
        "LIME must perturb and rescore the actual input text"
    )
    assert isinstance(out, list)


def test_lime_uses_trained_model_predict_fn():
    from backend.app.models.text_classifier import get_text_classifier
    from backend.app.explainability.lime_text import explain_text_with_model

    clf = get_text_classifier()
    if clf is None:
        pytest.skip("no trained text model checkpoint")
    feats = explain_text_with_model(
        "我最近情绪很低落，觉得很绝望", clf.predict_proba_batch, num_samples=20
    )
    assert isinstance(feats, list)
    # Attribution features must come from the actual input text (LIME may
    # merge adjacent characters into word-like features, e.g. 觉得很绝望).
    text = "我最近情绪很低落，觉得很绝望"
    for f in feats:
        assert str(f["feature"]) in text, (
            f"LIME feature {f['feature']!r} is not part of the input text"
        )


def test_photo_service_has_no_fake_depression_gradcam():
    """No Grad-CAM may be presented as explaining a depression prediction."""
    from PIL import Image
    import io
    import numpy as np
    from backend.app.services.photo_service import analyze_photo

    # Synthetic gray image (no real face — triggers proxy path, which is fine)
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(buf, format="PNG")
    result = analyze_photo(buf.getvalue())
    assert result.get("signal_type") in ("auxiliary_expression_score", "proxy_heuristic")
    assert result["explanation"].get("gradcam", None) is None or (
        "no legitimate target" in str(result["explanation"].get("gradcam_note", ""))
    )
    assert "visual_expression_score" in result or result["signal_type"] == "proxy_heuristic"


def test_text_service_labels_fallback_clearly():
    from backend.app.services.text_service import analyze_text

    r = analyze_text("I feel very low and hopeless lately.")
    assert r["method"] == "lexicon_proxy"
    assert r["score_type"] == "rule_based_proxy"
    assert "NOT a trained model probability" in r["explanation"]["note"]


# ===========================================================================
# QUESTIONNAIRE (Phase 12)
# ===========================================================================

def test_questionnaire_result_labeled_as_research_score():
    from backend.app.services.screening_service import build_questionnaire_result

    features = {
        "sleep": {"severity": 3.0, "confidence": 0.9},
        "energy": {"severity": 2.5, "confidence": 0.8},
    }
    res = build_questionnaire_result(features, question_count=10)
    assert res["signal_type"] == "rule_based_screening_score"
    assert "NOT a trained ML probability" in res["explanation"]["mapping"]


# ===========================================================================
# FUSION (Phase 11) — missing modality + honesty labels
# ===========================================================================

def test_fusion_missing_modalities_and_signal_labels():
    from backend.app.services.fusion_service import SCREENING_WEIGHTS, fuse_multimodal

    text = {"probability": 80.0, "confidence": 80.0, "signal_type": "trained_model_probability"}
    voice = {"probability": 30.0, "confidence": 60.0, "signal_type": "acoustic_feature_score"}
    fused = fuse_multimodal({"text": text, "voice": voice}, SCREENING_WEIGHTS)
    assert set(fused["fusion_weights"].keys()) == {"text", "voice"}
    assert abs(sum(fused["fusion_weights"].values()) - 1.0) < 1e-9
    assert fused["signal_types"]["text"] == "trained_model_probability"
    assert fused["signal_types"]["voice"] == "acoustic_feature_score"
    assert fused["validated_modalities"] == ["text"]
    assert fused["fusion_type"] == "experimental_heuristic_fusion"
    assert "NOT clinically validated" in fused["explanation"]["interpretation_note"]


def test_fusion_no_fake_modality_inserted():
    from backend.app.services.fusion_service import SCREENING_WEIGHTS, fuse_multimodal

    text = {"probability": 65.0, "confidence": 70.0}
    fused = fuse_multimodal({"text": text}, SCREENING_WEIGHTS)
    assert abs(fused["probability"] - 65.0) < 0.11  # text-only → fused == text
    assert set(fused["fusion_weights"].keys()) == {"text"}


# ===========================================================================
# SAFETY + RECOMMENDATIONS (regression: must remain intact)
# ===========================================================================

def test_safety_pathway_remains_intact():
    from backend.app.session_manager import screening_session_manager
    from backend.app.services.response_analysis_service import analyze_response
    from backend.app.services.screening_service import complete_screening

    class _Fake:
        loaded = False
        text_branch = None
        vision_branch = None
        tokenizer = None

    s = screening_session_manager.create()
    s.record_question({"id": "q1", "text": "How have you been feeling?"})
    s.record_response("I want to end my life.", analyze_response("I want to end my life."))
    result = complete_screening(s, _Fake())
    assert result["screening_result"].get("safety_override") is True
    assert all(r["category"] == "crisis_support" for r in result["recommendations"])
    assert result["crisis_resources"]


def test_personalized_recommendations_still_differ_by_profile():
    from backend.app.recommendations import (
        UserAssessmentProfile,
        generate_personalized_recommendations,
    )

    def dims(spec):
        return {k: {"severity": s, "confidence": c} for k, (s, c) in spec.items()}

    A = UserAssessmentProfile(
        overall_screening_level="elevated",
        dimensions=dims({"sleep": (3.0, 0.9), "energy": (2.8, 0.85), "concentration": (2.0, 0.8)}),
        functional_impact="moderate", duration="several weeks", uncertainty=0.2)
    B = UserAssessmentProfile(
        overall_screening_level="elevated",
        dimensions=dims({"social_withdrawal": (3.0, 0.9), "loss_of_interest": (2.8, 0.85), "mood": (2.0, 0.8)}),
        functional_impact="moderate", duration="several weeks", uncertainty=0.2)
    ids_a = {r["id"] for r in generate_personalized_recommendations(A)}
    ids_b = {r["id"] for r in generate_personalized_recommendations(B)}
    assert ids_a != ids_b


# ===========================================================================
# PHASE 21 — no placeholder metrics are served
# ===========================================================================

def test_results_metrics_files_carry_real_provenance():
    text_metrics = PROJECT_ROOT / "results" / "metrics" / "text_metrics.json"
    if not text_metrics.exists():
        pytest.skip("text metrics not evaluated yet")
    data = json.loads(text_metrics.read_text(encoding="utf-8"))
    assert data.get("status") != "placeholder"
    assert data.get("split") == "test"
    assert data.get("samples") == data.get("subjects", 0) * 3 or data.get("samples") > 0
    assert "evaluated_at" in data and "model_version" in data


def test_baseline_metrics_labeled_as_literature():
    from backend.app.baselines import get_baseline_metrics

    base = get_baseline_metrics()
    for m in base["models"]:
        assert m.get("metric_source") == "literature"
        assert "NOT a result produced by this project" in m["description"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
