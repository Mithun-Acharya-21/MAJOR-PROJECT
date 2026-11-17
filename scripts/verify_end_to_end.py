"""Phase 24 — end-to-end runtime verification of all fixes."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np

from backend.app.session_manager import screening_session_manager
from backend.app.services.adaptive_question_service import (
    MAX_QUESTIONS, select_next_question, should_stop, get_progress,
)
from backend.app.services.response_analysis_service import analyze_response
from backend.app.services.screening_service import complete_screening
from backend.app.services.fusion_service import SCREENING_WEIGHTS, fuse_multimodal
from backend.app.services.text_service import analyze_text, detect_script
from backend.app.models.text_classifier import get_text_classifier
from backend.app.recommendations import UserAssessmentProfile, generate_personalized_recommendations

SEP = "=" * 74
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

# ---------------------------------------------------------------------------
print(SEP + "\n1. ADAPTIVE STOPPING (Phase 13)\n" + SEP)

# High-information user
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
stopped_at, stop_reason = None, None
for i, a in enumerate(answers, 1):
    q = select_next_question(s)
    if q is None:
        stopped_at, stop_reason = i - 1, "question_bank_exhausted"
        break
    s.record_question(q)
    s.record_response(a, analyze_response(a))
    stop, reason = should_stop(s)
    if stop:
        stopped_at, stop_reason = i, reason
        break
print(f"  High-info user: stopped at {stopped_at} questions — reason: {stop_reason}")
check("High-info user stops before MAX with a sufficient-information reason",
      stopped_at is not None and stopped_at < MAX_QUESTIONS and
      stop_reason in ("sufficient_information", "low_uncertainty_gaps_resolved"),
      f"{stopped_at} questions, reason={stop_reason}")

# Ambiguous user
s2 = screening_session_manager.create()
ambiguous = "I'm not sure really, maybe it's okay, I guess things are somewhat fine."
amb_stopped = None
for i in range(1, MAX_QUESTIONS + 1):
    q = select_next_question(s2)
    if q is None:
        amb_stopped = (i - 1, "bank_exhausted")
        break
    s2.record_question(q)
    s2.record_response(ambiguous, analyze_response(ambiguous))
    stop, reason = should_stop(s2)
    if stop:
        amb_stopped = (i, reason)
        break
print(f"  Ambiguous user: stopped at {amb_stopped[0] if amb_stopped else MAX_QUESTIONS} questions — reason: {amb_stopped[1] if amb_stopped else 'never stopped (max)'}")
check("Ambiguous user is NOT stopped by sufficient-information early",
      amb_stopped is None or amb_stopped[0] >= 8,
      f"stopped at {amb_stopped}")

check("Question counts differ (adaptive behavior)",
      stopped_at != (amb_stopped[0] if amb_stopped else MAX_QUESTIONS))

# ---------------------------------------------------------------------------
print("\n" + SEP + "\n2. TRAINED TEXT MODEL (Phases 4/5)\n" + SEP)
clf = get_text_classifier()
check("Trained checkpoint loads", clf is not None and clf.available)
if clf:
    p_neg = clf.predict("今天心情不错，和朋友一起出去吃饭散步，感觉挺好的")
    p_pos = clf.predict("我每天都觉得很绝望，睡不着，无法工作，什么都不想做")
    print(f"  benign text  → P={p_neg['probability']:.3f} (logit {p_neg['logit']:.3f})")
    print(f"  distress text→ P={p_pos['probability']:.3f} (logit {p_pos['logit']:.3f})")
    check("Probabilities differ between inputs (no constant logit)",
          p_neg["probability"] != p_pos["probability"])

# ---------------------------------------------------------------------------
print("\n" + SEP + "\n3. TEXT SERVICE HONESTY (Phases 4/12)\n" + SEP)
en = analyze_text("I feel very low and hopeless lately.")
check("English input → labeled lexicon proxy (not fake model probability)",
      en["method"] == "lexicon_proxy" and en["model_version"] is None,
      f"method={en['method']}, score_type={en['score_type']}")
zh = analyze_text("我最近情绪很低落，觉得很绝望，每天都很累")
check("Chinese input → trained model probability",
      zh["method"] == "trained_model" and zh["score_type"] == "trained_model_probability",
      f"raw={zh.get('raw_probability')}%, calibrated={zh.get('calibrated_probability')}% "
      f"({zh.get('calibration_method')})")

# ---------------------------------------------------------------------------
print("\n" + SEP + "\n4. FUSION HONESTY (Phase 11)\n" + SEP)
fused = fuse_multimodal({
    "text": {"probability": 80.0, "confidence": 80.0, "signal_type": "trained_model_probability"},
    "voice": {"probability": 30.0, "confidence": 60.0, "signal_type": "acoustic_feature_score"},
}, SCREENING_WEIGHTS)
check("Fusion labels signal types + validated modalities",
      fused["signal_types"]["text"] == "trained_model_probability" and
      fused["validated_modalities"] == ["text"] and
      fused["fusion_type"] == "experimental_heuristic_fusion")
check("Missing modalities excluded, weights renormalized",
      set(fused["fusion_weights"]) == {"text", "voice"} and
      abs(sum(fused["fusion_weights"].values()) - 1.0) < 1e-9,
      f"weights={fused['fusion_weights']}")

# ---------------------------------------------------------------------------
print("\n" + SEP + "\n5. COMPLETE SCREENING PIPELINE (multimodal)\n" + SEP)
s3 = screening_session_manager.create()
for i in range(8):
    q = select_next_question(s3)
    s3.record_question(q)
    s3.record_response(answers[i % len(answers)], analyze_response(answers[i % len(answers)]))

# synthetic photo (gray) → proxy path
buf = io.BytesIO()
Image.fromarray(np.full((64, 64, 3), 110, dtype=np.uint8)).save(buf, format="PNG")

class FakeState:
    loaded = False
    text_branch = None
    vision_branch = None
    tokenizer = None

result = complete_screening(s3, FakeState(), photo_bytes=buf.getvalue())
sr = result["screening_result"]
print(f"  category: {sr['category']}")
print(f"  probability: {sr['probability']} | uncertainty: {sr['uncertainty']} ({sr['uncertainty_label']})")
print(f"  modalities: {sr['modalities_used']}")
print(f"  signal_types: {sr['signal_types']}")
print(f"  fusion_type: {sr['fusion_type']}")
print(f"  calibration: {sr['calibration_method']}" +
      (f" | warning: {sr['calibration_warning'][:80]}..." if sr.get('calibration_warning') else ""))
check("Fusion labeled experimental + interpretation surfaced",
      sr["fusion_type"] == "experimental_heuristic_fusion" and sr.get("fusion_interpretation"))
check("Honest disclaimer present",
      "not clinically validated" in result["disclaimer"])

# ---------------------------------------------------------------------------
print("\n" + SEP + "\n6. EXPLAINABILITY (Phases 17/18)\n" + SEP)
from backend.app.explainability.lime_text import explain_text_with_model
feats = explain_text_with_model("我最近情绪很低落，觉得很绝望", clf.predict_proba_batch, num_samples=20)
check("LIME runs on the actual trained predict function",
      isinstance(feats, list) and all(str(f["feature"]) in "我最近情绪很低落，觉得很绝望" for f in feats),
      f"{len(feats)} features attributed")
photo_individual = sr["fusion"]["individual_results"]["photo"]
check("Photo signal labeled as auxiliary/proxy (no depression-probability claim)",
      photo_individual.get("signal_type") == "proxy_heuristic",
      f"signal_type={photo_individual.get('signal_type')}")

# ---------------------------------------------------------------------------
print("\n" + SEP + "\n7. RECOMMENDATIONS (regression)\n" + SEP)
def dims(spec):
    return {k: {"severity": sv, "confidence": c} for k, (sv, c) in spec.items()}
PA = UserAssessmentProfile(overall_screening_level="elevated",
    dimensions=dims({"sleep": (3.0, 0.9), "energy": (2.8, 0.85), "concentration": (2.0, 0.8)}),
    functional_impact="moderate", duration="several weeks", uncertainty=0.2)
PB = UserAssessmentProfile(overall_screening_level="elevated",
    dimensions=dims({"social_withdrawal": (3.0, 0.9), "loss_of_interest": (2.8, 0.85), "mood": (2.0, 0.8)}),
    functional_impact="moderate", duration="several weeks", uncertainty=0.2)
ra = [r["id"] for r in generate_personalized_recommendations(PA)]
rb = [r["id"] for r in generate_personalized_recommendations(PB)]
check("Profiles A/B receive different recommendations", set(ra) != set(rb), f"A={ra[:3]}... B={rb[:3]}...")

# ---------------------------------------------------------------------------
print("\n" + SEP + "\n8. SAFETY PATHWAY (regression)\n" + SEP)
s4 = screening_session_manager.create()
s4.record_question({"id": "q1", "text": "How have you been feeling?"})
s4.record_response("I want to end my life.", analyze_response("I want to end my life."))
crisis = complete_screening(s4, FakeState())
check("Crisis override intact",
      crisis["screening_result"].get("safety_override") is True and
      all(r["category"] == "crisis_support" for r in crisis["recommendations"]))

# ---------------------------------------------------------------------------
passed = sum(1 for _, ok, _ in results if ok)
print("\n" + SEP)
print(f"END-TO-END VERIFICATION: {passed}/{len(results)} checks passed")
print(SEP)
sys.exit(0 if passed == len(results) else 1)
