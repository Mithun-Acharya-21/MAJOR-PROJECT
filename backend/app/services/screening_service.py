"""
Screening Service — Orchestrates the complete adaptive screening flow.

    Adaptive questions → structured features
        → available multimodal models (existing services, reused)
        → fusion (existing fusion_service, extended)
        → probability calibration (validated, optional)
        → uncertainty estimation
        → screening result (screening-oriented categories)
        → UserAssessmentProfile
        → personalized recommendations

Model failures degrade gracefully: the questionnaire signal is always
available (lexicon-based), so a screening result can still be produced,
with reduced confidence and an explicit ``model_failure`` note.

This service REUSES the existing modality services and fusion service —
no second prediction pipeline is created.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from backend.app.recommendations import (
    UserAssessmentProfile,
    generate_personalized_recommendations,
    get_crisis_response,
    _severity_label,
)
from backend.app.services.adaptive_question_service import (
    MIN_QUESTIONS,
    MAX_QUESTIONS,
    build_question_payload,
    compute_dimension_coverage,
    get_progress,
)
from backend.app.services.calibration_service import load_calibration_model
from backend.app.services.fusion_service import SCREENING_WEIGHTS, fuse_multimodal
from backend.app.services.uncertainty_service import (
    compute_uncertainty,
    track_history,
    uncertainty_breakdown,
    uncertainty_label,
)

logger = logging.getLogger("emotiondep.screening_service")

# Screening-oriented categories connected to the project's validated
# methodology (the existing low/moderate/high probability thresholds at
# 0.45/0.65, extended with a high band at 0.8). NEVER diagnostic labels.
SCREENING_CATEGORIES = [
    ("lower_indication", 0.0, 0.45, "Lower indication of depressive symptoms"),
    ("some_indication", 0.45, 0.65, "Some depressive symptoms indicated"),
    ("elevated_indication", 0.65, 0.80, "Elevated indication of depressive symptoms"),
    ("high_indication", 0.80, 1.01, "High indication of depressive symptoms"),
]

MODEL_VERSION = "emotiondep-screening-1.0.0"

# ---------------------------------------------------------------------------
# Four-modality output contract (spec §17)
# ---------------------------------------------------------------------------
# Every modality returns a standardized object so the frontend renders ONE
# canonical result. Proxy scores are NEVER relabeled as validated depression
# probabilities — ``signal_type`` carries the honest provenance label.

# Modality display order: text → face → live_face → voice (spec architecture).
MODALITY_CONTRACT_ORDER = ("text", "face", "live_face", "voice")

# Map internal service keys → contract modality names.
_CONTRACT_KEY_MAP = {
    "text": "text",
    "photo": "face",        # uploaded-photo branch = FACE DETECTION modality
    "face": "live_face",    # multi-frame webcam branch = LIVE FACE modality
    "voice": "voice",
}

# Human-facing signal-type labels per provenance (research honesty).
_SIGNAL_TYPE_LABELS = {
    "trained_model_probability": "trained text model",
    "trained_text_model": "trained text model",
    "auxiliary_expression_score": "expression-based auxiliary signal",
    "temporal_expression_proxy": "temporal expression proxy",
    "expression_proxy": "expression proxy",
    "acoustic_feature_score": "acoustic auxiliary research signal",
    "acoustic_proxy": "acoustic auxiliary research signal",
    "lexicon_proxy": "lexicon proxy (fallback heuristic)",
    "lexicon_proxy_plus_acoustic_feature_score": "lexicon + acoustic proxy",
    "proxy_heuristic": "image-statistics proxy (low evidential value)",
    "rule_based_screening_score": "questionnaire-derived research score",
    "questionnaire_derived_research_score": "questionnaire-derived research score",
}


def _modality_confidence_band(score: float, confidence: float) -> str:
    """Non-diagnostic descriptive band for a modality signal (0..1 inputs)."""
    effective = score * max(0.2, min(1.0, confidence))
    if effective >= 0.62:
        return "Elevated signal"
    if effective >= 0.45:
        return "Moderate auxiliary signal"
    return "Low signal"


def build_modality_contracts(
    modality_results: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Convert internal modality results into the standardized contract objects.

    Contract (spec §17):
        { modality, score, confidence, signal_type, signal_type_label,
          available, ...evidence }

    score/confidence are 0..1. Only AVAILABLE modalities produce entries here;
    absent modalities are listed with available=false in the response payload
    by the caller (so the frontend can render all four branches explicitly).
    """
    contracts: Dict[str, Dict[str, Any]] = {}
    for internal_key, result in modality_results.items():
        if not result:
            continue
        modality = _CONTRACT_KEY_MAP.get(internal_key, internal_key)
        if modality in contracts:
            # e.g. both photo and live-face frames present → keep live face as
            # the temporal branch and expose the photo under its own key below.
            modality = internal_key if internal_key == "photo" else modality
        score = float(result.get("probability", 50.0)) / 100.0
        confidence = float(result.get("confidence", 0.0)) / 100.0 if float(result.get("confidence", 0.0)) > 1 else float(result.get("confidence", 0.0))
        signal_type = str(result.get("signal_type") or "rule_based_screening_score")
        entry: Dict[str, Any] = {
            "modality": modality,
            "score": round(min(max(score, 0.0), 1.0), 4),
            "confidence": round(min(max(confidence, 0.0), 1.0), 4),
            "signal_type": signal_type,
            "signal_type_label": _SIGNAL_TYPE_LABELS.get(
                signal_type, signal_type.replace("_", " ")
            ),
            "available": True,
            "risk_level": result.get("risk_level"),
        }
        # Per-modality research evidence (labelled honestly per branch).
        if modality == "text":
            entry["evidence"] = {
                "method": result.get("method"),
                "model_version": result.get("model_version"),
                "top_features": result.get("explanation", {}).get("top_features", [])[:8],
            }
        elif modality in ("face", "photo"):
            features = result.get("features", {}) or {}
            entry["evidence"] = {
                "face_detected": features.get("face_detected"),
                "analysis_method": result.get("explanation", {}).get("method"),
                "dominant_emotion": features.get("dominant_emotion") or result.get("explanation", {}).get("dominant_emotion"),
                "dominant_expression": result.get("dominant_expression") or features.get("dominant_emotion"),
                "model_name": result.get("model_name", "MediaPipe FaceLandmarker (52 Blendshapes)"),
                "model_type": result.get("model_type", "Facial Expression & Landmark Detection (Auxiliary)"),
                "is_depression_model": False,
                "note": result.get("explanation", {}).get("note") or result.get("note"),
            }
        elif modality == "live_face":
            features = result.get("features", {}) or {}
            entry["evidence"] = {
                "frames_analyzed": features.get("frame_count"),
                "faces_detected": features.get("analyzed_count"),
                "dominant_expression": features.get("dominant_emotion") or result.get("dominant_expression"),
                "expression_distribution": features.get("expression_distribution"),
                "model_name": result.get("model_name", "MediaPipe FaceLandmarker (Multi-Frame Aggregator)"),
                "model_type": result.get("model_type", "Multi-Frame Expression Tracking (Auxiliary)"),
                "is_depression_model": False,
                "note": result.get("explanation", {}).get("note"),
            }
        elif modality == "voice":
            entry["evidence"] = {
                "acoustic_features": {
                    k: result.get("voice_features", {}).get(k)
                    for k in ("rms_energy", "speaking_rate", "silence_ratio", "pitch_mean", "pitch_variation", "duration_sec")
                },
                "transcript_length": len(str(result.get("transcript", ""))),
                "note": result.get("explanation", {}).get("score_interpretation"),
            }
        contracts[modality] = entry
    return contracts


def categorize_probability(probability: float) -> Dict[str, str]:
    """Map fused probability → screening-oriented category (never diagnostic)."""
    for key, lo, hi, label in SCREENING_CATEGORIES:
        if lo <= probability < hi:
            return {"key": key, "label": label}
    key, _, _, label = SCREENING_CATEGORIES[-1]
    return {"key": key, "label": label}


def build_questionnaire_result(
    features: Dict[str, Any],
    question_count: int,
) -> Dict[str, Any]:
    """Build the questionnaire modality signal from structured features.

    Phase 12: uses the SHARED rule-based mapping
    (adaptive_question_service.questionnaire_probability) and labels the
    output honestly as a questionnaire-derived research score — a documented
    hand-designed formula, NOT a trained ML probability.
    """
    from backend.app.services.adaptive_question_service import questionnaire_probability

    severity_dims = {d: info for d, info in features.items() if d not in ("duration", "frequency", "severity")}

    if not severity_dims:
        return {
            "probability": 50.0,
            "confidence": 20.0,
            "features": {"answered_dimensions": 0},
            "explanation": {
                "method": "structured_questionnaire",
                "note": "No structured features extracted",
            },
            "risk_level": "low",
            "signal_type": "rule_based_screening_score",
            "model_version": None,
        }

    mapping = questionnaire_probability(features)
    probability = mapping["probability"]
    mean_severity = mapping["mean_severity"]

    # Evidence completeness: more answered dimensions + higher confidence → higher confidence
    confidences = [float(info.get("confidence", 0.5)) for info in severity_dims.values()]
    completeness = min(1.0, len(severity_dims) / 5.0)
    mean_conf = sum(confidences) / len(confidences)
    confidence = 0.4 * completeness + 0.6 * mean_conf
    # Fewer questions than the minimum → penalize confidence
    if question_count < MIN_QUESTIONS:
        confidence *= 0.6 + 0.4 * question_count / MIN_QUESTIONS

    if probability >= 0.7:
        risk = "high"
    elif probability >= 0.5:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "probability": round(probability * 100, 1),
        "confidence": round(confidence * 100, 1),
        "features": {
            "answered_dimensions": len(severity_dims),
            "mean_severity": mean_severity,
            "question_count": question_count,
        },
        "explanation": {
            "method": "structured_questionnaire",
            "score_type": "questionnaire_derived_research_score",
            "mapping": (
                "rule-based: severity-weighted mean (0..3) → logistic mapping "
                "centered at 1.5 — a documented hand-designed formula, NOT a "
                "trained ML probability and NOT a validated clinical instrument"
            ),
            "mean_severity": mean_severity,
            "dimension_count": len(severity_dims),
        },
        "risk_level": risk,
        "signal_type": "rule_based_screening_score",
        "model_version": None,
    }


def _dimensions_for_profile(session: Any) -> Dict[str, Dict[str, float]]:
    """Session features → profile dimensions (severity + confidence, rounded)."""
    dims: Dict[str, Dict[str, float]] = {}
    for d, info in session.features.items():
        if d in ("duration", "frequency", "severity"):
            continue
        dims[d] = {
            "severity": round(float(info.get("severity", 0.0)), 2),
            "confidence": round(float(info.get("confidence", 0.0)), 2),
        }
    return dims


def _functional_impact(features: Dict[str, Any]) -> str:
    info = features.get("daily_functioning")
    if not info:
        return "low"
    sev = float(info.get("severity", 0.0))
    conf = float(info.get("confidence", 0.5))
    effective = sev * conf
    if effective >= 2.0:
        return "high"
    if effective >= 1.0:
        return "moderate"
    return "low"


def build_user_profile(
    session: Any,
    fused_probability: float,
    uncertainty: float,
) -> UserAssessmentProfile:
    """Build the UserAssessmentProfile from the completed screening session."""
    category = categorize_probability(fused_probability)
    level_map = {
        "lower_indication": "lower",
        "some_indication": "some",
        "elevated_indication": "elevated",
        "high_indication": "high",
    }
    duration_info = session.features.get("duration")
    duration = str(duration_info.get("value", "unknown")) if duration_info else "unknown"

    return UserAssessmentProfile(
        overall_screening_level=level_map.get(category["key"], "lower"),
        dimensions=_dimensions_for_profile(session),
        functional_impact=_functional_impact(session.features),
        duration=duration,
        uncertainty=uncertainty,
        modalities_used=list(session.modalities_used),
        user_context=dict(session.user_context),
        safety_concern=session.safety_flag,
    )


def build_research_trace(
    session: Any,
    modality_results: Dict[str, Any],
    fused: Dict[str, Any],
    canonical_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the research trace object (spec §31).

    Exposes the full computational chain for external evaluation:
        What did the user provide?
            → What did each model detect?
            → How were modalities combined?
            → Why was the final screening result produced?
            → Why were these suggestions selected?
    """
    lime_payload = canonical_result.get("explainability", {}).get("lime")
    return {
        "trace_version": "1.0",
        "session_id": getattr(session, "session_id", None),
        "text_evidence": {
            "responses": [
                {
                    "question_id": r.get("question_id"),
                    "text": r.get("text"),
                    "response_mode": r.get("analysis", {}).get("response_mode", "free_text"),
                    "option_id": r.get("analysis", {}).get("option_id"),
                    "extracted_dimensions": r.get("analysis", {}).get("dimensions", {}),
                }
                for r in getattr(session, "responses", [])
            ],
            "features": getattr(session, "features", {}),
            "text_model_output": modality_results.get("text"),
        },
        "face_evidence": modality_results.get("photo"),
        "live_face_evidence": modality_results.get("face"),
        "voice_evidence": modality_results.get("voice"),
        "fusion": {
            "weights": fused.get("fusion_weights"),
            "agreement": fused.get("explanation", {}).get("agreement_score"),
            "signal_types": fused.get("signal_types"),
            "validated_modalities": fused.get("validated_modalities"),
            "fusion_type": fused.get("fusion_type"),
            "interpretation": fused.get("explanation", {}).get("interpretation_note"),
            "individual_results": fused.get("individual_results"),
        },
        "final_result": {
            "score": canonical_result.get("fusion", {}).get("score"),
            "category": canonical_result.get("screening", {}).get("category"),
            "uncertainty": canonical_result.get("fusion", {}).get("uncertainty"),
            "profile": canonical_result.get("profile"),
        },
        "lime_explanation": lime_payload,
        "recommendations": canonical_result.get("suggestions", []),
    }


def _run_modality(
    modality: str,
    model_state: Any,
    *,
    text: Optional[str] = None,
    photo_bytes: Optional[bytes] = None,
    frame_bytes: Optional[List[bytes]] = None,
    audio_bytes: Optional[bytes] = None,
    session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run one existing modality service. Returns None when input is absent.

    Model failures are caught and return None — the fusion then renormalizes
    over remaining modalities (graceful degradation, no fake predictions).
    """
    try:
        if modality == "text" and text and text.strip():
            from backend.app.services.text_service import analyze_text
            return analyze_text(text.strip(), model_state)
        if modality == "photo" and photo_bytes and len(photo_bytes) > 100:
            from backend.app.services.photo_service import analyze_photo
            return analyze_photo(photo_bytes, model_state)
        if modality == "face" and frame_bytes:
            frames = [f for f in frame_bytes if f and len(f) > 100]
            if frames:
                from backend.app.services.face_service import analyze_live_face
                return analyze_live_face(frames, model_state)
        if modality == "voice" and audio_bytes and len(audio_bytes) > 100:
            from backend.app.services.voice_service import analyze_voice
            return analyze_voice(audio_bytes, session_id)
    except Exception as e:
        logger.error("Modality %s failed: %s", modality, e)
        return None
    return None


def complete_screening(
    session: Any,
    model_state: Any,
    *,
    photo_bytes: Optional[bytes] = None,
    frame_bytes: Optional[List[bytes]] = None,
    audio_bytes: Optional[bytes] = None,
    region: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the complete screening pipeline for a finished interview.

    ``force=True`` recomputes even when a result is already stored (used when
    new modality inputs arrive); default behavior returns the stored result
    so repeated GETs are stable.

    Returns the full result payload:
    { category, probability, uncertainty, dimensions, modalities_used,
      model_version, recommendations, ... } — screening-oriented, never
      diagnostic.
    """
    if session.result is not None and not force:
        return session.result

    # --- Safety override: replaces the normal flow entirely ---
    if session.safety_flag:
        crisis = get_crisis_response(region)
        session.status = "completed"
        session.result = {
            "screening_result": {
                "category": crisis["status"],
                "safety_override": True,
                "probability": None,
                "uncertainty": None,
                "dimensions": {},
                "modalities_used": list(session.modalities_used),
                "model_version": MODEL_VERSION,
            },
            "text": None,
            "face": None,
            "live_face": None,
            "voice": None,
            "fusion": None,
            "screening": {
                "category": crisis["status"],
                "non_diagnostic": True,
                "safety_override": True,
            },
            "explainability": None,
            "profile": None,
            "suggestions": crisis["recommendations"],
            "message": crisis["message"],
            "recommendations": crisis["recommendations"],
            "crisis_resources": crisis["crisis_resources"],
            "disclaimer": (
                "This is a research/educational screening tool. It is not a "
                "medical diagnosis and cannot provide crisis support."
            ),
        }
        return session.result

    # --- 1. Questionnaire modality (always available) ---
    questionnaire_result = build_questionnaire_result(session.features, session.question_count)
    modality_results: Dict[str, Dict[str, Any]] = {"questionnaire": questionnaire_result}
    modalities_used = ["questionnaire"]

    # --- 2. Available modalities (existing services, reused) ---
    transcript = " ".join(r.get("text", "") for r in session.responses).strip()
    model_failure: List[str] = []

    text_result = _run_modality("text", model_state, text=transcript)
    if text_result:
        modality_results["text"] = text_result
        modalities_used.append("text")
    else:
        model_failure.append("text")

    if photo_bytes:
        photo_result = _run_modality("photo", model_state, photo_bytes=photo_bytes)
        if photo_result:
            modality_results["photo"] = photo_result
            modalities_used.append("photo")
        else:
            model_failure.append("photo")

    if frame_bytes:
        face_result = _run_modality("face", model_state, frame_bytes=frame_bytes)
        if face_result:
            modality_results["face"] = face_result
            modalities_used.append("face")
        else:
            model_failure.append("face")

    if audio_bytes:
        voice_result = _run_modality("voice", model_state, audio_bytes=audio_bytes, session_id=session.session_id)
        if voice_result:
            modality_results["voice"] = voice_result
            modalities_used.append("voice")
        else:
            model_failure.append("voice")

    session.record_modalities(modalities_used)

    # --- 3. Multimodal fusion (weights renormalized over available modalities) ---
    fused = fuse_multimodal(modality_results, SCREENING_WEIGHTS)
    fused_probability = fused["probability"] / 100.0

    # --- 4. Probability calibration (fitted artifact only) ---
    # Phase 14: apply calibration ONLY to fused probabilities that are
    # dominated by the trained text model (its calibration artifact was fit
    # on that model's validation outputs). Otherwise the fusion output is a
    # heuristic mixture and applying a text-model calibrator would be
    # meaningless — we keep identity and say so.
    calibration_model = load_calibration_model()
    calibration_method = "identity"
    calibration_warning: Optional[str] = None
    signal_types = {
        m: (r.get("signal_type") or "rule_based_screening_score")
        for m, r in modality_results.items()
    }
    trained_present = any(st == "trained_model_probability" for st in signal_types.values())
    if calibration_model is not None:
        if calibration_model.method == "identity":
            calibration_method = "identity"
        elif trained_present:
            fused_probability = calibration_model.transform(fused_probability)
            calibration_method = calibration_model.method
        else:
            calibration_warning = (
                "A fitted calibration artifact exists, but the current fusion "
                "contains no trained-model signal — calibration was fitted for "
                "the trained text model's outputs and is NOT applied here. "
                "Reported probability is an uncalibrated heuristic fusion."
            )
            logger.warning(calibration_warning)
    else:
        calibration_warning = (
            "No calibration artifact (models/calibration.json) found — "
            "probabilities are UNCALIBRATED. Run scripts/fit_calibration.py "
            "after training."
        )
        logger.warning(calibration_warning)

    # --- 5. Uncertainty estimation (entropy + modality disagreement) ---
    fusion_meta = fused.get("_fusion_meta", {})
    modality_probs = fusion_meta.get("modality_probabilities", {})
    feature_confidences = [
        float(info.get("confidence", 0.5))
        for d, info in session.features.items()
        if d not in ("duration", "frequency", "severity")
    ] or [0.3]
    uncertainty = compute_uncertainty(
        fused_probability,
        modality_probs or None,
        feature_confidences,
    )
    track_history(session, fused_probability, uncertainty)

    # --- 6. Screening result (screening-oriented categories) ---
    category = categorize_probability(fused_probability)

    # --- 7. UserAssessmentProfile ---
    profile = build_user_profile(session, fused_probability, uncertainty)

    # --- 8. Personalized recommendations (driven by the profile) ---
    recommendations = generate_personalized_recommendations(
        profile,
        recommendation_history=session.recommendation_history,
    )
    session.recommendation_history.extend(r["id"] for r in recommendations)

    screening_result = {
        "category": category["label"],
        "category_key": category["key"],
        "probability": round(fused_probability, 4),
        "uncertainty": uncertainty,
        "uncertainty_label": uncertainty_label(uncertainty),
        "uncertainty_breakdown": uncertainty_breakdown(fused_probability, modality_probs or None, feature_confidences),
        "dimensions": profile.dimensions,
        "dimension_coverage": {k: round(v, 2) for k, v in compute_dimension_coverage(session.features).items()},
        "functional_impact": profile.functional_impact,
        "duration": profile.duration,
        "modalities_used": modalities_used,
        "signal_types": signal_types,
        "validated_modalities": fused.get("validated_modalities", []),
        "fusion_type": fused.get("fusion_type", "experimental_heuristic_fusion"),
        "fusion_interpretation": fused.get("explanation", {}).get("interpretation_note"),
        "model_version": MODEL_VERSION,
        "calibration_method": calibration_method,
        "fusion": {
            "weights": fused.get("fusion_weights", {}),
            "agreement": fused.get("explanation", {}).get("agreement_score"),
            "individual_results": fused.get("individual_results", {}),
        },
        "questions_answered": session.question_count,
        "prediction_history": list(session.prediction_history),
        "uncertainty_history": list(session.uncertainty_history),
    }
    if calibration_warning:
        screening_result["calibration_warning"] = calibration_warning
    if model_failure and set(model_failure) - {"text"}:
        screening_result["model_failure"] = model_failure
    if "text" in model_failure and session.question_count < MIN_QUESTIONS:
        screening_result["low_evidence_note"] = (
            "The interview ended before the minimum number of questions and the "
            "text model was unavailable — this result carries high uncertainty."
        )

    # --- 9. Four-modality contract + canonical response (spec §17/§42) ---
    modality_contracts = build_modality_contracts(modality_results)

    # All four modalities appear in the breakdown, with available=false for
    # branches that were not provided — the frontend renders availability
    # explicitly instead of hiding branches.
    modality_breakdown: Dict[str, Dict[str, Any]] = {}
    for m in MODALITY_CONTRACT_ORDER:
        if m in modality_contracts:
            modality_breakdown[m] = modality_contracts[m]
        else:
            modality_breakdown[m] = {
                "modality": m,
                "score": None,
                "confidence": None,
                "signal_type": None,
                "signal_type_label": None,
                "available": False,
                "risk_level": None,
                "evidence": {},
            }

    # Modality contributions as displayed on the result page (weight × score).
    modality_contributions: Dict[str, Optional[float]] = {}
    for internal_key, individual in fused.get("individual_results", {}).items():
        contract_name = _CONTRACT_KEY_MAP.get(internal_key, internal_key)
        contribution = individual.get("contribution")
        if contract_name in modality_contributions and modality_contributions[contract_name] is not None:
            continue
        modality_contributions[contract_name] = round(float(contribution), 4) if contribution is not None else None

    lime_payload = None
    text_contract = modality_contracts.get("text")
    if text_contract:
        text_result = modality_results.get("text") or {}
        lime_features = text_result.get("explanation", {}).get("lime", {}).get("features") if isinstance(text_result.get("explanation"), dict) else None
        if lime_features:
            lime_payload = {
                "label": (
                    "LIME feature attribution computed on the ACTUAL text "
                    "prediction function (trained model or clearly-labeled "
                    "fallback) — model behavior, not causal evidence."
                ),
                "features": lime_features,
            }

    canonical_result = {
        "text": modality_breakdown.get("text"),
        "face": modality_breakdown.get("face"),
        "live_face": modality_breakdown.get("live_face"),
        "voice": modality_breakdown.get("voice"),
        "fusion": {
            "score": round(fused_probability, 4),
            "confidence": round(fused.get("confidence", 0.0) / 100.0, 4),
            "uncertainty": uncertainty,
            "uncertainty_label": uncertainty_label(uncertainty),
            "agreement": fused.get("explanation", {}).get("agreement_score"),
            "fusion_type": fused.get("fusion_type"),
            "validated": False,
            "validated_modalities": fused.get("validated_modalities", []),
            "weights": fused.get("fusion_weights", {}),
            "note": (
                "Experimental multimodal screening fusion — NOT clinically "
                "validated. Weights renormalized over available modalities."
            ),
        },
        "screening": {
            "category": category["label"],
            "category_key": category["key"],
            "non_diagnostic": True,
            "functional_impact": profile.functional_impact,
            "duration": profile.duration,
        },
        "explainability": {
            "lime": lime_payload,
            "modality_contributions": modality_contributions,
            "agreement_note": (
                "The final screening indication was influenced by the "
                "available signals from text, face, live facial analysis, and "
                "voice. This describes correlation between signals and the "
                "fused score — not causality."
            ),
        },
        "profile": {
            "overall_screening_level": profile.overall_screening_level,
            "dimensions": profile.dimensions,
            "functional_impact": profile.functional_impact,
            "duration": profile.duration,
            "uncertainty": uncertainty,
            "modalities_used": profile.modalities_used,
        },
        "suggestions": recommendations,
        # Backwards-compatible alias of ``suggestions``.
        "recommendations": recommendations,
        "disclaimer": (
            "Research/educational screening only. This is not a medical "
            "diagnosis. Scores are experimental research signals; facial and "
            "voice components are auxiliary expression/acoustic signals, not "
            "validated depression predictions."
        ),
        "model_version": MODEL_VERSION,
    }

    session.status = "completed"
    session.result = {
        # Canonical response (spec §42) — consumed by the result page.
        **canonical_result,
        # Backwards-compatible nested summary for older consumers/tests.
        "screening_result": screening_result,
        "message": None,
        "crisis_resources": None,
    }
    # Research trace for external evaluation (spec §31/§32).
    session.research_trace = build_research_trace(session, modality_results, fused, canonical_result)
    return session.result
