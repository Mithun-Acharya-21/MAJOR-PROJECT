"""
Screening Routes — Adaptive screening API (extends, never breaks, existing endpoints).

Endpoints:
    POST /api/screening/start                       → session + first adaptive question
    POST /api/screening/{session_id}/response       → analyze response, update state, next question
    GET  /api/screening/{session_id}/state          → full interview state + progress
    GET  /api/screening/{session_id}/result         → screening result (completes if needed)
    GET  /api/screening/{session_id}/recommendations → personalized recommendations
    POST /api/screening/{session_id}/complete       → run full multimodal pipeline → result

All responses are screening-oriented (never diagnostic).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from backend.app.recommendations import get_crisis_response
from backend.app.session_manager import screening_session_manager
from backend.app.services.adaptive_question_service import (
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    build_question_payload,
    find_option,
    get_progress,
    select_next_question,
    should_stop,
)
from backend.app.services.response_analysis_service import analyze_option_response, analyze_response
from backend.app.services.screening_service import complete_screening

logger = logging.getLogger("emotiondep.screening_routes")

router = APIRouter(prefix="/api/screening", tags=["screening"])


def _get_session_or_404(session_id: str):
    session = screening_session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Screening session not found")
    return session


@router.post("/start")
async def screening_start():
    """Start an adaptive screening interview. Returns the first question."""
    session = screening_session_manager.create()
    question = select_next_question(session)
    if question is None:
        raise HTTPException(status_code=500, detail="Question bank exhausted")
    session.record_question(question)
    return {
        "session_id": session.session_id,
        "question": build_question_payload(question),
        "progress": get_progress(session),
        "status": session.status,
    }


@router.post("/{session_id}/response")
async def screening_response(
    session_id: str,
    payload: dict = Body(...),
):
    """Submit a response; the next question depends on previous responses."""
    session = _get_session_or_404(session_id)
    if session.status != "in_progress":
        raise HTTPException(status_code=409, detail="Screening session is already completed")

    answer = str(payload.get("answer", "")).strip()
    option_id = payload.get("option_id")
    option_id = str(option_id).strip() if option_id else None
    if not answer and not option_id:
        raise HTTPException(status_code=400, detail="Response text is required")

    question = session.questions[-1] if session.questions else None

    # --- Response analysis → structured information (no chain-of-thought) ---
    # Structured option selections take priority: the question-specific option
    # bank is the deterministic evidence source. Free text remains supported.
    option = None
    if option_id and question is not None:
        option = find_option(question, option_id)
        if option is None:
            raise HTTPException(status_code=400, detail="Unknown option for the current question")
    if option is not None:
        analysis = analyze_option_response(option, question)
        # Keep the human-readable selection for the transcript/research trace.
        response_text = answer or option["label"]
        if not answer:
            response_text = f"[Selected: {option['label']}]"
    else:
        response_text = answer
        analysis = analyze_response(answer, question)
    session.record_response(response_text, analysis)

    # --- Safety signals override the normal path ---
    if session.safety_flag:
        crisis = get_crisis_response(payload.get("region"))
        return {
            "session_id": session.session_id,
            "safety_concern": True,
            "message": crisis["message"],
            "crisis_resources": crisis["crisis_resources"],
            "question": None,
            "done": True,
            "progress": get_progress(session),
            "status": session.status,
        }

    # --- Adaptive stopping ---
    stop, reason = should_stop(session)
    if stop:
        return {
            "session_id": session.session_id,
            "safety_concern": False,
            "question": None,
            "done": True,
            "stop_reason": reason,
            "progress": get_progress(session),
            "status": session.status,
        }

    # --- Select next question (depends on all previous responses) ---
    next_question = select_next_question(session)
    if next_question is None:
        return {
            "session_id": session.session_id,
            "safety_concern": False,
            "question": None,
            "done": True,
            "stop_reason": "question_bank_exhausted",
            "progress": get_progress(session),
            "status": session.status,
        }
    session.record_question(next_question)

    return {
        "session_id": session.session_id,
        "safety_concern": False,
        "question": build_question_payload(next_question),
        "done": False,
        "progress": get_progress(session),
        "status": session.status,
    }


@router.get("/{session_id}/state")
async def screening_state(session_id: str):
    """Return the adaptive screening session state."""
    session = _get_session_or_404(session_id)
    return {
        **session.to_state_dict(),
        "progress": get_progress(session),
    }


def _ensure_completed(session, model_state, photo_bytes, frame_bytes, audio_bytes, region, force=False):
    """Run the pipeline once; return the stored result afterwards."""
    return complete_screening(
        session,
        model_state,
        photo_bytes=photo_bytes,
        frame_bytes=frame_bytes,
        audio_bytes=audio_bytes,
        region=region,
        force=force,
    )


@router.get("/{session_id}/result")
async def screening_result(session_id: str, region: Optional[str] = None):
    """Screening result — runs the pipeline on first access (questionnaire-only)."""
    session = _get_session_or_404(session_id)
    model_state = router.model_state  # type: ignore[attr-defined]
    result = _ensure_completed(session, model_state, None, None, None, region)
    return result


@router.get("/{session_id}/recommendations")
async def screening_recommendations(session_id: str, region: Optional[str] = None):
    """Personalized recommendations generated from the completed assessment state.

    Arbitrary recommendation text from the frontend is never accepted.
    """
    session = _get_session_or_404(session_id)
    model_state = router.model_state  # type: ignore[attr-defined]
    result = _ensure_completed(session, model_state, None, None, None, region)
    recommendations = result.get("suggestions") or result.get("recommendations", [])
    return {
        "session_id": session.session_id,
        "recommendations": recommendations,
        "safety_override": bool(result.get("screening_result", {}).get("safety_override")),
        "crisis_resources": result.get("crisis_resources"),
    }


@router.get("/{session_id}/research-trace")
async def screening_research_trace(session_id: str):
    """Research/debug trace (spec §31/§32): the full computational chain.

    Exposes question responses, structured features, per-modality model
    outputs, fusion weights, final score, uncertainty, LIME features, and
    recommendation reasoning — for external evaluation of WHY a result was
    produced. Intended for research inspection, not general production use.
    """
    session = _get_session_or_404(session_id)
    model_state = router.model_state  # type: ignore[attr-defined]
    _ensure_completed(session, model_state, None, None, None, None)
    if session.research_trace is None:
        raise HTTPException(status_code=404, detail="Research trace not available for this session")
    return {
        "session_id": session.session_id,
        **session.research_trace,
    }


@router.post("/{session_id}/complete")
async def screening_complete(
    session_id: str,
    photo: Optional[UploadFile] = File(None, description="Optional facial photo"),
    frames: Optional[List[UploadFile]] = File(None, description="Optional webcam frames"),
    audio: Optional[UploadFile] = File(None, description="Optional voice recording"),
    region: Optional[str] = Form(None, description="Optional region for crisis resources"),
):
    """Complete the screening: run available multimodal models → fusion →
    calibration → uncertainty → result → profile → recommendations."""
    session = _get_session_or_404(session_id)
    if session.status == "completed" and session.result is not None:
        return session.result

    model_state = router.model_state  # type: ignore[attr-defined]

    photo_bytes = None
    if photo is not None:
        photo_bytes = await photo.read()
    frame_bytes: List[bytes] = []
    if frames:
        for frame in frames:
            try:
                data = await frame.read()
                if data and len(data) > 100:
                    frame_bytes.append(data)
            except Exception:
                continue
    audio_bytes = None
    if audio is not None:
        audio_bytes = await audio.read()

    try:
        result = _ensure_completed(session, model_state, photo_bytes, frame_bytes, audio_bytes, region, force=True)
    except Exception as e:
        logger.exception("Screening completion failed: %s", e)
        session.status = "analysis_failed"
        raise HTTPException(status_code=500, detail=f"Screening analysis failed: {e}")

    return result
