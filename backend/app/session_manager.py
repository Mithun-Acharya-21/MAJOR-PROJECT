"""
Session Manager — Accumulates transcript, audio features, and facial features
across live analysis ticks within a single session.

Architecture:
    Session start
        ↓
    [tick 1] audio → Whisper → segment → dedup → append to transcript
              audio → voice features → accumulate
              image → facial features → accumulate
        ↓
    [tick 2] audio → Whisper → segment → dedup → append to transcript
              audio → voice features → accumulate
              image → facial features → accumulate
        ↓
    Session stop → final analysis with full accumulated data

Exports:
    SessionManager  — manages session state
    Session         — dataclass for a single session's state
    ScreeningSession        — dataclass for adaptive screening interview state
    ScreeningSessionManager — manages adaptive screening sessions
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("emotiondep.session")


@dataclass
class VoiceFeatures:
    """Acoustic voice features extracted from raw audio."""
    rms_energy: float = 0.0
    speaking_rate: float = 0.0       # syllables per second estimate
    silence_ratio: float = 0.0       # fraction of near-silence frames
    pitch_mean: float = 0.0          # estimated pitch in Hz (0 if unavailable)
    pitch_variation: float = 0.0     # std of pitch
    duration_sec: float = 0.0
    intensity_mean: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "rms_energy": round(self.rms_energy, 4),
            "speaking_rate": round(self.speaking_rate, 2),
            "silence_ratio": round(self.silence_ratio, 4),
            "pitch_mean": round(self.pitch_mean, 1),
            "pitch_variation": round(self.pitch_variation, 2),
            "duration_sec": round(self.duration_sec, 3),
            "intensity_mean": round(self.intensity_mean, 4),
        }


@dataclass
class Session:
    """Mutable state for a single live-analysis session."""
    session_id: str = ""
    started_at: float = 0.0

    # Transcript accumulation
    transcript_segments: List[str] = field(default_factory=list)
    full_transcript: str = ""

    # Audio features (accumulated across ticks)
    voice_features: List[VoiceFeatures] = field(default_factory=list)
    avg_voice_features: Optional[VoiceFeatures] = None

    # Facial analysis state
    facial_scores: List[float] = field(default_factory=list)
    last_image_bytes: Optional[bytes] = None

    # Analysis results (last tick)
    last_result: Optional[Dict[str, Any]] = None

    # Modality availability tracking
    modalities_available: Dict[str, bool] = field(default_factory=dict)

    # Errors encountered (non-fatal, per modality)
    errors: List[str] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all accumulated data for a fresh session."""
        self.session_id = str(uuid.uuid4())[:8]
        self.started_at = time.time()
        self.transcript_segments = []
        self.full_transcript = ""
        self.voice_features = []
        self.avg_voice_features = None
        self.facial_scores = []
        self.last_image_bytes = None
        self.last_result = None
        self.modalities_available = {}
        self.errors = []

    def get_summary(self) -> Dict[str, Any]:
        """Return a serializable summary of the session state."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "duration_sec": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "transcript": self.full_transcript,
            "num_segments": len(self.transcript_segments),
            "voice_features": self.avg_voice_features.to_dict() if self.avg_voice_features else None,
            "facial_score_count": len(self.facial_scores),
            "modalities_available": dict(self.modalities_available),
            "errors": list(self.errors),
        }


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for duplicate detection — lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _word_overlap_ratio(a: str, b: str) -> float:
    """Compute word-level Jaccard overlap between two strings."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def _is_duplicate(new_text: str, existing_segments: List[str], overlap_threshold: float = 0.5) -> bool:
    """Check if new_text substantially overlaps with recent transcript segments.

    Uses word-level Jaccard overlap to detect duplicate/overlapping transcriptions
    from consecutive Whisper chunks. Returns True if the new text is likely a
    duplicate of already-accumulated content.
    """
    if not new_text.strip():
        return True  # Empty transcription is "duplicate" of nothing

    norm_new = _normalize_for_dedup(new_text)
    if not norm_new:
        return True

    # Check against the last 3 segments (most likely to overlap)
    recent = existing_segments[-3:] if existing_segments else []
    for seg in recent:
        norm_seg = _normalize_for_dedup(seg)
        if not norm_seg:
            continue

        # Exact match
        if norm_new == norm_seg:
            return True

        # High overlap → likely duplicate from overlapping audio chunks
        overlap = _word_overlap_ratio(norm_new, norm_seg)
        if overlap >= overlap_threshold:
            return True

        # Substring containment (new text is mostly contained in existing)
        if len(norm_new) > 10 and norm_new in norm_seg:
            return True
        if len(norm_seg) > 10 and norm_seg in norm_new:
            # The new text contains existing — this is an extension, not duplicate
            # But if the extension is tiny, treat as duplicate
            extra = norm_new.replace(norm_seg, "").strip()
            if len(extra.split()) <= 3:
                return True

    return False


def _merge_extending_text(existing_segments: List[str], new_text: str) -> str:
    """Merge new text with existing segments, preferring the longer/extended version.

    When Whisper produces overlapping but slightly different transcriptions of the
    same speech, this picks the most complete version.
    """
    if not existing_segments:
        return new_text.strip()

    norm_new = _normalize_for_dedup(new_text)
    last_seg = existing_segments[-1] if existing_segments else ""
    norm_last = _normalize_for_dedup(last_seg)

    # If the new text is a strict extension of the last segment, use the new text
    if norm_last and norm_last in norm_new and len(norm_new) > len(norm_last):
        return new_text.strip()

    # If the last segment extends the new text, keep the last segment
    if norm_new and norm_new in norm_last and len(norm_last) > len(norm_new):
        return last_seg.strip()

    # Otherwise, just append
    return new_text.strip()


class SessionManager:
    """Manages live-analysis sessions with transcript accumulation and dedup."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """Get an existing session or create a new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        session = Session()
        session.session_id = session_id or str(uuid.uuid4())[:8]
        session.started_at = time.time()
        self._sessions[session.session_id] = session
        return session

    def start_session(self, session_id: Optional[str] = None) -> Session:
        """Start a new session, resetting any previous state."""
        sid = session_id or str(uuid.uuid4())[:8]
        session = Session()
        session.session_id = sid
        session.started_at = time.time()
        self._sessions[sid] = session
        logger.info("Session started: %s", sid)
        return session

    def end_session(self, session_id: str) -> Optional[Session]:
        """End and return a session (for final analysis)."""
        session = self._sessions.pop(session_id, None)
        if session:
            logger.info("Session ended: %s (segments: %d)", session_id, len(session.transcript_segments))
        return session

    def add_transcript(self, session_id: str, new_text: str) -> Tuple[str, bool]:
        """Add a new transcript segment to the session.

        Handles deduplication and returns the updated full transcript plus
        whether the new text was accepted (not a duplicate).

        Returns:
            (full_transcript, was_accepted)
        """
        session = self._sessions.get(session_id)
        if not session:
            return new_text.strip(), True

        if not new_text or not new_text.strip():
            return session.full_transcript, False

        # Check for duplicates
        if _is_duplicate(new_text, session.transcript_segments):
            logger.debug("Duplicate transcript segment rejected: %s", new_text[:80])
            return session.full_transcript, False

        # Merge/append
        merged = _merge_extending_text(session.transcript_segments, new_text)
        if merged:
            session.transcript_segments.append(merged)
            session.full_transcript = " ".join(session.transcript_segments).strip()
            logger.debug("Transcript updated (%d segments): %s", len(session.transcript_segments), merged[:80])

        return session.full_transcript, True

    def add_voice_features(self, session_id: str, features: VoiceFeatures) -> None:
        """Accumulate voice features and update running averages."""
        session = self._sessions.get(session_id)
        if not session:
            return

        session.voice_features.append(features)

        # Compute running average
        n = len(session.voice_features)
        avg = VoiceFeatures(
            rms_energy=sum(f.rms_energy for f in session.voice_features) / n,
            speaking_rate=sum(f.speaking_rate for f in session.voice_features) / n,
            silence_ratio=sum(f.silence_ratio for f in session.voice_features) / n,
            pitch_mean=sum(f.pitch_mean for f in session.voice_features) / n,
            pitch_variation=sum(f.pitch_variation for f in session.voice_features) / n,
            duration_sec=sum(f.duration_sec for f in session.voice_features) / n,
            intensity_mean=sum(f.intensity_mean for f in session.voice_features) / n,
        )
        session.avg_voice_features = avg

    def add_facial_score(self, session_id: str, score: float) -> None:
        """Accumulate facial analysis score."""
        session = self._sessions.get(session_id)
        if session:
            session.facial_scores.append(score)

    def set_modality(self, session_id: str, modality: str, available: bool) -> None:
        """Track modality availability."""
        session = self._sessions.get(session_id)
        if session:
            session.modalities_available[modality] = available

    def add_error(self, session_id: str, error: str) -> None:
        """Record a non-fatal error."""
        session = self._sessions.get(session_id)
        if session:
            session.errors.append(error)

    def set_last_result(self, session_id: str, result: Dict[str, Any]) -> None:
        """Store the last analysis result for this session."""
        session = self._sessions.get(session_id)
        if session:
            session.last_result = result


# ---------------------------------------------------------------------------
# Adaptive screening session state
# ---------------------------------------------------------------------------

# Structured information dimensions tracked across the interview.
# These are information categories for screening research — NOT medical diagnoses.
SCREENING_DIMENSIONS = (
    "mood",
    "loss_of_interest",
    "sleep",
    "energy",
    "appetite",
    "concentration",
    "self_worth",
    "guilt",
    "psychomotor_changes",
    "social_withdrawal",
    "daily_functioning",
    "duration",
    "frequency",
    "severity",
)

# Dimensions that carry symptom-severity values (0-3); meta-dimensions
# (duration/frequency/severity) are stored as descriptive strings.
SEVERITY_DIMENSIONS = frozenset(
    {
        "mood",
        "loss_of_interest",
        "sleep",
        "energy",
        "appetite",
        "concentration",
        "self_worth",
        "guilt",
        "psychomotor_changes",
        "social_withdrawal",
        "daily_functioning",
    }
)


class ScreeningSession:
    """Mutable state for one adaptive screening interview.

    Stored per the project spec:
      session_id, questions, responses, features, dimension_coverage,
      asked_question_ids, prediction_history, uncertainty_history,
      modalities_used, question_count, status (+ recommendation_history).
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid.uuid4())[:12]
        self.started_at: float = time.time()
        self.questions: List[Dict[str, Any]] = []          # questions asked (with ids)
        self.responses: List[Dict[str, Any]] = []          # raw + analyzed responses
        self.features: Dict[str, Dict[str, Any]] = {}      # dimension -> {severity, confidence, evidence}
        self.dimension_coverage: Dict[str, float] = {}     # dimension -> 0..1 coverage
        self.asked_question_ids: List[str] = []            # exact-duplicate guard
        self.question_semantics: List[str] = []            # normalized question texts (semantic-dedup)
        self.prediction_history: List[float] = []          # fused probability after each question
        self.uncertainty_history: List[float] = []         # uncertainty estimate after each question
        self.modalities_used: List[str] = ["text"]         # questionnaire text is always present
        self.question_count: int = 0
        self.status: str = "in_progress"                  # in_progress | completed | analysis_failed
        self.recommendation_history: List[str] = []        # recommendation ids already shown
        self.user_context: Dict[str, Any] = {}             # optional user-provided context
        self.safety_flag: bool = False
        self.result: Optional[Dict[str, Any]] = None       # final screening result
        self.research_trace: Optional[Dict[str, Any]] = None  # spec §31 research trace

    # ------------------------------------------------------------------
    # Single source of truth for coverage (Phase 13).
    #
    # ``compute_dimension_coverage`` (adaptive_question_service) is the ONE
    # coverage definition: coverage = confidence, boosted slightly by
    # repeated observations. ``should_stop`` and ``get_progress`` both use
    # it — there is no longer a separately-incremented dimension_coverage
    # dict whose values could diverge from the stopping logic.
    # ------------------------------------------------------------------
    def refresh_dimension_coverage(self) -> None:
        """Recompute dimension_coverage from features via the shared formula."""
        try:
            from backend.app.services.adaptive_question_service import (
                compute_dimension_coverage,
            )
            self.dimension_coverage = compute_dimension_coverage(self.features)
        except Exception as e:  # pragma: no cover - non-fatal
            logger.debug("Could not refresh coverage: %s", e)

    # -- Question/response bookkeeping -------------------------------------

    def record_question(self, question: Dict[str, Any]) -> None:
        """Store a question that was (or is about to be) asked."""
        self.questions.append(question)
        if question.get("id") and question["id"] not in self.asked_question_ids:
            self.asked_question_ids.append(question["id"])
        semantic = _normalize_for_dedup(question.get("text", ""))
        if semantic:
            self.question_semantics.append(semantic)
        self.question_count += 1

    def record_response(self, response_text: str, analysis: Dict[str, Any]) -> None:
        """Store a raw response plus its structured analysis (no chain-of-thought)."""
        self.responses.append(
            {
                "question_id": self.questions[-1]["id"] if self.questions else None,
                "text": response_text,
                "analysis": analysis,
            }
        )
        for dim, info in analysis.get("dimensions", {}).items():
            if dim not in SCREENING_DIMENSIONS:
                continue
            existing = self.features.get(dim)
            confidence = float(info.get("confidence", 0.5))
            severity = float(info.get("severity", 0))
            if existing is None:
                self.features[dim] = {
                    "severity": severity,
                    "confidence": confidence,
                    "observations": 1,
                }
            else:
                # Confidence-weighted merge of repeated observations.
                total = existing["confidence"] + confidence
                if total <= 0:
                    merged_severity = max(existing["severity"], severity)
                else:
                    merged_severity = (
                        existing["severity"] * existing["confidence"]
                        + severity * confidence
                    ) / total
                self.features[dim] = {
                    "severity": round(merged_severity, 3),
                    "confidence": min(1.0, max(existing["confidence"], confidence) * 0.9 + confidence * 0.1),
                    "observations": existing["observations"] + 1,
                }
        # Coverage is ALWAYS derived from features (single source of truth).
        self.refresh_dimension_coverage()
        # Meta-dimension extraction (duration/frequency) from the analysis payload.
        for meta_key in ("duration", "frequency", "severity"):
            value = analysis.get(meta_key)
            if value and meta_key not in self.features:
                self.features[meta_key] = {"value": value, "confidence": 0.8}
        self.refresh_dimension_coverage()
        # Optional user context supplied with a response.
        context = analysis.get("context") or {}
        for key, value in context.items():
            if value not in (None, ""):
                self.user_context[key] = value
        if analysis.get("safety_concern"):
            self.safety_flag = True

    def record_modalities(self, modalities: List[str]) -> None:
        """Track which modalities will be used in the final analysis."""
        for m in modalities:
            if m and m not in self.modalities_used:
                self.modalities_used.append(m)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serializable state matching the documented schema."""
        return {
            "session_id": self.session_id,
            "questions": list(self.questions),
            "responses": list(self.responses),
            "features": self.features,
            "dimension_coverage": self.dimension_coverage,
            "asked_question_ids": list(self.asked_question_ids),
            "prediction_history": list(self.prediction_history),
            "uncertainty_history": list(self.uncertainty_history),
            "modalities_used": list(self.modalities_used),
            "question_count": self.question_count,
            "status": self.status,
            "recommendation_history": list(self.recommendation_history),
            "safety_flag": self.safety_flag,
        }


class ScreeningSessionManager:
    """Manages adaptive screening interview sessions (separate from live analysis)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ScreeningSession] = {}

    def create(self, session_id: Optional[str] = None) -> ScreeningSession:
        session = ScreeningSession(session_id=session_id or str(uuid.uuid4())[:12])
        self._sessions[session.session_id] = session
        logger.info("Screening session started: %s", session.session_id)
        return session

    def get(self, session_id: str) -> Optional[ScreeningSession]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str]) -> ScreeningSession:
        if session_id:
            existing = self._sessions.get(session_id)
            if existing:
                return existing
        return self.create(session_id)

    def pop(self, session_id: str) -> Optional[ScreeningSession]:
        return self._sessions.pop(session_id, None)


# Global singletons
session_manager = SessionManager()
screening_session_manager = ScreeningSessionManager()
