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
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("neuroscreen.session")


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


# Global singleton
session_manager = SessionManager()
