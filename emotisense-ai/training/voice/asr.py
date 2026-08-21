"""Phase 11 — Local/offline ASR using OpenAI Whisper.

Design principles
-----------------
- Runs entirely offline; no audio is sent to any external service.
- Uses the smallest model that meets quality requirements (default: "base").
- Audio is processed in-memory; no temporary files are written.
- Every result carries a language confidence field and a quality assessment.
- Low-confidence transcripts reduce the voice branch's fusion weight.
- The transcript feeds the text branch (DeBERTa); ASR errors reduce quality.

ASRResult fields
----------------
  transcript      : str   — normalised Whisper output
  language        : str   — detected BCP-47 language code
  language_prob   : float — Whisper's language confidence ∈ [0, 1]
  duration_s      : float — audio length
  no_speech_prob  : float — Whisper segment-level no-speech probability
  word_segments   : list[dict] — word-level timestamps if available
  quality_score   : float — composite ∈ [0, 1]; fed to fusion gate
  is_usable       : bool  — True only if the transcript is reliable enough
  rejection_reasons: list[str]

Quality thresholds (all configurable)
--------------------------------------
  min_language_prob : 0.70 — below → low_language_confidence
  max_no_speech_prob: 0.50 — above → likely_no_speech
  min_word_count    : 3    — below → too_few_words
  allowed_languages : None (any) or a list of BCP-47 codes
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ASR_CONFIG: dict = {
    "model_size": "base",            # tiny | base | small | medium | large
    "language": None,                # Force language or None for auto-detect
    "min_language_prob": 0.70,
    "max_no_speech_prob": 0.50,
    "min_word_count": 3,
    "allowed_languages": None,       # None = all; or e.g. ["en", "zh"]
    "word_timestamps": True,
    "fp16": False,                   # Disable on CPU
    "beam_size": 5,
    "best_of": 5,
    "temperature": 0.0,              # Greedy decoding for reproducibility
    "condition_on_previous_text": False,
    "target_sample_rate": 16000,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ASRResult:
    transcript: str = ""
    language: str = ""
    language_prob: float = 0.0
    duration_s: float = 0.0
    no_speech_prob: float = 0.0
    word_segments: list[dict] = field(default_factory=list)
    quality_score: float = 0.0       # Composite ∈ [0, 1]; fed to fusion gate
    is_usable: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    model_size: str = ""
    word_count: int = 0

    def to_metadata(self) -> dict:
        """JSON-serialisable summary — transcript included, no raw audio."""
        return {
            "transcript": self.transcript,
            "language": self.language,
            "language_prob": round(self.language_prob, 4),
            "duration_s": round(self.duration_s, 3),
            "no_speech_prob": round(self.no_speech_prob, 4),
            "word_count": self.word_count,
            "quality_score": round(self.quality_score, 4),
            "is_usable": self.is_usable,
            "rejection_reasons": self.rejection_reasons,
            "model_size": self.model_size,
        }


# ---------------------------------------------------------------------------
# ASR runner
# ---------------------------------------------------------------------------

class WhisperASR:
    """Thin wrapper around OpenAI Whisper for offline transcription.

    The model is loaded once on construction and reused across calls.
    Model weights are downloaded to the local cache on first use.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = {**DEFAULT_ASR_CONFIG, **(config or {})}
        self._model = None   # Lazy-load

    def _get_model(self):
        if self._model is None:
            try:
                import whisper
            except ImportError as exc:
                raise ImportError(
                    "openai-whisper is required. pip install openai-whisper"
                ) from exc
            self._model = whisper.load_model(self.cfg["model_size"])
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_path(self, audio_path: Path | str) -> ASRResult:
        """Transcribe an audio file from disk."""
        try:
            import whisper
            audio = whisper.load_audio(str(audio_path))
        except Exception as exc:
            result = ASRResult()
            result.rejection_reasons.append(f"load_failed:{exc}")
            return result
        return self._run(audio)

    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> ASRResult:
        """Transcribe a pre-loaded mono float32 waveform."""
        try:
            import whisper
            # Whisper expects float32 mono at 16 kHz
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[-1] else 1)
            if sample_rate != self.cfg["target_sample_rate"]:
                try:
                    import librosa
                    waveform = librosa.resample(
                        waveform,
                        orig_sr=sample_rate,
                        target_sr=self.cfg["target_sample_rate"],
                    )
                except ImportError:
                    pass  # Whisper will handle it internally
            audio = waveform.astype(np.float32)
        except Exception as exc:
            result = ASRResult()
            result.rejection_reasons.append(f"prepare_failed:{exc}")
            return result
        return self._run(audio)

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    def _run(self, audio: np.ndarray) -> ASRResult:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError("openai-whisper required") from exc

        result = ASRResult(
            model_size=self.cfg["model_size"],
            duration_s=round(len(audio) / self.cfg["target_sample_rate"], 3),
        )

        try:
            model = self._get_model()
            output = model.transcribe(
                audio,
                language=self.cfg["language"],
                word_timestamps=self.cfg["word_timestamps"],
                fp16=self.cfg["fp16"],
                beam_size=self.cfg["beam_size"],
                best_of=self.cfg["best_of"],
                temperature=self.cfg["temperature"],
                condition_on_previous_text=self.cfg["condition_on_previous_text"],
            )
        except Exception as exc:
            result.rejection_reasons.append(f"transcription_error:{exc}")
            return result

        # Extract transcript
        result.transcript = output.get("text", "").strip()
        result.language = output.get("language", "")

        # Language detection probability from first segment
        segments = output.get("segments", [])
        if segments:
            lang_probs = output.get("language_probs") or {}
            result.language_prob = float(
                lang_probs.get(result.language, 0.0)
                if lang_probs
                else (segments[0].get("avg_logprob", -10) + 10) / 10
            )
            result.no_speech_prob = float(
                max(s.get("no_speech_prob", 0.0) for s in segments)
            )
            # Word timestamps
            if self.cfg["word_timestamps"]:
                result.word_segments = [
                    {"word": w.get("word", ""), "start": w.get("start", 0.0), "end": w.get("end", 0.0)}
                    for seg in segments
                    for w in seg.get("words", [])
                ]
        else:
            result.language_prob = 0.0
            result.no_speech_prob = 1.0

        result.word_count = len(result.transcript.split()) if result.transcript else 0

        # Quality checks
        self._quality_checks(result)
        result.quality_score = self._composite_quality(result)
        result.is_usable = not bool(result.rejection_reasons)
        return result

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def _quality_checks(self, result: ASRResult) -> None:
        cfg = self.cfg
        if result.language_prob < cfg["min_language_prob"]:
            result.rejection_reasons.append("low_language_confidence")
        if result.no_speech_prob > cfg["max_no_speech_prob"]:
            result.rejection_reasons.append("likely_no_speech")
        if result.word_count < cfg["min_word_count"]:
            result.rejection_reasons.append("too_few_words")
        allowed = cfg.get("allowed_languages")
        if allowed and result.language and result.language not in allowed:
            result.rejection_reasons.append(f"unsupported_language:{result.language}")
        if not result.transcript:
            result.rejection_reasons.append("empty_transcript")

    def _composite_quality(self, result: ASRResult) -> float:
        """Weighted sub-scores ∈ [0, 1]; critical failures apply penalties."""
        lang_score = min(1.0, result.language_prob)
        speech_score = max(0.0, 1.0 - result.no_speech_prob)
        word_score = min(1.0, result.word_count / 20.0)  # Saturates at 20 words
        base = 0.40 * lang_score + 0.40 * speech_score + 0.20 * word_score
        # Hard penalty for each rejection
        penalty = len(result.rejection_reasons) * 0.25
        return round(max(0.0, min(1.0, base - penalty)), 4)


# ---------------------------------------------------------------------------
# Transcript quality assessment (post-ASR)
# ---------------------------------------------------------------------------

def assess_transcript_quality(transcript: str) -> dict:
    """Lightweight heuristic quality checks on a transcript string.

    Returns a dict with:
      word_count, sentence_count, avg_word_length,
      has_repetition, repetition_ratio,
      quality_score ∈ [0, 1], issues []
    """
    import re

    issues: list[str] = []
    if not transcript or not transcript.strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "has_repetition": False,
            "repetition_ratio": 0.0,
            "quality_score": 0.0,
            "issues": ["empty_transcript"],
        }

    words = re.findall(r"\b\w+\b", transcript.lower())
    sentences = re.split(r"[.!?]+", transcript.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

    # Repetition: consecutive duplicate words
    rep_count = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    repetition_ratio = rep_count / max(1, word_count)
    has_repetition = repetition_ratio > 0.1

    if word_count < 3:
        issues.append("too_few_words")
    if has_repetition:
        issues.append("high_repetition")
    if avg_word_length < 2.0:
        issues.append("very_short_words")

    # Quality score
    word_score = min(1.0, word_count / 30.0)
    rep_score = max(0.0, 1.0 - repetition_ratio * 5)
    quality = round(0.6 * word_score + 0.4 * rep_score, 4)
    if issues:
        quality = round(max(0.0, quality - len(issues) * 0.15), 4)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "has_repetition": has_repetition,
        "repetition_ratio": round(repetition_ratio, 4),
        "quality_score": quality,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Voice-branch orchestrator
# ---------------------------------------------------------------------------

def run_voice_branch(
    audio_path: Optional[Path] = None,
    waveform: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    asr_config: Optional[dict] = None,
) -> dict:
    """End-to-end voice pipeline: validate → transcribe → quality-assess.

    Returns a structured result dict suitable for logging and fusion.
    The raw waveform is never persisted.
    """
    from training.voice.audio_validate import AudioValidator

    validator = AudioValidator()
    asr = WhisperASR(asr_config)

    # 1. Audio validation
    if audio_path is not None:
        validation = validator.validate_path(audio_path)
        if not validation.is_valid:
            return {
                "status": "invalid_audio",
                "validation": validation.to_metadata(),
                "asr": None,
                "transcript_quality": None,
            }
        asr_result = asr.transcribe_path(audio_path)
    elif waveform is not None:
        validation = validator.validate_array(waveform, sample_rate)
        # Proceed with ASR even for borderline quality — quality_score handles it
        asr_result = asr.transcribe_array(waveform, sample_rate)
    else:
        raise ValueError("Either audio_path or waveform must be provided")

    # 2. Transcript quality
    tq = assess_transcript_quality(asr_result.transcript)

    # 3. Combine quality scores
    combined_quality = round(
        0.4 * validation.quality_score
        + 0.4 * asr_result.quality_score
        + 0.2 * tq["quality_score"],
        4,
    )

    return {
        "status": "ok" if asr_result.is_usable else "low_quality",
        "validation": validation.to_metadata(),
        "asr": asr_result.to_metadata(),
        "transcript_quality": tq,
        "combined_quality_score": combined_quality,  # → fusion gate (Phase 14)
    }
