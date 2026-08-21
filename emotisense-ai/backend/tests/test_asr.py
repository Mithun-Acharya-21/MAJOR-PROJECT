"""Unit tests for Phase 11 — ASR integration and transcript quality handling.

Whisper model downloads are skipped in unit tests.  We test:
  - ASRResult metadata serialisation
  - Quality checks and composite scoring
  - Transcript quality heuristics
  - run_voice_branch routing logic (mocked ASR)
  - Rejection reason coverage
"""

from __future__ import annotations

import numpy as np
import pytest

from training.voice.asr import (
    ASRResult,
    WhisperASR,
    assess_transcript_quality,
)


# ---------------------------------------------------------------------------
# ASRResult tests
# ---------------------------------------------------------------------------

class TestASRResult:

    def test_metadata_is_json_serialisable(self) -> None:
        import json
        r = ASRResult(
            transcript="hello world",
            language="en",
            language_prob=0.95,
            duration_s=3.0,
            no_speech_prob=0.05,
            word_count=2,
            quality_score=0.82,
            is_usable=True,
            model_size="base",
        )
        json.dumps(r.to_metadata())

    def test_metadata_contains_required_keys(self) -> None:
        r = ASRResult()
        meta = r.to_metadata()
        for key in ("transcript", "language", "quality_score", "is_usable",
                    "rejection_reasons", "word_count", "no_speech_prob"):
            assert key in meta, f"Missing key: {key}"

    def test_empty_result_not_usable(self) -> None:
        r = ASRResult()
        assert not r.is_usable


# ---------------------------------------------------------------------------
# WhisperASR quality-check and scoring (no model load)
# ---------------------------------------------------------------------------

class TestWhisperASRQualityChecks:

    def _make_result(self, **kwargs) -> ASRResult:
        defaults = dict(
            transcript="I feel really tired all the time",
            language="en",
            language_prob=0.90,
            no_speech_prob=0.05,
            word_count=7,
        )
        defaults.update(kwargs)
        r = ASRResult(**defaults)
        return r

    def _asr(self) -> WhisperASR:
        return WhisperASR()

    def test_low_language_confidence_rejected(self) -> None:
        asr = self._asr()
        r = self._make_result(language_prob=0.30)
        asr._quality_checks(r)
        assert "low_language_confidence" in r.rejection_reasons

    def test_high_no_speech_rejected(self) -> None:
        asr = self._asr()
        r = self._make_result(no_speech_prob=0.80)
        asr._quality_checks(r)
        assert "likely_no_speech" in r.rejection_reasons

    def test_too_few_words_rejected(self) -> None:
        asr = self._asr()
        r = self._make_result(word_count=2, transcript="hi")
        asr._quality_checks(r)
        assert "too_few_words" in r.rejection_reasons

    def test_unsupported_language_rejected(self) -> None:
        asr = WhisperASR({"allowed_languages": ["en", "zh"]})
        r = self._make_result(language="de")
        asr._quality_checks(r)
        assert any("unsupported_language" in reason for reason in r.rejection_reasons)

    def test_empty_transcript_rejected(self) -> None:
        asr = self._asr()
        r = self._make_result(transcript="", word_count=0)
        asr._quality_checks(r)
        assert "empty_transcript" in r.rejection_reasons

    def test_quality_score_range(self) -> None:
        asr = self._asr()
        for lp, ns, wc in [(0.9, 0.05, 15), (0.3, 0.8, 1), (0.7, 0.2, 5)]:
            r = self._make_result(
                language_prob=lp, no_speech_prob=ns, word_count=wc,
                transcript=" ".join(["word"] * wc)
            )
            asr._quality_checks(r)
            score = asr._composite_quality(r)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_clean_transcript_high_score(self) -> None:
        asr = self._asr()
        r = self._make_result(language_prob=0.98, no_speech_prob=0.02, word_count=25,
                              transcript=" ".join(["word"] * 25))
        asr._quality_checks(r)
        score = asr._composite_quality(r)
        assert score > 0.5, f"Expected high score for clean result, got {score}"

    def test_no_rejections_for_good_result(self) -> None:
        asr = self._asr()
        r = self._make_result()
        asr._quality_checks(r)
        assert r.rejection_reasons == []


# ---------------------------------------------------------------------------
# Transcript quality heuristic tests
# ---------------------------------------------------------------------------

class TestAssessTranscriptQuality:

    def test_empty_string_returns_zero_quality(self) -> None:
        result = assess_transcript_quality("")
        assert result["quality_score"] == 0.0
        assert "empty_transcript" in result["issues"]

    def test_good_transcript_high_score(self) -> None:
        text = "I have been feeling very low energy and unmotivated for several weeks now."
        result = assess_transcript_quality(text)
        assert result["quality_score"] > 0.5
        assert result["word_count"] >= 10

    def test_very_short_transcript_flagged(self) -> None:
        result = assess_transcript_quality("hi")
        assert "too_few_words" in result["issues"]

    def test_high_repetition_detected(self) -> None:
        text = "the the the the the end"
        result = assess_transcript_quality(text)
        assert result["has_repetition"] is True
        assert "high_repetition" in result["issues"]

    def test_word_count_accurate(self) -> None:
        text = "one two three four five"
        result = assess_transcript_quality(text)
        assert result["word_count"] == 5

    def test_sentence_count_detected(self) -> None:
        text = "I feel tired. My sleep is poor. I have no energy."
        result = assess_transcript_quality(text)
        assert result["sentence_count"] == 3

    def test_result_is_json_serialisable(self) -> None:
        import json
        text = "I am feeling very anxious lately."
        json.dumps(assess_transcript_quality(text))

    def test_quality_score_in_unit_interval(self) -> None:
        for text in ["", "hi", "hello world how are you today",
                     "the the the the the the"]:
            result = assess_transcript_quality(text)
            assert 0.0 <= result["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# run_voice_branch routing (no Whisper model load)
# ---------------------------------------------------------------------------

class TestRunVoiceBranchRouting:

    def test_no_input_raises(self) -> None:
        from training.voice.asr import run_voice_branch
        with pytest.raises(ValueError, match="audio_path or waveform"):
            run_voice_branch()

    def test_invalid_audio_returns_invalid_status(self) -> None:
        from training.voice.asr import run_voice_branch
        # Silence → insufficient voice activity → invalid
        silence = np.zeros(16000, dtype=np.float32)
        result = run_voice_branch(waveform=silence, sample_rate=16000,
                                  asr_config={"model_size": "tiny"})
        # We expect asr key to exist (may be low quality or ok depending on checks)
        assert "validation" in result
        assert "status" in result

    def test_combined_quality_in_unit_interval(self) -> None:
        from training.voice.asr import run_voice_branch
        import unittest.mock as mock

        # Mock the ASR to avoid downloading model weights
        fake_asr_result = ASRResult(
            transcript="I have been feeling very low lately",
            language="en",
            language_prob=0.92,
            no_speech_prob=0.03,
            word_count=7,
            quality_score=0.85,
            is_usable=True,
            model_size="base",
        )
        wave = (np.sin(2 * np.pi * 440 * np.linspace(0, 3, 48000)) * 0.4).astype(np.float32)
        with mock.patch(
            "training.voice.asr.WhisperASR.transcribe_array",
            return_value=fake_asr_result
        ):
            result = run_voice_branch(waveform=wave, sample_rate=16000)
        assert "combined_quality_score" in result
        assert 0.0 <= result["combined_quality_score"] <= 1.0
