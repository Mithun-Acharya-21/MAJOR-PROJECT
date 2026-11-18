"""
Unit & Integration Tests for Doctor Portal & Voice Pitch Analysis
NeuroScreen Major Project - Phase 9

Tests:
    1. GET /api/doctor/sessions - Session list & cohort statistics
    2. GET /api/doctor/sessions/{session_id} - Detailed breakdown & modality evidence
    3. POST /api/doctor/sessions/{session_id}/notes - Clinician notes persistence
    4. GET /api/doctor/patient/{patient_id}/timeline - Longitudinal patient trajectory
    5. GET /api/doctor/analytics - Verified benchmark metrics across 8 models
    6. GET /api/doctor/datasets - Dataset provenance & taxonomy
    7. Voice Pitch Analysis - F0 pitch extraction, contour, speech & silence dynamics
    8. Multilingual & Kannada Audio - Acoustic invariance across speech patterns
    9. Error & Edge Cases - Corrupted, short, or invalid audio formats
"""

import io
import asyncio
import pytest
import numpy as np
import soundfile as sf
from fastapi import HTTPException

from backend.app.services.pitch_service import (
    analyze_voice_pitch_and_prosody,
    decode_audio_bytes,
)
from backend.app.doctor_routes import (
    get_doctor_sessions,
    get_doctor_session_detail,
    add_session_note,
    get_patient_timeline,
    get_doctor_model_analytics,
    get_doctor_dataset_information,
)


def _generate_synthetic_wav(
    duration: float = 2.0,
    sr: int = 16000,
    f0_mean: float = 175.0,
    f0_mod: float = 25.0,
    with_pause: bool = False,
) -> bytes:
    """Generates a synthetic speech-like WAV with vibrato or frequency modulation."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0_contour = f0_mean + f0_mod * np.sin(2 * np.pi * 2.0 * t)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr
    sig = 0.5 * np.sin(phase).astype(np.float32)

    if with_pause:
        # Insert a 400ms pause (0.8s to 1.2s)
        p_start = int(0.8 * sr)
        p_end = int(1.2 * sr)
        sig[p_start:p_end] = 0.001 * np.random.randn(p_end - p_start)

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def test_doctor_sessions_list():
    """Verify session list returns active sessions and clinical research fixtures."""
    res = get_doctor_sessions()
    assert "sessions" in res
    assert "total" in res
    assert res["total"] >= 1
    sessions = res["sessions"]

    first = sessions[0]
    assert "session_id" in first
    assert "patient_id" in first
    assert "status" in first
    assert "screening_score" in first
    assert "modalities_used" in first


def test_doctor_session_detail():
    """Verify session detail returns calibrated scores, symptom dimensions, and disclaimers."""
    res = get_doctor_session_detail("DEMO-PT101-S3")
    assert res["session_id"] == "DEMO-PT101-S3"
    assert res["patient_id"] == "PT-RESEARCH-101"
    assert res["status"] == "completed"

    s_res = res["screening_result"]
    assert "category" in s_res
    assert "score" in s_res
    assert "confidence" in s_res
    assert "uncertainty" in s_res
    assert s_res["score"] == 0.72

    # Verify symptom dimensions profile
    q_res = res["questionnaire_result"]
    assert "symptom_dimensions" in q_res
    dims = q_res["symptom_dimensions"]
    assert "mood" in dims
    assert "sleep" in dims
    assert "energy" in dims

    # Verify modality breakdown
    mods = res["modality_breakdown"]
    assert "text" in mods
    assert "voice" in mods
    assert "face" in mods
    assert "fusion" in mods

    # Verify clinical research disclaimer
    assert "disclaimer" in res
    assert "experimental screening result" in res["disclaimer"].lower()


def test_doctor_session_notes_persistence():
    """Verify clinician notes can be added and persisted."""
    test_session = "TEST-SESSION-NOTE-001"
    payload = {"note": "Patient exhibits psychomotor slowing during response.", "clinician": "Dr. Unit Test"}
    
    add_res = add_session_note(test_session, payload)
    assert add_res["success"] is True
    assert len(add_res["notes"]) >= 1
    assert any(n["note"] == "Patient exhibits psychomotor slowing during response." for n in add_res["notes"])


def test_doctor_patient_longitudinal_timeline():
    """Verify longitudinal trajectory across multiple sessions for a patient."""
    res = get_patient_timeline("PT-RESEARCH-101")
    assert res["patient_id"] == "PT-RESEARCH-101"
    assert res["total_sessions"] >= 2
    assert "timeline" in res or "sessions" in res or "trajectory" in res

    sessions = res.get("sessions") or res.get("timeline")
    assert len(sessions) >= 2
    first_s = sessions[0]
    assert "session_id" in first_s


def test_doctor_analytics_metrics():
    """Verify verified metrics across all 8 models, ROC curves, and confusion matrices."""
    res = get_doctor_model_analytics()
    assert "total_models_evaluated" in res
    assert res["total_models_evaluated"] == 8
    models = res["models"]
    assert len(models) == 8

    # Verify AVDD Multimodal Late Fusion HDS metrics (AUC = 0.8659)
    avdd_fusion = next(m for m in models if m["model_id"] == "M6-AVDD-FUSION-HDS")
    assert avdd_fusion["roc_auc"] == 0.8659
    assert avdd_fusion["balanced_accuracy"] == 0.7944
    assert avdd_fusion["brier_score"] == 0.1412

    # Verify EATD DistilBERT metrics
    eatd_text = next(m for m in models if m["model_id"] == "M1-EATD-DISTILBERT")
    assert eatd_text["roc_auc"] == 0.6756
    assert eatd_text["balanced_accuracy"] == 0.5978

    # Verify FER-2013 Affect Classifier status
    fer_model = next(m for m in models if m["model_id"] == "M8-FER2013-RESNET18")
    assert "Auxiliary Affect" in fer_model["status"]

    # Verify confusion matrices and calibration data
    assert "confusion_matrices" in res
    assert "M6_multimodal_hds" in res["confusion_matrices"]
    assert "calibration_data" in res


def test_doctor_datasets_provenance():
    """Verify official dataset metadata for EATD, AVDD, FER-2013, Harvard Tweets, DAIC-WOZ."""
    res = get_doctor_dataset_information()
    assert "datasets" in res
    datasets = res["datasets"]
    assert len(datasets) == 5

    abbrevs = {d["abbreviation"] for d in datasets}
    assert "EATD-Corpus" in abbrevs
    assert "AVDD-TCMI" in abbrevs
    assert "FER-2013" in abbrevs
    assert "Harvard Tweets" in abbrevs
    assert "DAIC-WOZ" in abbrevs

    # Verify Harvard Tweets quarantine note
    harvard = next(d for d in datasets if d["abbreviation"] == "Harvard Tweets")
    assert "BLOCKED FOR SUPERVISED TRAINING" in harvard["clinical_boundary"]
    assert "datasets" in res
    datasets = res["datasets"]
    assert len(datasets) == 5

    abbrevs = {d["abbreviation"] for d in datasets}
    assert "EATD-Corpus" in abbrevs
    assert "AVDD-TCMI" in abbrevs
    assert "FER-2013" in abbrevs
    assert "Harvard Tweets" in abbrevs
    assert "DAIC-WOZ" in abbrevs

    # Verify Harvard Tweets quarantine note
    harvard = next(d for d in datasets if d["abbreviation"] == "Harvard Tweets")
    assert "BLOCKED FOR SUPERVISED TRAINING" in harvard["clinical_boundary"]


def test_voice_pitch_analysis_synthetic_wav():
    """Test fundamental frequency extraction on synthetic vibrato tone."""
    audio_bytes = _generate_synthetic_wav(duration=2.0, f0_mean=175.0, f0_mod=25.0, with_pause=True)
    res = analyze_voice_pitch_and_prosody(audio_bytes)

    assert res["valid"] is True
    metrics = res["metrics"]
    assert 140.0 <= metrics["mean_pitch_hz"] <= 210.0
    assert metrics["pitch_range_hz"] > 20.0
    assert metrics["speech_duration_sec"] > 0.5
    assert len(res["pitch_contour"]) > 10

    # Verify contour point structure
    first_pt = res["pitch_contour"][0]
    assert "time_sec" in first_pt
    assert "pitch_hz" in first_pt
    assert "voiced" in first_pt


def test_voice_pitch_kannada_audio_simulation():
    """Test pitch extraction on multi-tone intonation pattern simulating Kannada vocal input."""
    # Synthesize multi-pitch vocalization simulating inflection patterns
    sr = 16000
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Kannada-like continuous tonal inflection: base 160 Hz with inflection peaks up to 215 Hz
    f0 = 160 + 30 * np.sin(2 * np.pi * 1.5 * t) + 15 * np.cos(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = (0.4 * np.sin(phase) + 0.2 * np.sin(2 * phase)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    buf.seek(0)

    res = analyze_voice_pitch_and_prosody(buf.read())
    assert res["valid"] is True
    assert 130.0 <= res["metrics"]["mean_pitch_hz"] <= 210.0
    assert res["metrics"]["voiced_frames_count"] > 20


def test_voice_pitch_invalid_and_empty_audio():
    """Test pitch analysis handles empty, corrupted, and too-short audio gracefully."""
    # 1. Empty bytes
    empty_res = analyze_voice_pitch_and_prosody(b"")
    assert empty_res["valid"] is False
    assert "error" in empty_res

    # 2. Corrupt bytes
    corrupt_res = analyze_voice_pitch_and_prosody(b"NOT_A_VALID_AUDIO_FILE_DATA_BYTES")
    assert corrupt_res["valid"] is False

    # 3. Too short audio (< 0.3s)
    short_audio = _generate_synthetic_wav(duration=0.15)
    short_res = analyze_voice_pitch_and_prosody(short_audio)
    assert short_res["valid"] is False
    assert "too short" in short_res["error"].lower()
