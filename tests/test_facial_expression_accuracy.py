"""
Unit tests for FACS-grounded facial expression classification accuracy and uncertainty handling.

Verifies:
1. Duchenne smile (AU12 + AU6) classifies as Happy.
2. Distressed face with grimace / tension (smile/mouth stretch + brow furrow/distress)
   is NOT classified as Happy.
3. Sad omega brow activation (AU1 + AU4 + AU15) classifies as Sad.
4. Angry brow furrow and lid tightening (AU4 + AU7 + AU24) classifies as Angry.
5. Disgust nose wrinkle and upper lip raise (AU9 + AU10) classifies as Disgust.
6. Fear wide eyes and raised inner brow (AU5 + AU1 + AU20) classifies as Fear.
7. Surprise raised outer/inner brows and jaw drop (AU1 + AU2 + AU26) classifies as Surprise.
8. Neutral resting face (all blendshapes below noise floor) classifies as Neutral / Resting.
9. Ambiguous expression handling (close scores flag ambiguity / mixed indicators).
10. Normalization and relative distribution strictly sum to 1.0.
11. Expression prominence is continuous and not forced to 100%.
12. Missing and zero blendshapes handled gracefully without crashing.
"""

import sys
from pathlib import Path
from typing import Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.face_expression_analyzer import classify_blendshapes, ALL_EMOTIONS


def test_happy_requires_smile_and_cheek_squint():
    """Verify genuine Duchenne smile (AU12 + AU6) classifies as Happy."""
    blendshapes = {
        "mouthSmileLeft": 0.65,
        "mouthSmileRight": 0.65,
        "cheekSquintLeft": 0.50,
        "cheekSquintRight": 0.50,
        "browDownLeft": 0.0,
        "browDownRight": 0.0,
        "mouthFrownLeft": 0.0,
        "mouthFrownRight": 0.0,
    }
    result = classify_blendshapes(blendshapes)

    assert result["dominant_emotion"] == "happy"
    assert result["dominant_expression"] == "Happy"
    assert result["all_emotions"]["happy"] > 0.50
    assert result["is_conflict"] is False
    assert result["expression_prominence"] > 0.40


def test_distressed_face_with_grimace_is_not_classified_as_happy():
    """Verify that a distressed/grimacing face (mouth stretch/smile + strong furrow) is NOT Happy.

    This directly tests the reported bug where a crying/distressed grimace
    was misclassified as Happy 100% with 65% prominence.
    """
    blendshapes = {
        "mouthSmileLeft": 0.28,
        "mouthSmileRight": 0.28,
        "mouthStretchLeft": 0.35,
        "mouthStretchRight": 0.35,
        "browDownLeft": 0.45,
        "browDownRight": 0.45,
        "browInnerUp": 0.40,
        "mouthFrownLeft": 0.25,
        "mouthFrownRight": 0.25,
    }
    result = classify_blendshapes(blendshapes)

    # Must NOT classify as Happy
    assert result["dominant_emotion"] != "happy"
    assert result["dominant_expression"] != "Happy"
    assert result["all_emotions"]["happy"] < 0.10

    # Conflict condition must trigger
    assert result["is_conflict"] is True
    assert "Tension / Grimace" in result["dominant_expression"] or result["dominant_expression"] == "Sad"
    assert "Conflicting indicators" in result["uncertainty_label"] or "Grimace" in result["uncertainty_label"]


def test_sad_omega_brow_activation():
    """Verify co-activation of AU1 (browInnerUp), AU4 (browDown), and AU15 (frown) classifies as Sad."""
    blendshapes = {
        "browInnerUp": 0.55,
        "browDownLeft": 0.45,
        "browDownRight": 0.45,
        "mouthFrownLeft": 0.40,
        "mouthFrownRight": 0.40,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }
    result = classify_blendshapes(blendshapes)

    assert result["dominant_emotion"] == "sad"
    assert "Sad" in result["dominant_expression"]
    assert result["all_emotions"]["sad"] > 0.40


def test_angry_brow_and_lid_tightening():
    """Verify AU4 (browDown) + AU7 (eyeSquint) + AU24 (mouthPress) classifies as Angry."""
    blendshapes = {
        "browDownLeft": 0.60,
        "browDownRight": 0.60,
        "eyeSquintLeft": 0.50,
        "eyeSquintRight": 0.50,
        "mouthPressLeft": 0.45,
        "mouthPressRight": 0.45,
        "browInnerUp": 0.02,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }
    result = classify_blendshapes(blendshapes)

    assert result["dominant_emotion"] == "angry"
    assert result["dominant_expression"] == "Angry"
    assert result["all_emotions"]["angry"] > 0.40


def test_disgust_nose_wrinkle():
    """Verify AU9 (noseSneer) + AU10 (upperLip) classifies as Disgust."""
    blendshapes = {
        "noseSneerLeft": 0.65,
        "noseSneerRight": 0.65,
        "mouthUpperUpLeft": 0.50,
        "mouthUpperUpRight": 0.50,
        "browDownLeft": 0.25,
        "browDownRight": 0.25,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }
    result = classify_blendshapes(blendshapes)

    assert result["dominant_emotion"] == "disgust"
    assert result["dominant_expression"] == "Disgust"
    assert result["all_emotions"]["disgust"] > 0.40


def test_fear_wide_eyes_and_brow_raise():
    """Verify AU5 (eyeWide) + AU1 (browInnerUp) + AU20 (mouthStretch) classifies as Fear."""
    blendshapes = {
        "eyeWideLeft": 0.60,
        "eyeWideRight": 0.60,
        "browInnerUp": 0.50,
        "browOuterUpLeft": 0.35,
        "browOuterUpRight": 0.35,
        "mouthStretchLeft": 0.30,
        "mouthStretchRight": 0.30,
        "browDownLeft": 0.0,
        "browDownRight": 0.0,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }
    result = classify_blendshapes(blendshapes)

    assert result["dominant_emotion"] == "fear"
    assert result["dominant_expression"] == "Fear"
    assert result["all_emotions"]["fear"] > 0.35


def test_surprise_raised_brows_and_jaw_drop():
    """Verify AU1 + AU2 (browOuterUp) + AU5 (eyeWide) + AU26 (JawOpen) classifies as Surprise."""
    blendshapes = {
        "browOuterUpLeft": 0.65,
        "browOuterUpRight": 0.65,
        "browInnerUp": 0.55,
        "eyeWideLeft": 0.50,
        "eyeWideRight": 0.50,
        "JawOpen": 0.55,
        "browDownLeft": 0.0,
        "browDownRight": 0.0,
        "mouthFrownLeft": 0.0,
        "mouthFrownRight": 0.0,
    }
    result = classify_blendshapes(blendshapes)

    assert result["dominant_emotion"] == "surprise"
    assert result["dominant_expression"] == "Surprise"
    assert result["all_emotions"]["surprise"] > 0.40


def test_neutral_resting_face():
    """Verify that when all blendshapes are below noise floor, face is Neutral / Resting."""
    blendshapes = {
        "mouthSmileLeft": 0.04,
        "mouthSmileRight": 0.04,
        "browDownLeft": 0.03,
        "browDownRight": 0.03,
        "mouthFrownLeft": 0.02,
        "mouthFrownRight": 0.02,
    }
    result = classify_blendshapes(blendshapes)

    assert result["dominant_emotion"] == "neutral"
    assert result["dominant_expression"] == "Neutral / Resting Face"
    assert result["is_resting"] is True
    assert "Resting baseline" in result["uncertainty_label"]
    assert result["all_emotions"]["neutral"] > 0.50


def test_ambiguous_expression_handling():
    """Verify that close competing scores between two emotions are flagged as Ambiguous."""
    # Balanced activation between Sad and Angry
    blendshapes = {
        "browDownLeft": 0.45,
        "browDownRight": 0.45,
        "eyeSquintLeft": 0.35,
        "eyeSquintRight": 0.35,
        "mouthPressLeft": 0.25,
        "mouthPressRight": 0.25,
        "browInnerUp": 0.15,
        "mouthFrownLeft": 0.25,
        "mouthFrownRight": 0.25,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }
    result = classify_blendshapes(blendshapes)

    # When scores are closely matched or conflicting, uncertainty is properly flagged
    assert result["is_ambiguous"] is True or result["is_conflict"] is True
    assert ("Ambiguous" in result["dominant_expression"] or
            "Tension" in result["dominant_expression"] or
            "Mixed" in result["uncertainty_label"] or
            "Conflicting" in result["uncertainty_label"])


def test_normalization_and_distribution_sums_to_1():
    """Verify relative emotion distribution strictly sums to 1.0 (100%)."""
    for test_bs in [
        {},
        {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8},
        {"browDownLeft": 0.5, "browInnerUp": 0.4, "mouthFrownLeft": 0.3},
        {"eyeWideLeft": 0.6, "JawOpen": 0.5},
        {k: 0.1 for k in ALL_EMOTIONS},
    ]:
        result = classify_blendshapes(test_bs)
        total = sum(result["all_emotions"].values())
        assert abs(total - 1.0) < 0.005, f"Expected sum 1.0, got {total}"


def test_prominence_is_not_forced_to_100_percent():
    """Verify a moderate muscle activation (e.g. smile 0.25) does NOT produce 100% prominence."""
    blendshapes = {
        "mouthSmileLeft": 0.25,
        "mouthSmileRight": 0.25,
        "cheekSquintLeft": 0.20,
        "cheekSquintRight": 0.20,
    }
    result = classify_blendshapes(blendshapes)

    # Prominence reflects continuous physical activation, NOT an inflated 100%
    assert result["expression_prominence"] < 0.50
    assert result["expression_prominence"] > 0.10


def test_missing_and_zero_blendshapes():
    """Verify empty or zero blendshapes return valid resting structure without exceptions."""
    empty_result = classify_blendshapes({})
    assert empty_result["dominant_emotion"] == "neutral"
    assert empty_result["dominant_expression"] == "Neutral / Resting Face"
    assert empty_result["expression_prominence"] == 0.0
    assert empty_result["is_resting"] is True
    assert empty_result["all_emotions"]["neutral"] == 1.0

    zeros = {k: 0.0 for k in ["mouthSmileLeft", "browDownLeft", "browInnerUp"]}
    zero_result = classify_blendshapes(zeros)
    assert zero_result["dominant_emotion"] == "neutral"
    assert zero_result["is_resting"] is True
