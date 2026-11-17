"""
Facial Expression Analyzer — MediaPipe Face Landmarker with 52 blendshapes.

Uses MediaPipe's FaceLandmarker with 52 face blendshapes grounded in the
Facial Action Coding System (FACS) for transparent, continuous facial movement
and affect analysis.

Architecture & Principles:
  1. MediaPipe FaceLandmarker -> 478 facial landmarks + 52 blendshapes.
  2. FACS Action Unit (AU) extraction (AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU15, AU17, AU20, AU24, AU26).
  3. Continuous evidence accumulation with mutual inhibitory penalties.
  4. Conflict detection: non-Duchenne smile with brow furrow / distress (grimace/tension).
  5. Uncertainty quantification: ambiguity detection (close competing scores), resting baseline detection.
  6. Separation of raw continuous prominence [0, 1] from normalized relative distribution [0, 1].
  7. Non-diagnostic honesty: facial expressions measure surface muscle activity and affect, NOT depression diagnoses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("emotiondep.face_expression")

# Resolve model path relative to this file's location
_FILE_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _FILE_DIR.parent.parent / "models" / "face_landmarker.task"

# ---------------------------------------------------------------------------
# Auxiliary expression weights (for experimental visual expression score)
# Used ONLY for auxiliary expression scoring, NEVER as a clinical diagnosis.
# ---------------------------------------------------------------------------
DEPRESSION_WEIGHTS = {
    "sad": 0.35,
    "fear": 0.15,
    "angry": 0.15,
    "disgust": 0.10,
    "happy": -0.25,     # Protective / positive affect
    "surprise": -0.10,  # Alertness
    "neutral": 0.05,    # Flat affect weighting
}

ALL_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ---------------------------------------------------------------------------
# MediaPipe blendshape keys mapping
# ---------------------------------------------------------------------------
BLEND_SHAPES_KEYS = {
    "eye_blink_left": "eyeBlinkLeft",
    "eye_blink_right": "eyeBlinkRight",
    "eye_look_down_left": "eyeLookDownLeft",
    "eye_look_down_right": "eyeLookDownRight",
    "eye_look_in_left": "eyeLookInLeft",
    "eye_look_in_right": "eyeLookInRight",
    "eye_look_out_left": "eyeLookOutLeft",
    "eye_look_out_right": "eyeLookOutRight",
    "eye_look_up_left": "eyeLookUpLeft",
    "eye_look_up_right": "eyeLookUpRight",
    "eye_squint_left": "eyeSquintLeft",
    "eye_squint_right": "eyeSquintRight",
    "eye_wide_left": "eyeWideLeft",
    "eye_wide_right": "eyeWideRight",
    "cheek_squint_left": "cheekSquintLeft",
    "cheek_squint_right": "cheekSquintRight",
    "jaw_forward": "jawForward",
    "jaw_left": "jawLeft",
    "jaw_open": "JawOpen",
    "jaw_right": "jawRight",
    "brow_down_left": "browDownLeft",
    "brow_down_right": "browDownRight",
    "brow_inner_up": "browInnerUp",
    "brow_outer_up_left": "browOuterUpLeft",
    "brow_outer_up_right": "browOuterUpRight",
    "mouth_close": "mouthClose",
    "mouth_frown_left": "mouthFrownLeft",
    "mouth_frown_right": "mouthFrownRight",
    "mouth_left": "mouthLeft",
    "mouth_right": "mouthRight",
    "mouth_roll_lower": "mouthRollLower",
    "mouth_roll_upper": "mouthRollUpper",
    "mouth_shrug_lower": "mouthShrugLower",
    "mouth_shrug_upper": "mouthShrugUpper",
    "mouth_smile_left": "mouthSmileLeft",
    "mouth_smile_right": "mouthSmileRight",
    "mouth_pucker": "mouthPucker",
    "mouth_funnel": "mouthFunnel",
    "mouth_dimple_left": "mouthDimpleLeft",
    "mouth_dimple_right": "mouthDimpleRight",
    "mouth_stretch_left": "mouthStretchLeft",
    "mouth_stretch_right": "mouthStretchRight",
    "mouth_press_left": "mouthPressLeft",
    "mouth_press_right": "mouthPressRight",
    "mouth_upper_up_left": "mouthUpperUpLeft",
    "mouth_upper_up_right": "mouthUpperUpRight",
    "nose_sneer_left": "noseSneerLeft",
    "nose_sneer_right": "noseSneerRight",
}


def classify_blendshapes(blendshapes: Dict[str, float]) -> Dict[str, Any]:
    """Classify facial expressions from 52 MediaPipe blendshapes grounded in FACS.

    FACS Action Units extracted:
      - AU1 (browInnerUp): Inner brow raise
      - AU2 (browOuterUpLeft/Right): Outer brow raise
      - AU4 (browDownLeft/Right): Brow lowerer / furrow (corrugator)
      - AU5 (eyeWideLeft/Right): Upper eyelid raise
      - AU6 (cheekSquintLeft/Right): Cheek raise (orbicularis oculi, Duchenne marker)
      - AU7 (eyeSquintLeft/Right): Eyelid tightener
      - AU9 (noseSneerLeft/Right): Nose wrinkler
      - AU10 (mouthUpperUpLeft/Right): Upper lip raiser
      - AU12 (mouthSmileLeft/Right): Lip corner puller (zygomaticus major)
      - AU15 (mouthFrownLeft/Right): Lip corner depressor
      - AU17 (mouthShrugLower/Upper): Chin raiser
      - AU20 (mouthStretchLeft/Right): Lip stretcher (risorius)
      - AU24 (mouthPressLeft/Right): Lip pressor
      - AU26 (JawOpen / jawOpen): Jaw drop

    Returns:
        Dict containing:
          - all_emotions: Dict[str, float] normalized relative distribution (sums to 1.0)
          - raw_scores: Dict[str, float] continuous evidence scores
          - dominant_emotion: str (standard 7-emotion identifier)
          - dominant_expression: str (human-readable title or pattern)
          - uncertainty_label: str (resting baseline, ambiguous, grimace/tension, or dominant)
          - expression_prominence: float [0, 1] (raw continuous activation strength)
          - observed_movement: Dict[str, float] key AU activations
          - is_conflict: bool
          - is_ambiguous: bool
          - is_resting: bool
    """
    def bs_val(name: str, alt_names: Tuple[str, ...] = ()) -> float:
        if name in blendshapes:
            return float(blendshapes[name])
        for alt in alt_names:
            if alt in blendshapes:
                return float(blendshapes[alt])
        lower = name.lower()
        for k, v in blendshapes.items():
            if k.lower() == lower:
                return float(v)
        return 0.0

    # Extract Action Units
    brow_down = (bs_val("browDownLeft") + bs_val("browDownRight")) / 2.0  # AU4
    brow_inner_up = bs_val("browInnerUp")  # AU1
    brow_outer_up = (bs_val("browOuterUpLeft") + bs_val("browOuterUpRight")) / 2.0  # AU2

    eye_wide = (bs_val("eyeWideLeft") + bs_val("eyeWideRight")) / 2.0  # AU5
    eye_squint = (bs_val("eyeSquintLeft") + bs_val("eyeSquintRight")) / 2.0  # AU7
    cheek_squint = (bs_val("cheekSquintLeft") + bs_val("cheekSquintRight")) / 2.0  # AU6 (Duchenne)

    smile = (bs_val("mouthSmileLeft") + bs_val("mouthSmileRight")) / 2.0  # AU12
    frown = (bs_val("mouthFrownLeft") + bs_val("mouthFrownRight")) / 2.0  # AU15
    mouth_stretch = (bs_val("mouthStretchLeft") + bs_val("mouthStretchRight")) / 2.0  # AU20
    mouth_press = (bs_val("mouthPressLeft") + bs_val("mouthPressRight")) / 2.0  # AU24
    mouth_shrug = (bs_val("mouthShrugLower") + bs_val("mouthShrugUpper")) / 2.0  # AU17
    nose_sneer = (bs_val("noseSneerLeft") + bs_val("noseSneerRight")) / 2.0  # AU9
    upper_lip_up = (bs_val("mouthUpperUpLeft") + bs_val("mouthUpperUpRight")) / 2.0  # AU10
    jaw_open = bs_val("jawOpen", ("JawOpen",))  # AU26

    observed_movement = {
        "smile_intensity": round(smile, 4),
        "cheek_squint": round(cheek_squint, 4),
        "brow_furrow": round(brow_down, 4),
        "inner_brow_raise": round(brow_inner_up, 4),
        "outer_brow_raise": round(brow_outer_up, 4),
        "mouth_frown": round(frown, 4),
        "mouth_stretch": round(mouth_stretch, 4),
        "eye_wideness": round(eye_wide, 4),
        "eye_squint": round(eye_squint, 4),
        "nose_wrinkle": round(nose_sneer, 4),
        "jaw_openness": round(jaw_open, 4),
    }

    # Empty or zero blendshapes check
    total_movement = sum(observed_movement.values())
    if not blendshapes or total_movement < 0.005:
        all_emotions = {e: 0.0 for e in ALL_EMOTIONS}
        all_emotions["neutral"] = 1.0
        return {
            "all_emotions": all_emotions,
            "raw_scores": {e: 0.0 for e in ALL_EMOTIONS},
            "dominant_emotion": "neutral",
            "dominant_expression": "Neutral / Resting Face",
            "uncertainty_label": "No movement detected (Resting baseline)",
            "expression_prominence": 0.0,
            "observed_movement": observed_movement,
            "is_conflict": False,
            "is_ambiguous": False,
            "is_resting": True,
        }

    # Conflict detection: Mouth stretch / smile co-occurring with deep distress muscles (grimace / crying tension)
    # A genuine smile does NOT co-occur with strong AU4 furrow, AU1 inner brow grief, or AU9 sneer.
    # Note: If eyes are wide (AU5 eyeWide >= 0.25) without strong furrow (AU4 <= 0.18),
    # it represents Fear or Surprise rather than a grimace conflict.
    is_conflict = (
        (smile > 0.18 or (mouth_stretch > 0.25 and eye_wide < 0.25))
        and (brow_down > 0.18 or (brow_inner_up > 0.20 and eye_wide < 0.25) or nose_sneer > 0.20 or frown > 0.15)
    )

    raw_scores: Dict[str, float] = {}

    # 1. Happy (FACS: AU12 + AU6)
    if is_conflict:
        raw_happy = 0.0
    else:
        duchenne_eye = max(cheek_squint, eye_squint * 0.7)
        base_happy = 0.65 * smile + 0.35 * duchenne_eye
        happy_penalties = 2.0 * brow_down + 1.5 * brow_inner_up + 1.5 * nose_sneer + 1.8 * frown
        raw_happy = max(0.0, base_happy - happy_penalties)
    raw_scores["happy"] = round(raw_happy, 4)

    # 2. Sad (FACS: AU1 + AU4 + AU15 + AU17)
    # Darwin's "omega brow" / grief muscle: co-activation of AU1 and AU4
    omega_brow = min(brow_inner_up, brow_down) * 1.5
    mouth_sad = max(frown, mouth_shrug * 0.8)
    base_sad = 0.30 * brow_inner_up + 0.20 * brow_down + 0.35 * mouth_sad + 0.15 * omega_brow

    if is_conflict:
        base_sad += 0.25 * mouth_stretch + 0.15 * brow_down
        sad_penalties = 0.0
    else:
        sad_penalties = max(0.0, 1.2 * (smile - frown)) if (smile > frown and brow_down < 0.15) else 0.0

    raw_sad = max(0.0, base_sad - sad_penalties)
    raw_scores["sad"] = round(raw_sad, 4)

    # 3. Angry (FACS: AU4 + AU7 + AU24)
    base_angry = 0.45 * brow_down + 0.30 * eye_squint + 0.25 * mouth_press
    angry_penalties = 0.5 * brow_inner_up + 1.5 * smile + 0.5 * jaw_open
    raw_angry = max(0.0, base_angry - angry_penalties)
    raw_scores["angry"] = round(raw_angry, 4)

    # 4. Fear (FACS: AU1 + AU2 + AU5 + AU20)
    base_fear = 0.40 * eye_wide + 0.30 * brow_inner_up + 0.15 * brow_outer_up + 0.15 * mouth_stretch
    fear_penalties = 0.6 * brow_down + 1.2 * smile
    raw_fear = max(0.0, base_fear - fear_penalties)
    raw_scores["fear"] = round(raw_fear, 4)

    # 5. Disgust (FACS: AU9 + AU10 + AU4)
    base_disgust = 0.50 * nose_sneer + 0.35 * upper_lip_up + 0.15 * brow_down
    disgust_penalties = 1.2 * smile + 0.8 * eye_wide
    raw_disgust = max(0.0, base_disgust - disgust_penalties)
    raw_scores["disgust"] = round(raw_disgust, 4)

    # 6. Surprise (FACS: AU1 + AU2 + AU5 + AU26)
    base_surprise = 0.30 * brow_outer_up + 0.25 * brow_inner_up + 0.25 * eye_wide + 0.20 * jaw_open
    surprise_penalties = 2.0 * brow_down + 1.2 * frown + 0.5 * smile
    raw_surprise = max(0.0, base_surprise - surprise_penalties)
    raw_scores["surprise"] = round(raw_surprise, 4)

    # 7. Neutral (Resting Baseline)
    non_neutral_max = max(raw_scores[e] for e in ALL_EMOTIONS if e != "neutral")
    if non_neutral_max < 0.12:
        raw_neutral = max(0.50, 1.0 - non_neutral_max * 3.5)
    else:
        raw_neutral = max(0.05, 0.35 - non_neutral_max * 0.40)
    raw_scores["neutral"] = round(raw_neutral, 4)

    # Normalize relative distribution across 7 emotions
    total_score = sum(raw_scores.values())
    if total_score > 0:
        all_emotions = {k: round(v / total_score, 4) for k, v in raw_scores.items()}
        # Adjust small rounding difference to sum to exactly 1.0
        diff = round(1.0 - sum(all_emotions.values()), 4)
        if diff != 0:
            all_emotions["neutral"] = round(all_emotions["neutral"] + diff, 4)
    else:
        all_emotions = {e: 0.0 for e in ALL_EMOTIONS}
        all_emotions["neutral"] = 1.0

    # Categorization & Uncertainty Analysis
    is_resting = non_neutral_max < 0.12
    is_ambiguous = False

    if is_conflict:
        dominant_expression = "Tension / Grimace (Mixed affect)"
        uncertainty_label = "Conflicting indicators (Mouth stretch/smile with brow furrow or distress)"
        dominant_emotion = "sad" if raw_scores["sad"] >= raw_scores["angry"] else "angry"
        expression_prominence = max(brow_down, brow_inner_up, frown, smile)
    elif is_resting:
        dominant_expression = "Neutral / Resting Face"
        uncertainty_label = "Low facial movement (Resting baseline)"
        dominant_emotion = "neutral"
        expression_prominence = raw_neutral
    else:
        # Rank non-neutral emotions
        non_neutral_sorted = sorted(
            [(e, raw_scores[e]) for e in ALL_EMOTIONS if e != "neutral"],
            key=lambda x: x[1],
            reverse=True,
        )
        top1_emo, top1_val = non_neutral_sorted[0]
        top2_emo, top2_val = non_neutral_sorted[1]

        if top1_val >= 0.15 and (top1_val - top2_val) < 0.08:
            is_ambiguous = True
            dominant_expression = "Ambiguous / Mixed Expression"
            uncertainty_label = f"Mixed indicators ({top1_emo.capitalize()} / {top2_emo.capitalize()})"
            dominant_emotion = top1_emo
            expression_prominence = top1_val
        else:
            dominant_expression = top1_emo.capitalize()
            uncertainty_label = "Dominant movement pattern"
            dominant_emotion = top1_emo
            expression_prominence = top1_val

    expression_prominence = round(max(0.0, min(1.0, float(expression_prominence))), 4)

    return {
        "all_emotions": all_emotions,
        "raw_scores": raw_scores,
        "dominant_emotion": dominant_emotion,
        "dominant_expression": dominant_expression,
        "uncertainty_label": uncertainty_label,
        "expression_prominence": expression_prominence,
        "observed_movement": observed_movement,
        "is_conflict": is_conflict,
        "is_ambiguous": is_ambiguous,
        "is_resting": is_resting,
    }


class FaceMeshAnalyzer:
    """MediaPipe FaceLandmarker based facial expression analyzer."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize MediaPipe FaceLandmarker.

        Args:
            model_path: Path to the face landmarker model file.
                       If None, uses the default path resolved relative to this file.
        """
        self._landmarker = None
        self._model_path = model_path if model_path else str(_MODEL_PATH)
        self._initialize()

    def _initialize(self):
        """Initialize MediaPipe FaceLandmarker with blendshape output."""
        try:
            import mediapipe as mp

            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._model_path),
                output_face_blendshapes=True,
                num_faces=1,
            )

            self._landmarker = FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize MediaPipe FaceLandmarker: %s", e)
            raise

    def _get_blendshapes(self, img: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract face blendshapes from image.

        Returns:
            Dict mapping blendshape names to scores [0, 1],
            or None if no face detected.
        """
        if self._landmarker is None:
            return None

        import mediapipe as mp

        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        result = self._landmarker.detect(mp_image)

        if not result.face_blendshapes:
            return None

        # Convert blendshapes to dict
        blendshapes = {}
        for bs in result.face_blendshapes[0]:
            blendshapes[bs.category_name] = bs.score

        return blendshapes

    def _classify_expression(self, blendshapes: Dict[str, float]) -> Dict[str, float]:
        """Classify facial expression from blendshapes (backward compatible method)."""
        res = classify_blendshapes(blendshapes)
        return res["all_emotions"]

    def analyze_single_frame(self, frame_bytes: bytes) -> Dict[str, Any]:
        """Analyze facial expression from a single frame.

        Args:
            frame_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Dict with:
                - depression_prob: float [0.05, 0.95]
                - dominant_emotion: str
                - dominant_expression: str
                - uncertainty_label: str
                - expression_prominence: float [0, 1]
                - all_emotions: dict[str, float] (7 emotions summing to 1.0)
                - raw_scores: dict[str, float]
                - observed_movement: dict[str, float]
                - is_conflict: bool
                - is_ambiguous: bool
                - is_resting: bool
                - face_detected: bool
                - blendshapes: dict with key blendshape values
        """
        # Decode image bytes
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {
                "depression_prob": 0.5,
                "dominant_emotion": "unknown",
                "dominant_expression": None,
                "uncertainty_label": "Invalid or unreadable image",
                "expression_prominence": 0.0,
                "all_emotions": {e: 0.0 for e in ALL_EMOTIONS},
                "raw_scores": {e: 0.0 for e in ALL_EMOTIONS},
                "observed_movement": {},
                "is_conflict": False,
                "is_ambiguous": False,
                "is_resting": False,
                "face_detected": False,
                "blendshapes": {},
            }

        # Get face blendshapes
        blendshapes = self._get_blendshapes(img)
        if blendshapes is None:
            return {
                "depression_prob": 0.5,
                "dominant_emotion": "no_face",
                "dominant_expression": None,
                "uncertainty_label": "No face detected in the image",
                "expression_prominence": 0.0,
                "all_emotions": {e: 0.0 for e in ALL_EMOTIONS},
                "raw_scores": {e: 0.0 for e in ALL_EMOTIONS},
                "observed_movement": {},
                "is_conflict": False,
                "is_ambiguous": False,
                "is_resting": False,
                "face_detected": False,
                "blendshapes": {},
            }

        # Classify expression using FACS grounding
        classification = classify_blendshapes(blendshapes)
        emotions = classification["all_emotions"]

        # Extract key blendshapes for reporting
        key_blendshapes = {
            "smile": round((blendshapes.get("mouthSmileLeft", 0) + blendshapes.get("mouthSmileRight", 0)) / 2, 4),
            "frown": round((blendshapes.get("mouthFrownLeft", 0) + blendshapes.get("mouthFrownRight", 0)) / 2, 4),
            "brow_down": round((blendshapes.get("browDownLeft", 0) + blendshapes.get("browDownRight", 0)) / 2, 4),
            "brow_up": round(blendshapes.get("browInnerUp", 0), 4),
            "eye_wide": round((blendshapes.get("eyeWideLeft", 0) + blendshapes.get("eyeWideRight", 0)) / 2, 4),
            "eye_squint": round((blendshapes.get("eyeSquintLeft", 0) + blendshapes.get("eyeSquintRight", 0)) / 2, 4),
            "jaw_open": round(blendshapes.get("JawOpen", blendshapes.get("jawOpen", 0)), 4),
        }

        # Calculate auxiliary visual expression score
        depression_score = sum(
            emotions.get(emotion, 0) * weight
            for emotion, weight in DEPRESSION_WEIGHTS.items()
        )
        # Sigmoid mapping to [0.05, 0.95]
        depression_prob = float(1.0 / (1.0 + np.exp(-depression_score * 4.0)))
        depression_prob = 0.05 + depression_prob * 0.90

        return {
            "depression_prob": round(depression_prob, 4),
            "dominant_emotion": classification["dominant_emotion"],
            "dominant_expression": classification["dominant_expression"],
            "uncertainty_label": classification["uncertainty_label"],
            "expression_prominence": classification["expression_prominence"],
            "all_emotions": classification["all_emotions"],
            "raw_scores": classification["raw_scores"],
            "observed_movement": classification["observed_movement"],
            "is_conflict": classification["is_conflict"],
            "is_ambiguous": classification["is_ambiguous"],
            "is_resting": classification["is_resting"],
            "face_detected": True,
            "blendshapes": key_blendshapes,
        }

    def analyze_multiple_frames(self, frames: List[bytes]) -> Dict[str, Any]:
        """Analyze multiple webcam frames and aggregate results."""
        if not frames:
            return {
                "depression_prob": 0.5,
                "prediction": "Non-Depressed",
                "confidence": 0.0,
                "risk_level": "low",
                "face_detected_count": 0,
                "total_frames": 0,
                "emotion_distribution": {e: 0.0 for e in ALL_EMOTIONS},
                "dominant_emotion_overall": "unknown",
                "dominant_expression": None,
                "uncertainty_label": "No frames provided",
                "avg_expression_prominence": 0.0,
                "observed_movement": {},
            }

        depression_probs = []
        emotion_totals = {e: 0.0 for e in ALL_EMOTIONS}
        movement_totals: Dict[str, float] = {}
        prominences = []
        expression_counts: Dict[str, int] = {}
        face_count = 0

        for frame_bytes in frames:
            try:
                result = self.analyze_single_frame(frame_bytes)
                if result["face_detected"]:
                    depression_probs.append(result["depression_prob"])
                    prominences.append(result["expression_prominence"])
                    face_count += 1
                    for emotion, score in result["all_emotions"].items():
                        emotion_totals[emotion] = emotion_totals.get(emotion, 0.0) + score
                    for k, v in result.get("observed_movement", {}).items():
                        movement_totals[k] = movement_totals.get(k, 0.0) + v
                    expr = result.get("dominant_expression", "Neutral")
                    expression_counts[expr] = expression_counts.get(expr, 0) + 1
            except Exception as e:
                logger.debug("Frame analysis failed: %s", e)
                continue

        if not depression_probs:
            return {
                "depression_prob": 0.5,
                "prediction": "Non-Depressed",
                "confidence": 0.0,
                "risk_level": "low",
                "face_detected_count": 0,
                "total_frames": len(frames),
                "emotion_distribution": {e: 0.0 for e in ALL_EMOTIONS},
                "dominant_emotion_overall": "no_face",
                "dominant_expression": None,
                "uncertainty_label": "No face detected in video frames",
                "avg_expression_prominence": 0.0,
                "observed_movement": {},
            }

        avg_prob = float(np.mean(depression_probs))
        std_prob = float(np.std(depression_probs))
        consistency = max(0.0, 1.0 - std_prob * 3)
        avg_prominence = float(np.mean(prominences)) if prominences else 0.0
        confidence = round(avg_prominence * 100 * consistency, 1)

        depression_prob = max(0.05, min(0.95, avg_prob))
        if depression_prob >= 0.7:
            risk_level = "high"
        elif depression_prob >= 0.5:
            risk_level = "moderate"
        else:
            risk_level = "low"

        avg_emotions = {
            k: round(v / face_count, 4)
            for k, v in emotion_totals.items()
        }
        avg_movement = {
            k: round(v / face_count, 4)
            for k, v in movement_totals.items()
        }
        dominant_emotion = max(avg_emotions, key=avg_emotions.get) if any(avg_emotions.values()) else "unknown"
        dominant_expression = max(expression_counts, key=expression_counts.get) if expression_counts else dominant_emotion.capitalize()

        return {
            "depression_prob": round(depression_prob, 4),
            "prediction": "Depressed" if depression_prob >= 0.5 else "Non-Depressed",
            "confidence": round(confidence, 1),
            "risk_level": risk_level,
            "face_detected_count": face_count,
            "total_frames": len(frames),
            "emotion_distribution": avg_emotions,
            "dominant_emotion_overall": dominant_emotion,
            "dominant_expression": dominant_expression,
            "uncertainty_label": f"Temporal consistency {round(consistency * 100)}%",
            "avg_expression_prominence": round(avg_prominence, 4),
            "observed_movement": avg_movement,
            "std_across_frames": round(std_prob, 4),
        }

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


# ---------------------------------------------------------------------------
# Module-level singleton for lazy initialization
# ---------------------------------------------------------------------------

_analyzer: Optional[FaceMeshAnalyzer] = None


def _get_analyzer() -> FaceMeshAnalyzer:
    """Get or create the singleton FaceMeshAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FaceMeshAnalyzer()
    return _analyzer


def analyze_single_frame(frame_bytes: bytes) -> Dict[str, Any]:
    """Analyze facial expression from a single frame (module-level function)."""
    return _get_analyzer().analyze_single_frame(frame_bytes)


def analyze_multiple_frames(frames: List[bytes]) -> Dict[str, Any]:
    """Analyze multiple frames and aggregate results (module-level function)."""
    return _get_analyzer().analyze_multiple_frames(frames)

