"""Face detection, alignment, and quality scoring for the vision pipeline.

This module operates on individual image files or NumPy arrays.  It never
stores the raw face crop by default – it returns a dict of metadata and an
optional normalised tensor for immediate in-memory use.

Quality scoring is deterministic and based on image statistics (blur,
brightness, clipping) rather than any ML model, so it runs offline.

All outputs carry a ``quality_score`` in [0, 1] and a ``reliability``
boolean.  A missing face, multiple faces, occlusion proxy, blur, or poor
lighting all lower the quality score and set ``reliability = False``.

IMPORTANT: facial expressions detected here are *emotion* signals only.
They must never be labelled as depression indicators directly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Thresholds (tunable via config dict in FacePreprocessor)
DEFAULT_CONFIG: dict = {
    "target_size": (224, 224),           # (H, W) for CNN input
    "blur_threshold": 80.0,              # Laplacian variance below → blurry
    "brightness_low": 30.0,              # Mean pixel value below → under-exposed
    "brightness_high": 225.0,            # Mean pixel value above → over-exposed
    "min_face_fraction": 0.02,           # Face area / image area minimum
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "padding_fraction": 0.15,            # Extra border around detected face ROI
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Everything the vision pipeline needs downstream."""

    detected: bool = False
    n_faces: int = 0
    quality_score: float = 0.0          # Composite [0, 1]
    reliability: bool = False           # True only if exactly one adequate face
    face_tensor: Optional[np.ndarray] = None   # (C, H, W) float32 in [0, 1] or None
    blur_score: float = 0.0            # Higher = sharper
    brightness: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) in original image
    rejection_reasons: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        """Return a JSON-serialisable summary, dropping the tensor."""
        return {
            "detected": self.detected,
            "n_faces": self.n_faces,
            "quality_score": round(self.quality_score, 4),
            "reliability": self.reliability,
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 2),
            "bbox": list(self.bbox) if self.bbox else None,
            "rejection_reasons": self.rejection_reasons,
        }


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class FacePreprocessor:
    """Detect, align, quality-score, and normalise a face from an image."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(
                "Failed to load Haar cascade. Ensure opencv-python is installed "
                "and FACE_CASCADE_PATH is valid."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_path(self, image_path: Path | str) -> FaceResult:
        """Load an image file and return a FaceResult."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            result = FaceResult()
            result.rejection_reasons.append("file_unreadable")
            return result
        return self.process_array(img_bgr, is_bgr=True)

    def process_bytes(self, data: bytes) -> FaceResult:
        """Decode in-memory image bytes and return a FaceResult."""
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            result = FaceResult()
            result.rejection_reasons.append("decode_failed")
            return result
        return self.process_array(img_bgr, is_bgr=True)

    def process_array(self, image: np.ndarray, is_bgr: bool = True) -> FaceResult:
        """Process a raw numpy image array (H×W×C)."""
        result = FaceResult()

        if is_bgr:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image.copy()

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._detect_faces(gray)
        result.n_faces = len(faces)

        if len(faces) == 0:
            result.rejection_reasons.append("no_face_detected")
            result.quality_score = 0.0
            return result

        if len(faces) > 1:
            result.rejection_reasons.append("multiple_faces")
            # Use the largest face but reduce quality
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]

        result.detected = True
        x, y, w, h = faces[0]
        result.bbox = (int(x), int(y), int(w), int(h))

        # Padding
        pad_x = int(w * self.cfg["padding_fraction"])
        pad_y = int(h * self.cfg["padding_fraction"])
        ih, iw = img_rgb.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + w + pad_x)
        y2 = min(ih, y + h + pad_y)
        crop = img_rgb[y1:y2, x1:x2]

        # Face area fraction check
        face_fraction = (w * h) / (iw * ih)
        if face_fraction < self.cfg["min_face_fraction"]:
            result.rejection_reasons.append("face_too_small")

        # Quality measures
        result.blur_score = self._blur_score(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
        result.brightness = float(np.mean(crop))

        if result.blur_score < self.cfg["blur_threshold"]:
            result.rejection_reasons.append("blurry")

        if result.brightness < self.cfg["brightness_low"]:
            result.rejection_reasons.append("under_exposed")
        elif result.brightness > self.cfg["brightness_high"]:
            result.rejection_reasons.append("over_exposed")

        # Composite quality score
        result.quality_score = self._composite_quality(result, len(faces) > 1)
        result.reliability = (
            result.detected
            and not result.rejection_reasons
        )

        # Normalised tensor only if reliable
        if result.detected:
            target_h, target_w = self.cfg["target_size"]
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
            tensor = resized.astype(np.float32) / 255.0          # (H, W, C)
            result.face_tensor = np.transpose(tensor, (2, 0, 1))  # (C, H, W)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg["scale_factor"],
            minNeighbors=self.cfg["min_neighbors"],
            minSize=(30, 30),
        )
        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return []
        return [tuple(int(v) for v in f) for f in faces]

    @staticmethod
    def _blur_score(gray_crop: np.ndarray) -> float:
        return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

    def _composite_quality(self, result: FaceResult, multi_face: bool) -> float:
        """Weighted combination of sub-scores, clamped to [0, 1]."""
        blur_norm = min(result.blur_score / 500.0, 1.0)
        bright_norm = 1.0 - abs(result.brightness - 128.0) / 128.0
        bright_norm = max(0.0, bright_norm)
        multi_penalty = 0.3 if multi_face else 0.0
        score = 0.5 * blur_norm + 0.5 * bright_norm - multi_penalty
        return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Video frame sampling
# ---------------------------------------------------------------------------

def sample_video_frames(
    video_path: Path | str,
    max_frames: int = 30,
    sample_fps: float = 1.0,
) -> list[np.ndarray]:
    """Extract up to ``max_frames`` BGR frames at ``sample_fps`` from a video.

    Returns an empty list if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(source_fps / sample_fps))
    frames: list[np.ndarray] = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def process_video(
    video_path: Path | str,
    preprocessor: FacePreprocessor | None = None,
    max_frames: int = 30,
    sample_fps: float = 1.0,
) -> dict:
    """Process a video file: sample frames, detect faces, aggregate quality.

    Returns:
        {
          "n_frames_sampled": int,
          "n_frames_with_face": int,
          "mean_quality_score": float,
          "min_quality_score": float,
          "reliable_frame_ratio": float,
          "frame_results": [FaceResult.to_metadata(), ...],
        }
    """
    if preprocessor is None:
        preprocessor = FacePreprocessor()

    frames = sample_video_frames(video_path, max_frames=max_frames, sample_fps=sample_fps)
    if not frames:
        return {
            "n_frames_sampled": 0,
            "n_frames_with_face": 0,
            "mean_quality_score": 0.0,
            "min_quality_score": 0.0,
            "reliable_frame_ratio": 0.0,
            "frame_results": [],
        }

    results = [preprocessor.process_array(f, is_bgr=True) for f in frames]
    n_with_face = sum(1 for r in results if r.detected)
    qualities = [r.quality_score for r in results]
    n_reliable = sum(1 for r in results if r.reliability)

    return {
        "n_frames_sampled": len(results),
        "n_frames_with_face": n_with_face,
        "mean_quality_score": round(float(np.mean(qualities)), 4),
        "min_quality_score": round(float(np.min(qualities)), 4),
        "reliable_frame_ratio": round(n_reliable / len(results), 4),
        "frame_results": [r.to_metadata() for r in results],
    }
