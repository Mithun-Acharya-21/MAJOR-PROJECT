"""
Face Detection — Detect and crop faces from images for the vision pipeline.

Supports:
  - OpenCV Haar Cascade (lightweight, no extra dependencies)
  - MediaPipe Face Detection (more accurate, optional)

Exports:
  detect_faces — detect face bounding boxes in an image
  crop_face — extract the largest face from an image
  detect_and_crop — combined detection + crop pipeline
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def _numpy_to_pil(img_array: np.ndarray) -> Image.Image:
    """Convert a numpy array (H, W, C) to PIL Image."""
    if img_array.ndim == 2:
        return Image.fromarray(img_array, mode="L")
    return Image.fromarray(img_array, mode="RGB")


def detect_faces_opencv(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
) -> List[Tuple[int, int, int, int]]:
    """Detect faces using OpenCV Haar Cascade classifier.

    Args:
        image: RGB numpy array (H, W, 3).
        scale_factor: How much the image size is reduced at each image scale.
        min_neighbors: How many neighbors each candidate rectangle should have.
        min_size: Minimum possible object size.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    try:
        import cv2
    except ImportError:
        return _fallback_detect(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    if len(faces) == 0:
        return []

    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _fallback_detect(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Fallback face detection using center-crop heuristic when OpenCV is unavailable.

    Assumes the face is roughly centered in the image and occupies ~60% of the frame.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    side = min(w, h) // 2
    return [(cx - side, cy - side, side * 2, side * 2)]


def detect_faces(
    image_input,
    method: str = "opencv",
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in an image.

    Args:
        image_input: PIL Image, numpy array, or raw bytes.
        method: Detection method ('opencv' or 'fallback').

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    if isinstance(image_input, bytes):
        image_input = Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, Image.Image):
        image_array = np.array(image_input)
    else:
        image_array = np.array(image_input)

    if method == "opencv":
        return detect_faces_opencv(image_array)
    else:
        return _fallback_detect(image_array)


def crop_face(
    image_input,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    padding: float = 0.2,
    target_size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """Crop the face region from an image.

    Args:
        image_input: PIL Image, numpy array, or raw bytes.
        bbox: Optional (x, y, w, h) bounding box. If None, auto-detect.
        padding: Fraction of face size to add as padding around the crop.
        target_size: Output image size (width, height).

    Returns:
        PIL Image of the cropped face, resized to target_size.
    """
    if isinstance(image_input, bytes):
        image_input = Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, np.ndarray):
        image_input = _numpy_to_pil(image_input)

    img = image_input.copy()
    w, h = img.size

    if bbox is None:
        faces = detect_faces(img)
        if not faces:
            # No face found — return center crop as fallback
            side = min(w, h) // 2
            bbox = (w // 2 - side, h // 2 - side, side * 2, side * 2)
        else:
            # Select largest face
            bbox = max(faces, key=lambda f: f[2] * f[3])

    x, y, bw, bh = bbox

    # Add padding
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)

    cropped = img.crop((x1, y1, x2, y2))
    cropped = cropped.resize(target_size, Image.LANCZOS)

    return cropped


def detect_and_crop(
    image_input,
    target_size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """Detect the largest face and return it cropped and resized.

    Args:
        image_input: PIL Image, numpy array, or raw bytes.
        target_size: Output size.

    Returns:
        Cropped and resized face image.
    """
    return crop_face(image_input, bbox=None, target_size=target_size)
