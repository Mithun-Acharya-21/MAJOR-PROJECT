"""Phase 17 — Vision branch explainability: Grad-CAM API layer.

Wraps the Phase 8 compute_gradcam() function with:
  - API-ready serialisation (heatmap as base64 PNG)
  - Quality metadata from the face pipeline
  - Mandatory disclaimer enforcement
  - Overlay generation (optional, for UI rendering)

All outputs label themselves as EMOTION-MODEL visualisations.
They must NEVER be described as depression-risk evidence.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np

DISCLAIMER = (
    "This heatmap shows which face regions influenced the emotion prediction. "
    "It is an emotion-model visualisation, not a depression-risk indicator, "
    "and must not be interpreted as clinical evidence."
)


def gradcam_to_base64_png(
    heatmap: np.ndarray,
    original_image_rgb: Optional[np.ndarray] = None,
    alpha: float = 0.45,
    height: int = 224,
    width: int = 224,
) -> str:
    """Convert a Grad-CAM heatmap to a base64-encoded PNG for API transport.

    Parameters
    ----------
    heatmap           : (H, W) float32 in [0, 1]
    original_image_rgb: (H, W, 3) uint8 RGB image; if provided, overlay is returned
    alpha             : Blend weight for overlay
    height, width     : Output image dimensions

    Returns
    -------
    Base64-encoded PNG string (data URI prefix NOT included)
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required") from exc

    h_resized = cv2.resize(heatmap, (width, height))
    heatmap_uint8 = (h_resized * 255).clip(0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    if original_image_rgb is not None:
        orig = cv2.resize(original_image_rgb, (width, height))
        if orig.dtype != np.uint8:
            orig = (orig * 255).clip(0, 255).astype(np.uint8)
        output = (alpha * color_map_rgb + (1 - alpha) * orig).astype(np.uint8)
    else:
        output = color_map_rgb

    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(output).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        # Fallback: use cv2 imencode
        ok, buf_arr = cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return base64.b64encode(buf_arr.tobytes()).decode("utf-8")


def build_vision_explanation(
    gradcam_result: dict,
    face_quality_metadata: Optional[dict] = None,
    include_heatmap_png: bool = True,
    original_image_rgb: Optional[np.ndarray] = None,
) -> dict:
    """Assemble the vision explanation bundle for the API.

    Parameters
    ----------
    gradcam_result       : Output of compute_gradcam() from Phase 8
    face_quality_metadata: Output of FaceResult.to_metadata() from Phase 7
    include_heatmap_png  : Whether to encode the heatmap as base64 PNG
    original_image_rgb   : Optional RGB image for overlay

    Returns
    -------
    API-ready dict with emotion label, probabilities, heatmap, and disclaimer
    """
    heatmap = gradcam_result.get("heatmap")
    heatmap_b64 = None
    if include_heatmap_png and heatmap is not None:
        try:
            heatmap_b64 = gradcam_to_base64_png(heatmap, original_image_rgb)
        except Exception as exc:
            heatmap_b64 = None

    result = {
        "type": "gradcam",
        "emotion_label": gradcam_result.get("emotion_label"),
        "target_class": gradcam_result.get("target_class"),
        "heatmap_base64_png": heatmap_b64,
        "disclaimer": gradcam_result.get("disclaimer", DISCLAIMER),
    }

    if face_quality_metadata:
        result["face_quality"] = {
            "quality_score": face_quality_metadata.get("quality_score"),
            "reliability": face_quality_metadata.get("reliability"),
            "rejection_reasons": face_quality_metadata.get("rejection_reasons", []),
            "note": (
                "A lower quality score reduces this modality's contribution "
                "to the fusion prediction."
            ),
        }

    return result
