"""
Explainability Module — Early Depression Detection

Provides human-readable explanations for model predictions:

  - **Text Explainability**: Converts raw attention weights into color-coded
    token highlights (green=protective → amber=neutral → red=risk indicator).
  - **Image Explainability**: Blends the original facial image with a Grad-CAM
    heatmap overlay and returns base64-encoded PNG data URLs for direct
    embedding in the frontend.

Exports:
  generate_text_explanation  — attention weights → colored token list
  generate_image_explanation — image bytes + heatmap → base64 overlays
  get_risk_color             — weight float → hex color string
"""

from __future__ import annotations

import base64
import io
from typing import List, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Color mapping utilities
# ---------------------------------------------------------------------------

def get_risk_color(weight: float) -> str:
    """Map a normalized attention weight (0–1) to a hex color.

    Gradient:
      0.0  → #22c55e  (green — low risk / protective)
      0.5  → #f59e0b  (amber — moderate)
      1.0  → #ef4444  (red — high risk indicator)

    Args:
        weight: Float in [0, 1].

    Returns:
        Hex color string, e.g. ``"#ef4444"``.
    """
    weight = max(0.0, min(1.0, weight))

    if weight < 0.5:
        # Green → Amber
        t = weight / 0.5
        r = int(34 + (245 - 34) * t)
        g = int(197 + (158 - 197) * t)
        b = int(94 + (11 - 94) * t)
    else:
        # Amber → Red
        t = (weight - 0.5) / 0.5
        r = int(245 + (239 - 245) * t)
        g = int(158 + (68 - 158) * t)
        b = int(11 + (68 - 11) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


def _normalize_weights(weights: list[float]) -> list[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    if not weights:
        return weights
    mn = min(weights)
    mx = max(weights)
    if mx - mn < 1e-8:
        return [0.5] * len(weights)
    return [(w - mn) / (mx - mn) for w in weights]


# ===================================================================
# Text Explanation
# ===================================================================

def generate_text_explanation(
    text: str,
    attention_weights: list[float],
    tokens: list[str],
) -> List[Dict[str, object]]:
    """Convert attention weights into a colored token-level explanation.

    Each token gets a normalized weight and a corresponding risk color
    that the frontend renders as a highlighted span.

    Args:
        text: Original preprocessed text.
        attention_weights: Raw attention weight per token from the model.
        tokens: Tokenized text (same length as attention_weights).

    Returns:
        List of dicts, each containing::

            {
                "token": str,        # display token
                "weight": float,     # normalized 0–1
                "color": str,        # hex color
            }
    """
    if not tokens or not attention_weights:
        return []

    # Ensure same length
    min_len = min(len(tokens), len(attention_weights))
    tokens = tokens[:min_len]
    weights = list(attention_weights[:min_len])

    # Filter out special tokens ([CLS], [SEP], [PAD], <s>, </s>)
    special = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    filtered = [
        (tok, w)
        for tok, w in zip(tokens, weights)
        if tok not in special
    ]

    if not filtered:
        return []

    filtered_tokens, filtered_weights = zip(*filtered)
    normalized = _normalize_weights(list(filtered_weights))

    result = []
    for tok, w in zip(filtered_tokens, normalized):
        # Clean sub-word markers
        display_token = tok.replace("##", "").replace("▁", "")
        if not display_token.strip():
            continue
        result.append({
            "token": display_token,
            "weight": round(w, 4),
            "color": get_risk_color(w),
        })

    return result


# ===================================================================
# Image Explanation
# ===================================================================

def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data URL string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _apply_heatmap_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Apply a JET-like colormap to a grayscale heatmap.

    Pure numpy implementation — no matplotlib/cv2 dependency for coloring.

    Args:
        heatmap: 2D array with values in [0, 1].

    Returns:
        RGB uint8 array of shape (H, W, 3).
    """
    h, w = heatmap.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # JET colormap approximation
    for i in range(h):
        for j in range(w):
            v = heatmap[i, j]
            if v < 0.25:
                r, g, b = 0, int(255 * (v / 0.25)), 255
            elif v < 0.5:
                r, g, b = 0, 255, int(255 * (1 - (v - 0.25) / 0.25))
            elif v < 0.75:
                r, g, b = int(255 * ((v - 0.5) / 0.25)), 255, 0
            else:
                r, g, b = 255, int(255 * (1 - (v - 0.75) / 0.25)), 0
            colored[i, j] = [r, g, b]

    return colored


def _apply_heatmap_fast(heatmap: np.ndarray) -> np.ndarray:
    """Vectorized heatmap colormap (much faster than pixel loop).

    Maps values 0→blue, 0.25→cyan, 0.5→green, 0.75→yellow, 1.0→red.
    """
    h_flat = heatmap.flatten()
    n = h_flat.shape[0]
    r = np.zeros(n, dtype=np.uint8)
    g = np.zeros(n, dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)

    # Region 1: [0, 0.25) — blue to cyan
    mask = h_flat < 0.25
    t = h_flat[mask] / 0.25
    r[mask] = 0
    g[mask] = (255 * t).astype(np.uint8)
    b[mask] = 255

    # Region 2: [0.25, 0.5) — cyan to green
    mask = (h_flat >= 0.25) & (h_flat < 0.5)
    t = (h_flat[mask] - 0.25) / 0.25
    r[mask] = 0
    g[mask] = 255
    b[mask] = (255 * (1 - t)).astype(np.uint8)

    # Region 3: [0.5, 0.75) — green to yellow
    mask = (h_flat >= 0.5) & (h_flat < 0.75)
    t = (h_flat[mask] - 0.5) / 0.25
    r[mask] = (255 * t).astype(np.uint8)
    g[mask] = 255
    b[mask] = 0

    # Region 4: [0.75, 1.0] — yellow to red
    mask = h_flat >= 0.75
    t = (h_flat[mask] - 0.75) / 0.25
    r[mask] = 255
    g[mask] = (255 * (1 - t)).astype(np.uint8)
    b[mask] = 0

    colored = np.stack([r, g, b], axis=-1).reshape(heatmap.shape[0], heatmap.shape[1], 3)
    return colored


def generate_image_explanation(
    original_image_bytes: bytes,
    gradcam_heatmap: np.ndarray,
) -> Dict[str, str]:
    """Produce base64 image data URLs for the original, heatmap, and overlay.

    Args:
        original_image_bytes: Raw image bytes of the uploaded file.
        gradcam_heatmap: Numpy array of shape (H, W) with values in [0, 1]
                         from the Grad-CAM generator.

    Returns:
        Dict with keys::

            {
                "original": "data:image/png;base64,...",
                "heatmap":  "data:image/png;base64,...",
                "overlay":  "data:image/png;base64,...",
            }
    """
    # Decode original image
    original = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    orig_size = (224, 224)
    original_resized = original.resize(orig_size, Image.LANCZOS)

    # Ensure heatmap is correct size
    if gradcam_heatmap.shape != (224, 224):
        heatmap_img = Image.fromarray(
            (gradcam_heatmap * 255).astype(np.uint8)
        ).resize(orig_size, Image.BILINEAR)
        gradcam_heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

    # Generate colored heatmap
    colored_heatmap = _apply_heatmap_fast(gradcam_heatmap)
    heatmap_pil = Image.fromarray(colored_heatmap)

    # Create overlay: blend original with heatmap
    overlay = Image.blend(original_resized, heatmap_pil, alpha=0.45)

    return {
        "original": _pil_to_base64(original_resized),
        "heatmap": _pil_to_base64(heatmap_pil),
        "overlay": _pil_to_base64(overlay),
    }
