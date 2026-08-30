"""
Visualization — Generate explanation visualizations for reports and demos.

Exports:
  plot_attention_heatmap — visualize token attention weights
  plot_gradcam_overlay — visualize Grad-CAM heatmap on face images
  plot_comparison_bar — bar chart comparing model metrics
  save_explanation_plot — save visualization to file
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def plot_attention_heatmap(
    tokens: List[str],
    weights: List[float],
    output_path: Optional[str] = None,
) -> Optional[bytes]:
    """Create a text heatmap visualization of attention weights.

    Args:
        tokens: List of token strings.
        weights: List of attention weights (same length as tokens).
        output_path: If provided, save the plot to this path.

    Returns:
        PNG image bytes if output_path is None, else None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.4), 2))

        # Color each token by its weight
        norm_weights = np.array(weights)
        mn, mx = norm_weights.min(), norm_weights.max()
        if mx - mn > 1e-8:
            norm_weights = (norm_weights - mn) / (mx - mn)

        cmap = plt.cm.RdYlGn_r  # Red (high risk) → Green (low risk)
        colors = [cmap(w) for w in norm_weights]

        for i, (token, color) in enumerate(zip(tokens, colors)):
            ax.text(
                i, 0.5, token, ha="center", va="center",
                fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )

        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Token Attribution (LIME / Attention)", fontsize=12, pad=10)

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            return buf.getvalue()

    except ImportError:
        return None


def plot_gradcam_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    output_path: Optional[str] = None,
) -> Optional[bytes]:
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: (H, W, 3) RGB array.
        heatmap: (H, W) float array in [0, 1].
        alpha: Blending factor.
        output_path: If provided, save to this path.

    Returns:
        PNG image bytes if output_path is None, else None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(original_image)
        axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            return buf.getvalue()

    except ImportError:
        return None


def plot_comparison_bar(
    metrics: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> Optional[bytes]:
    """Create a grouped bar chart comparing model metrics.

    Args:
        metrics: Dict of model_name → {metric_name: value}.
                 e.g. {"Text Only": {"f1": 0.86, "auc": 0.89}, ...}
        output_path: If provided, save to this path.

    Returns:
        PNG image bytes if output_path is None, else None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model_names = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())

        x = np.arange(len(model_names))
        width = 0.8 / len(metric_names)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#14b8a6", "#f59e0b", "#a855f7", "#22c55e", "#ef4444"]

        for i, metric in enumerate(metric_names):
            values = [metrics[model].get(metric, 0) for model in model_names]
            ax.bar(
                x + i * width, values, width,
                label=metric.upper(),
                color=colors[i % len(colors)],
                alpha=0.85,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Depression Detection Metrics")
        ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            return buf.getvalue()

    except ImportError:
        return None


def save_explanation_plot(
    plot_bytes: bytes,
    output_path: str,
) -> None:
    """Save plot bytes to a file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(plot_bytes)
