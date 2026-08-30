"""
Inference — Standalone inference helpers for the trained model.

Provides a simple API for running predictions outside of the web app:
  - Load a trained checkpoint
  - Run text analysis
  - Run image analysis
  - Run multimodal analysis

Usage:
    python app/inference.py --text "I feel hopeless" --checkpoint models/checkpoints/fusion_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def load_model(checkpoint_dir: str, device: str = "auto"):
    """Load a trained multimodal model from checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory containing fusion_model.pt.
        device: Device to load model onto ('auto', 'cuda', 'cpu').

    Returns:
        (model, device) tuple.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    from ml.fusion.train_fusion_model import MultimodalModel

    model = MultimodalModel()
    ckpt_path = Path(checkpoint_dir) / "fusion_model.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}, using untrained model")

    model = model.to(device).eval()
    return model, device


def predict_text(model, text: str, device, max_length: int = 128) -> Dict[str, Any]:
    """Run text-only prediction.

    Args:
        model: Loaded MultimodalModel.
        text: Input text string.
        device: Torch device.
        max_length: Max token length.

    Returns:
        Dict with prediction, confidence, probabilities.
    """
    from ml.text_pipeline.preprocessing import preprocess_text
    from ml.text_pipeline.tokenizer import get_tokenizer

    cleaned = preprocess_text(text)
    tokenizer = get_tokenizer("distilbert")

    encoding = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    dummy_img = torch.zeros(1, 3, 224, 224).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, dummy_img)
        probs = torch.softmax(logits, dim=1).cpu().squeeze()

    predicted_class = probs.argmax().item()
    confidence = probs[predicted_class].item()

    return {
        "prediction": "Depressed" if predicted_class == 1 else "Non-Depressed",
        "confidence": round(confidence * 100, 1),
        "probabilities": {
            "depressed": round(probs[1].item() * 100, 1),
            "non_depressed": round(probs[0].item() * 100, 1),
        },
    }


def predict_multimodal(model, text: str, image_path: str, device) -> Dict[str, Any]:
    """Run multimodal prediction with text and image.

    Args:
        model: Loaded MultimodalModel.
        text: Input text string.
        image_path: Path to facial image.
        device: Torch device.

    Returns:
        Dict with prediction, confidence, probabilities.
    """
    from ml.text_pipeline.preprocessing import preprocess_text
    from ml.text_pipeline.tokenizer import get_tokenizer
    from ml.face_pipeline.preprocessing import get_inference_transform
    from PIL import Image

    # Text
    cleaned = preprocess_text(text)
    tokenizer = get_tokenizer("distilbert")
    encoding = tokenizer(
        cleaned, return_tensors="pt", max_length=128,
        padding="max_length", truncation=True,
    )

    # Image
    transform = get_inference_transform()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze()

    predicted_class = probs.argmax().item()

    return {
        "prediction": "Depressed" if predicted_class == 1 else "Non-Depressed",
        "confidence": round(probs[predicted_class].item() * 100, 1),
        "probabilities": {
            "depressed": round(probs[1].item() * 100, 1),
            "non_depressed": round(probs[0].item() * 100, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference on trained model")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--image", type=str, help="Path to facial image")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/fusion_model")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, device = load_model(args.checkpoint, args.device)

    if args.text and args.image:
        result = predict_multimodal(model, args.text, args.image, device)
    elif args.text:
        result = predict_text(model, args.text, device)
    else:
        print("Please provide --text and/or --image")
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
