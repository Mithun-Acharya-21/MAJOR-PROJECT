"""
Image Preprocessing — Prepare facial images for the CNN model.

Applies standard transformations:
  - Resize to 224×224
  - Normalize with ImageNet statistics
  - Data augmentation for training

Exports:
  get_inference_transform — standard preprocessing for inference
  get_training_transform — augmentation pipeline for training
  preprocess_image_bytes — bytes → tensor for model input
"""

from __future__ import annotations

import io
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform(target_size: int = 224) -> transforms.Compose:
    """Standard preprocessing for inference (no augmentation).

    Args:
        target_size: Output image size.

    Returns:
        torchvision transforms pipeline.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform(
    target_size: int = 224,
    horizontal_flip_prob: float = 0.5,
    rotation_degrees: int = 10,
    color_jitter: bool = True,
) -> transforms.Compose:
    """Data augmentation pipeline for training.

    Includes:
      - Random horizontal flip
      - Random rotation
      - Random affine (translation + scale)
      - Optional color jitter (brightness, contrast, saturation)
      - Resize + normalize

    Args:
        target_size: Output image size.
        horizontal_flip_prob: Probability of horizontal flip.
        rotation_degrees: Max rotation in degrees.
        color_jitter: Whether to apply color augmentation.

    Returns:
        torchvision transforms pipeline.
    """
    transform_list = [
        transforms.Resize((target_size + 16, target_size + 16)),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
    ]

    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        )

    transform_list.extend([
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


def preprocess_image_bytes(
    image_bytes: bytes,
    target_size: int = 224,
    augment: bool = False,
) -> torch.Tensor:
    """Preprocess raw image bytes into a model-ready tensor.

    Args:
        image_bytes: Raw image file bytes (JPEG, PNG, etc.).
        target_size: Output spatial size.
        augment: If True, apply training augmentation.

    Returns:
        Tensor of shape (1, 3, target_size, target_size).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = get_training_transform(target_size) if augment else get_inference_transform(target_size)
    tensor = transform(img)

    return tensor.unsqueeze(0)  # Add batch dimension
