"""
Seed — Set random seeds for reproducibility across all libraries.

Exports:
  set_seed — set seeds for random, numpy, and torch
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
      - Python random
      - NumPy
      - PyTorch (CPU and CUDA)
      - CUBLAS deterministic (if CUDA available)

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
