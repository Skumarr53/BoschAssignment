"""Reproducibility utilities for deterministic training.

Seeds Python, NumPy, and PyTorch RNGs and sets deterministic CuDNN/MIOpen flags.
Config snapshots and git hash logging are handled by Hydra/MLflow in Phase 3d/3e.
[AMD] Guards CUDA/cuDNN for CPU-only runs; ROCm uses same device as CUDA.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all relevant RNG seeds and deterministic flags for reproducibility.

    Covers: random, numpy, torch (CPU + CUDA/ROCm), cuDNN/MIOpen.
    Call once at training start (e.g. in scripts/train.py).

    Args:
        seed: Integer seed. Default 42 per configs/training/default.yaml.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
