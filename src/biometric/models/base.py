"""Abstract base for modality-specific encoders.

All encoders produce a fixed-size embedding vector for fusion.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
    """Single conv block: Conv → BatchNorm → ReLU → MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


@runtime_checkable
class BaseEncoder(Protocol):
    """Protocol for modality encoders: input tensor → embedding vector."""

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding. Shape: (B, C, H, W) → (B, embedding_dim)."""
        ...


class EncoderBase(nn.Module, ABC):
    """Abstract base class for concrete encoder implementations."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension."""
        return self._embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding. Shape: (B, C, H, W) → (B, embedding_dim)."""
        ...
