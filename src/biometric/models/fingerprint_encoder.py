"""Fingerprint encoder: 96×96 grayscale → embedding.

Fingerprint branch encoder for multimodal fusion.
Architecture: Conv blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d) → Flatten → Dense.
"""

import torch
from torch import nn

from biometric.models.base import EncoderBase, conv_block


class FingerprintEncoder(EncoderBase):
    """CNN encoder for fingerprint images (1×96×96 grayscale).

    Produces a fixed-size embedding for late fusion.
    Architecture mirrors typical multimodal biometric reference implementations.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        input_channels: int = 1,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim)
        self.features = nn.Sequential(
            conv_block(input_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        # 96 → 48 → 24 → 12
        self._flat_size = 128 * 12 * 12
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode fingerprint. Input (B, 1, 96, 96) → output (B, embedding_dim)."""
        h = self.features(x)
        out: torch.Tensor = self.fc(h)
        return out
