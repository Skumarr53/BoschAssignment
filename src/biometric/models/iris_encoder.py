"""Iris encoder: 224×224 RGB → embedding.

Iris branch encoder for multimodal fusion.
Architecture: Conv blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d) → Flatten → Dense.
Larger input than fingerprint → more conv stages.
"""

import torch
from torch import nn

from biometric.models.base import EncoderBase, conv_block


class IrisEncoder(EncoderBase):
    """CNN encoder for iris images (3×224×224 RGB).

    Produces a fixed-size embedding for late fusion.
    Architecture mirrors typical multimodal biometric reference implementations.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        input_channels: int = 3,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim)
        self.features = nn.Sequential(
            conv_block(input_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        # 224 → 112 → 56 → 28 → 14
        self._flat_size = 256 * 14 * 14
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode iris. Input (B, 3, 224, 224) → output (B, embedding_dim)."""
        h = self.features(x)
        out: torch.Tensor = self.fc(h)
        return out
