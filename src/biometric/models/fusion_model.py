"""Multimodal fusion model: fingerprint + iris → logits.

Late fusion: concatenate modality embeddings → classifier.
"""

import torch
from torch import nn

from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.iris_encoder import IrisEncoder


class MultimodalFusionModel(nn.Module):
    """Late-fusion model: fingerprint and iris encoders → concatenated embeddings → classifier.

    Accepts batch dict from MultimodalBiometricDataset:
        {"fingerprint": (B, 1, 96, 96), "iris": (B, 3, 224, 224), "label": (B,)}
    """

    def __init__(
        self,
        num_classes: int = 45,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._embedding_dim = embedding_dim
        self.fingerprint_encoder = FingerprintEncoder(embedding_dim=embedding_dim)
        self.iris_encoder = IrisEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(2 * embedding_dim, num_classes)

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    def forward(
        self,
        fingerprint: torch.Tensor,
        iris: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns logits (B, num_classes)."""
        fp_emb = self.fingerprint_encoder(fingerprint)
        iris_emb = self.iris_encoder(iris)
        fused = torch.cat([fp_emb, iris_emb], dim=1)
        logits: torch.Tensor = self.classifier(fused)
        return logits
