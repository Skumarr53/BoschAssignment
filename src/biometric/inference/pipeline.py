"""End-to-end inference pipeline for multimodal biometric model.

[Phase 5a] Load checkpoint, run prediction on batches.
"""

from __future__ import annotations

from pathlib import Path

import torch

from biometric.models.fusion_model import MultimodalFusionModel
from biometric.utils.logging import get_logger

logger = get_logger(__name__)


def load_model(
    checkpoint_path: Path | str,
    num_classes: int,
    embedding_dim: int = 128,
    device: str | torch.device = "cpu",
) -> MultimodalFusionModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint (saved by CheckpointCallback).
        num_classes: Number of output classes (must match training).
        embedding_dim: Embedding dimension (must match training).
        device: Target device for inference.

    Returns:
        Loaded model in eval mode.
    """
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        msg = f"Checkpoint not found: {path}"
        raise FileNotFoundError(msg)

    ckpt = torch.load(path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = MultimodalFusionModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    logger.info(
        "model_loaded",
        path=str(path),
        num_classes=num_classes,
        device=str(device),
    )
    return model


def predict(
    model: MultimodalFusionModel,
    batch: dict[str, torch.Tensor],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Run inference on a batch. Returns logits (B, num_classes).

    Args:
        model: Loaded MultimodalFusionModel.
        batch: Dict with "fingerprint" (B,1,H,W) and "iris" (B,3,H,W) tensors.
        device: Device for tensors.

    Returns:
        Logits tensor (B, num_classes).
    """
    fp = batch["fingerprint"].to(device)
    iris = batch["iris"].to(device)

    with torch.no_grad():
        logits: torch.Tensor = model(fp, iris)

    return logits


class InferencePipeline:
    """End-to-end inference: load model + predict.

    [Phase 5a] Convenience wrapper for load + predict.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        num_classes: int,
        embedding_dim: int = 128,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize pipeline with checkpoint path and model config."""
        self._model = load_model(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            device=device,
        )
        self._device = device

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run prediction on batch. Returns logits."""
        return predict(self._model, batch, device=self._device)

    def predict_proba(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run prediction and return softmax probabilities."""
        logits = self.predict(batch)
        return torch.softmax(logits, dim=1)

    def predict_classes(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run prediction and return argmax class indices."""
        logits = self.predict(batch)
        return logits.argmax(dim=1)
