"""Unit tests for model layer."""

from pathlib import Path

import torch

from biometric.data.datamodule import BiometricDataModule
from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion_model import MultimodalFusionModel
from biometric.models.iris_encoder import IrisEncoder


class TestFingerprintEncoder:
    """Tests for FingerprintEncoder."""

    def test_forward_shape(self) -> None:
        """Forward pass produces (B, embedding_dim)."""
        model = FingerprintEncoder(embedding_dim=128)
        x = torch.randn(4, 1, 96, 96)
        out = model(x)
        assert out.shape == (4, 128)

    def test_embedding_dim_property(self) -> None:
        """embedding_dim property returns configured value."""
        model = FingerprintEncoder(embedding_dim=64)
        assert model.embedding_dim == 64


class TestIrisEncoder:
    """Tests for IrisEncoder."""

    def test_forward_shape(self) -> None:
        """Forward pass produces (B, embedding_dim)."""
        model = IrisEncoder(embedding_dim=128)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 128)

    def test_embedding_dim_property(self) -> None:
        """embedding_dim property returns configured value."""
        model = IrisEncoder(embedding_dim=64)
        assert model.embedding_dim == 64


class TestMultimodalFusionModel:
    """Tests for MultimodalFusionModel."""

    def test_forward_shape(self) -> None:
        """Forward pass produces (B, num_classes) logits."""
        model = MultimodalFusionModel(num_classes=45)
        fp = torch.randn(4, 1, 96, 96)
        iris = torch.randn(4, 3, 224, 224)
        logits = model(fingerprint=fp, iris=iris)
        assert logits.shape == (4, 45)

    def test_batch_dict_format(self) -> None:
        """Model accepts batch dict format from MultimodalBiometricDataset."""
        model = MultimodalFusionModel(num_classes=10)
        batch = {
            "fingerprint": torch.randn(2, 1, 96, 96),
            "iris": torch.randn(2, 3, 224, 224),
            "label": torch.tensor([0, 1]),
        }
        logits = model(fingerprint=batch["fingerprint"], iris=batch["iris"])
        assert logits.shape == (2, 10)

    def test_forward_with_datamodule_batch(self, synthetic_data: Path) -> None:
        """End-to-end: batch from DataModule → logits (B, num_classes)."""
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=4,
            num_workers=0,
            fingerprint_size=(96, 96),
            iris_size=(224, 224),
            use_cache=True,
        )
        dm.setup(stage="fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))

        num_classes = len({int(b) for b in batch["label"].tolist()})
        model = MultimodalFusionModel(num_classes=max(num_classes, 2))
        logits = model(fingerprint=batch["fingerprint"], iris=batch["iris"])
        assert logits.shape == (batch["fingerprint"].shape[0], model.num_classes)
