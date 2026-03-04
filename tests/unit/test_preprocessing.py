"""Unit tests for preprocessing transforms."""

import torch
from PIL import Image
from torchvision.transforms import v2

from biometric.data.preprocessing import (
    GRAYSCALE_MEAN,
    GRAYSCALE_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_fingerprint_transform,
    get_iris_transform,
    get_multimodal_transforms,
)


class TestFingerprintTransform:
    """Tests for fingerprint preprocessing pipeline."""

    def test_output_shape(self) -> None:
        """Fingerprint transform produces (1, H, W) tensor from grayscale input."""
        transform = get_fingerprint_transform(size=(96, 96))
        img = Image.new("L", (96, 103), color=128)
        out = transform(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 96, 96)
        assert out.dtype == torch.float32

    def test_output_value_range(self) -> None:
        """Normalized output has reasonable value range (not clipped)."""
        transform = get_fingerprint_transform(size=(64, 64))
        img = Image.new("L", (96, 103), color=0)
        out = transform(img)
        assert out.min() >= -3.0 and out.max() <= 3.0

    def test_train_mode_includes_augmentation(self) -> None:
        """Train mode adds RandomHorizontalFlip."""
        transform = get_fingerprint_transform(size=(32, 32), train=True)
        assert isinstance(transform, v2.Compose)
        # Compose contains RandomHorizontalFlip when train=True
        has_flip = any(isinstance(t, v2.RandomHorizontalFlip) for t in transform.transforms)
        assert has_flip

    def test_inference_mode_no_augmentation(self) -> None:
        """Inference mode has no random transforms."""
        transform = get_fingerprint_transform(size=(32, 32), train=False)
        has_random = any("Random" in type(t).__name__ for t in transform.transforms)
        assert not has_random


class TestIrisTransform:
    """Tests for iris preprocessing pipeline."""

    def test_output_shape(self) -> None:
        """Iris transform produces (3, H, W) tensor."""
        transform = get_iris_transform(size=(224, 224))
        img = Image.new("RGB", (320, 240), color=(100, 150, 200))
        out = transform(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32

    def test_output_value_range(self) -> None:
        """Normalized output has reasonable value range."""
        transform = get_iris_transform(size=(64, 64))
        img = Image.new("RGB", (320, 240), color=(0, 0, 0))
        out = transform(img)
        assert out.min() >= -3.0 and out.max() <= 3.0

    def test_train_mode_includes_augmentation(self) -> None:
        """Train mode adds RandomHorizontalFlip."""
        transform = get_iris_transform(size=(64, 64), train=True)
        has_flip = any(isinstance(t, v2.RandomHorizontalFlip) for t in transform.transforms)
        assert has_flip


class TestMultimodalTransforms:
    """Tests for get_multimodal_transforms."""

    def test_returns_tuple_of_two(self) -> None:
        """Returns (fingerprint_transform, iris_transform)."""
        fp_t, iris_t = get_multimodal_transforms()
        assert fp_t is not None
        assert iris_t is not None

    def test_custom_sizes(self) -> None:
        """Custom sizes are applied correctly."""
        fp_t, iris_t = get_multimodal_transforms(
            fingerprint_size=(128, 128),
            iris_size=(112, 112),
        )
        img_fp = Image.new("L", (96, 103), color=0)
        img_iris = Image.new("RGB", (320, 240), color=(0, 0, 0))
        assert fp_t(img_fp).shape == (1, 128, 128)
        assert iris_t(img_iris).shape == (3, 112, 112)


class TestConstants:
    """Tests for normalization constants."""

    def test_imagenet_constants(self) -> None:
        """ImageNet mean/std have 3 values."""
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3

    def test_grayscale_constants(self) -> None:
        """Grayscale mean/std have 1 value."""
        assert len(GRAYSCALE_MEAN) == 1
        assert len(GRAYSCALE_STD) == 1
