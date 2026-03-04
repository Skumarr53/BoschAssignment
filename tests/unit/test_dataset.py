"""Unit tests for biometric.data.dataset."""

from pathlib import Path

import torch

from biometric.data.dataset import (
    FingerprintDataset,
    IrisDataset,
    MultimodalBiometricDataset,
    _load_image,
)


class TestLoadImage:
    """Tests for _load_image helper."""

    def test_load_grayscale(self, synthetic_data: Path) -> None:
        fp_dir = synthetic_data / "1" / "Fingerprint"
        bmp = next(fp_dir.glob("*.BMP"))
        img = _load_image(bmp, mode="L")
        assert img.mode == "L"

    def test_load_rgb(self, synthetic_data: Path) -> None:
        iris_dir = synthetic_data / "1" / "left"
        bmp = next(iris_dir.glob("*.bmp"))
        img = _load_image(bmp, mode="RGB")
        assert img.mode == "RGB"


class TestFingerprintDataset:
    """Tests for FingerprintDataset using synthetic data."""

    def test_len(self, synthetic_data: Path) -> None:
        ds = FingerprintDataset(synthetic_data)
        assert len(ds) == 30  # 3 subjects * 10 fingerprints

    def test_getitem_returns_grayscale_pil(self, synthetic_data: Path) -> None:
        ds = FingerprintDataset(synthetic_data)
        img, label = ds[0]
        assert img.mode == "L"
        assert label >= 0

    def test_subject_ids_filter(self, synthetic_data: Path) -> None:
        ds = FingerprintDataset(synthetic_data, subject_ids=[1, 2])
        assert len(ds) == 20

    def test_transform_applied(self, synthetic_data: Path) -> None:
        def to_list(x: object) -> list[object]:
            return [x]

        ds = FingerprintDataset(synthetic_data, transform=to_list)
        result, _ = ds[0]
        assert isinstance(result, list)

    def test_preprocessing_produces_tensor(self, synthetic_data: Path) -> None:
        from biometric.data.preprocessing import get_fingerprint_transform

        transform = get_fingerprint_transform(size=(64, 64))
        ds = FingerprintDataset(synthetic_data, transform=transform)
        img, label = ds[0]
        assert img.shape == (1, 64, 64)
        assert img.dtype == torch.float32
        assert label >= 0

    def test_public_samples_property(self, synthetic_data: Path) -> None:
        ds = FingerprintDataset(synthetic_data)
        samples = ds.samples
        assert len(samples) == len(ds)
        assert samples is not ds._samples  # defensive copy


class TestIrisDataset:
    """Tests for IrisDataset using synthetic data."""

    def test_len(self, synthetic_data: Path) -> None:
        ds = IrisDataset(synthetic_data)
        assert len(ds) == 30  # 3 subjects * 10 iris

    def test_getitem_returns_rgb_pil(self, synthetic_data: Path) -> None:
        ds = IrisDataset(synthetic_data)
        img, label = ds[0]
        assert img.mode == "RGB"
        assert label >= 0

    def test_subject_ids_filter(self, synthetic_data: Path) -> None:
        ds = IrisDataset(synthetic_data, subject_ids=[1])
        assert len(ds) == 10

    def test_preprocessing_produces_tensor(self, synthetic_data: Path) -> None:
        from biometric.data.preprocessing import get_iris_transform

        transform = get_iris_transform(size=(64, 64))
        ds = IrisDataset(synthetic_data, transform=transform)
        img, label = ds[0]
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.float32

    def test_public_samples_property(self, synthetic_data: Path) -> None:
        ds = IrisDataset(synthetic_data)
        samples = ds.samples
        assert len(samples) == len(ds)


class TestMultimodalBiometricDataset:
    """Tests for MultimodalBiometricDataset."""

    def test_len(self, synthetic_data: Path) -> None:
        ds = MultimodalBiometricDataset(synthetic_data)
        assert len(ds) == 30  # 3 subjects * min(10fp, 10iris)

    def test_getitem_returns_dict(self, synthetic_data: Path) -> None:
        ds = MultimodalBiometricDataset(synthetic_data)
        sample = ds[0]
        assert "fingerprint" in sample
        assert "iris" in sample
        assert "label" in sample
        assert sample["label"] >= 0

    def test_subject_ids_filter(self, synthetic_data: Path) -> None:
        ds = MultimodalBiometricDataset(synthetic_data, subject_ids=[1, 2])
        assert len(ds) == 20

    def test_pairing_is_semantic(self, synthetic_data: Path) -> None:
        """Verify pairs come from the same subject."""
        ds = MultimodalBiometricDataset(synthetic_data)
        for fp_idx, iris_idx in ds._pairs:
            fp_sid = ds._fp_dataset._samples[fp_idx].subject_id
            iris_sid = ds._iris_dataset._samples[iris_idx].subject_id
            assert fp_sid == iris_sid

    def test_transforms_applied(self, synthetic_data: Path) -> None:
        from biometric.data.preprocessing import get_multimodal_transforms

        fp_t, iris_t = get_multimodal_transforms(fingerprint_size=(32, 32), iris_size=(32, 32))
        ds = MultimodalBiometricDataset(
            synthetic_data,
            fingerprint_transform=fp_t,
            iris_transform=iris_t,
        )
        sample = ds[0]
        assert isinstance(sample["fingerprint"], torch.Tensor)
        assert isinstance(sample["iris"], torch.Tensor)
        assert sample["fingerprint"].shape == (1, 32, 32)
        assert sample["iris"].shape == (3, 32, 32)
