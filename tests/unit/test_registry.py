"""Unit tests for biometric.data.registry."""

from pathlib import Path

import pytest

from biometric.data.registry import (
    get_dataset,
    get_transform,
    list_datasets,
    list_transforms,
    register_dataset,
    register_transform,
)


class TestDatasetRegistry:
    def test_list_datasets(self) -> None:
        names = list_datasets()
        assert "fingerprint" in names
        assert "iris" in names
        assert "multimodal" in names

    def test_get_dataset_fingerprint(self, synthetic_data: Path) -> None:
        ds = get_dataset("fingerprint", root=synthetic_data)
        assert len(ds) > 0

    def test_get_dataset_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown dataset"):
            get_dataset("unknown", root=Path("."))

    def test_register_custom_dataset(self) -> None:
        @register_dataset("test_custom")
        class _CustomDS:
            pass

        names = list_datasets()
        assert "test_custom" in names


class TestTransformRegistry:
    def test_list_transforms(self) -> None:
        names = list_transforms()
        assert "fingerprint" in names
        assert "iris" in names

    def test_get_transform_fingerprint(self) -> None:
        t = get_transform("fingerprint", size=(96, 96))
        assert t is not None

    def test_get_transform_iris(self) -> None:
        t = get_transform("iris", size=(224, 224))
        assert t is not None

    def test_get_transform_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown modality"):
            get_transform("unknown")

    def test_register_custom_transform(self) -> None:
        @register_transform("test_modality")
        def _custom_transform() -> str:
            return "custom"

        names = list_transforms()
        assert "test_modality" in names
