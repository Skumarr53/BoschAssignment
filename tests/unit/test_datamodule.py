"""Unit tests for biometric.data.datamodule."""

import os
from pathlib import Path

import pytest

from biometric.data.datamodule import BiometricDataModule, _split_subject_ids


class TestSplitSubjectIds:
    def test_deterministic(self) -> None:
        ids = list(range(1, 46))
        a1, b1, c1 = _split_subject_ids(ids, seed=42)
        a2, b2, c2 = _split_subject_ids(ids, seed=42)
        assert a1 == a2 and b1 == b2 and c1 == c2

    def test_partition_covers_all(self) -> None:
        ids = list(range(1, 46))
        train, val, test = _split_subject_ids(ids, seed=42)
        combined = set(train) | set(val) | set(test)
        assert combined == set(ids)

    def test_ratios_approximate(self) -> None:
        ids = list(range(1, 101))
        train, val, test = _split_subject_ids(ids, train_ratio=0.8, val_ratio=0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


class TestBiometricDataModule:
    def test_setup_builds_datasets(self, synthetic_data: Path) -> None:
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=4,
            num_workers=0,
            use_cache=False,
        )
        dm.setup(stage="fit")
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None

    def test_train_dataloader_returns_batches(self, synthetic_data: Path) -> None:
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=4,
            num_workers=0,
            use_cache=False,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "fingerprint" in batch
        assert "iris" in batch
        assert "label" in batch

    def test_train_dataloader_override_params(self, synthetic_data: Path) -> None:
        """[Phase 4] train_dataloader accepts batch_size/num_workers overrides for benchmarking."""
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=4,
            num_workers=0,
            use_cache=False,
        )
        dm.setup(stage="fit")
        loader = dm.train_dataloader(batch_size=8, num_workers=0)
        batch = next(iter(loader))
        assert batch["label"].shape[0] == 8

    def test_val_dataloader(self, synthetic_data: Path) -> None:
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=4,
            num_workers=0,
            use_cache=False,
        )
        loader = dm.val_dataloader()
        assert loader is not None

    def test_test_dataloader(self, synthetic_data: Path) -> None:
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=4,
            num_workers=0,
            use_cache=False,
        )
        loader = dm.test_dataloader()
        assert loader is not None

    def test_empty_root_raises(self, tmp_path: Path) -> None:
        dm = BiometricDataModule(tmp_path, use_cache=False)
        with pytest.raises(FileNotFoundError, match="No subjects"):
            dm.setup()

    def test_cache_integration(self, synthetic_data: Path) -> None:
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=4,
            num_workers=0,
            use_cache=True,
            cache_filename="test_cache.parquet",
        )
        dm.setup()
        assert dm._train_dataset is not None

    def test_runtime_error_not_assert(self, synthetic_data: Path) -> None:
        """Ensure RuntimeError raised (not AssertionError) if dataset is None."""
        dm = BiometricDataModule(synthetic_data, use_cache=False)
        dm._train_dataset = None
        dm._val_dataset = None
        dm._test_dataset = None
        dm.setup = lambda stage=None: None  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            dm.train_dataloader()

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="ProcessPoolExecutor hangs in GitHub Actions",
    )
    def test_parallel_preprocess_integration(self, synthetic_data: Path) -> None:
        """[Phase 1d] DataModule uses parallel_loader when use_parallel_preprocess=True."""
        dm = BiometricDataModule(
            synthetic_data,
            batch_size=2,
            num_workers=0,
            use_cache=False,
            use_parallel_preprocess=True,
            parallel_config={"backend": "multiprocessing", "max_workers": 2},
        )
        dm.setup(stage="fit")
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        batch = next(iter(dm.train_dataloader()))
        assert "fingerprint" in batch
        assert "iris" in batch
        assert "label" in batch
