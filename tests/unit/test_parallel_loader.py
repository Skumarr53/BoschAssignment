"""Unit tests for biometric.data.parallel_loader."""

import os
from pathlib import Path

import pytest
import torch

from biometric.data.arrow_cache import build_cache, filter_by_modality
from biometric.data.parallel_loader import (
    _build_transform_from_config,
    _load_and_preprocess,
    get_preprocessed_items_from_cache,
    preprocess_from_config,
    preprocess_with_backend,
    preprocess_with_pool,
    preprocess_with_ray,
)


class TestBuildTransformFromConfig:
    def test_fingerprint_transform(self) -> None:
        t = _build_transform_from_config("fingerprint", {"size": (64, 64), "train": False})
        assert t is not None

    def test_iris_transform(self) -> None:
        t = _build_transform_from_config("iris", {"size": (128, 128), "train": True})
        assert t is not None

    def test_default_sizes(self) -> None:
        t_fp = _build_transform_from_config("fingerprint", {})
        t_iris = _build_transform_from_config("iris", {})
        assert t_fp is not None
        assert t_iris is not None


class TestLoadAndPreprocess:
    def test_returns_numpy_and_label(self, synthetic_data: Path) -> None:
        fp_dir = synthetic_data / "1" / "Fingerprint"
        bmp = str(next(fp_dir.glob("*.BMP")))
        arr, label = _load_and_preprocess(bmp, 42, "fingerprint", {"size": (32, 32)})
        assert arr.shape[0] == 1  # grayscale channel
        assert label == 42

    def test_iris_preprocess(self, synthetic_data: Path) -> None:
        iris_dir = synthetic_data / "1" / "left"
        bmp = str(next(iris_dir.glob("*.bmp")))
        arr, label = _load_and_preprocess(bmp, 7, "iris", {"size": (32, 32)})
        assert arr.shape[0] == 3  # RGB
        assert label == 7

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_and_preprocess("/nonexistent.bmp", 0, "fingerprint", {})


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="ProcessPoolExecutor hangs in GitHub Actions (containerized env)",
)
class TestPreprocessWithPool:
    def test_returns_tensors_and_labels(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:3]
        labels = [int(x) for x in fp_table.column("label")][:3]
        results = preprocess_with_pool(paths, labels, "fingerprint", max_workers=2)
        assert len(results) == len(paths)
        for tensor, _label in results:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.dim() == 3

    def test_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            preprocess_with_pool(paths=["/a", "/b"], labels=[0], modality="fingerprint")

    def test_empty_input_returns_empty(self) -> None:
        results = preprocess_with_pool(paths=[], labels=[], modality="fingerprint")
        assert results == []

    def test_all_items_processed(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:6]
        labels = [int(x) for x in fp_table.column("label")][:6]
        results = preprocess_with_pool(paths, labels, "fingerprint", max_workers=2)
        assert len(results) == 6

    def test_custom_transform_config(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:2]
        labels = [int(x) for x in fp_table.column("label")][:2]
        config = {"size": (48, 48), "train": True}
        results = preprocess_with_pool(paths, labels, "fingerprint", transform_config=config)
        assert len(results) == 2
        assert results[0][0].shape[-1] == 48


class TestGetPreprocessedItemsFromCache:
    def test_extracts_paths_and_labels(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        paths, labels = get_preprocessed_items_from_cache(table, "fingerprint")
        assert len(paths) == 30  # 3 subjects * 10
        assert len(paths) == len(labels)
        assert all(isinstance(p, str) for p in paths)
        assert all(isinstance(lab, int) for lab in labels)

    def test_filter_by_subjects(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        paths, labels = get_preprocessed_items_from_cache(table, "fingerprint", subject_ids=[1, 2])
        assert len(paths) == 20

    def test_iris_modality(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        paths, labels = get_preprocessed_items_from_cache(table, "iris_left")
        assert len(paths) == 15  # 3 subjects * 5

    def test_empty_cache(self) -> None:
        import pyarrow

        from biometric.data.arrow_cache import CACHE_SCHEMA

        empty = pyarrow.table({c: [] for c in CACHE_SCHEMA.names}, schema=CACHE_SCHEMA)
        paths, labels = get_preprocessed_items_from_cache(empty, "fingerprint")
        assert paths == []
        assert labels == []


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="ProcessPoolExecutor/Ray hangs in GitHub Actions",
)
class TestPreprocessWithBackend:
    def test_backend_multiprocessing_dispatches_to_pool(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:3]
        labels = [int(x) for x in fp_table.column("label")][:3]
        results = preprocess_with_backend(
            paths, labels, "fingerprint", backend="multiprocessing", max_workers=2
        )
        assert len(results) == len(paths)
        for tensor, _ in results:
            assert isinstance(tensor, torch.Tensor)

    def test_backend_pool_alias_works(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:2]
        labels = [int(x) for x in fp_table.column("label")][:2]
        results = preprocess_with_backend(
            paths, labels, "fingerprint", backend="pool", max_workers=2
        )
        assert len(results) == 2

    def test_backend_ray_dispatches_when_installed(self, synthetic_data: Path) -> None:
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.skip("Ray not installed")
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:2]
        labels = [int(x) for x in fp_table.column("label")][:2]
        if not paths:
            pytest.skip("No fingerprint paths")
        results = preprocess_with_backend(
            paths, labels, "fingerprint", backend="ray", num_cpus_per_task=0.5
        )
        assert len(results) == len(paths)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown parallel backend"):
            preprocess_with_backend(
                paths=["/a"], labels=[0], modality="fingerprint", backend="invalid"
            )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="ProcessPoolExecutor/Ray hangs in GitHub Actions",
)
class TestPreprocessFromConfig:
    def test_reads_backend_from_config(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:2]
        labels = [int(x) for x in fp_table.column("label")][:2]
        config = {"backend": "multiprocessing", "max_workers": 2}
        results = preprocess_from_config(paths, labels, "fingerprint", config)
        assert len(results) == 2

    def test_default_backend_ray_when_omitted(self, synthetic_data: Path) -> None:
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.skip("Ray not installed")
        table = build_cache(synthetic_data, cache_path=None)
        fp_table = filter_by_modality(table, "fingerprint")
        paths = [str(x) for x in fp_table.column("filepath")][:2]
        labels = [int(x) for x in fp_table.column("label")][:2]
        if not paths:
            pytest.skip("No fingerprint paths")
        config = {}  # backend defaults to ray
        results = preprocess_from_config(paths, labels, "fingerprint", config)
        assert len(results) == len(paths)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Ray init/preprocess hangs in GitHub Actions",
)
class TestPreprocessWithRay:
    def test_ray_import_or_skip(self) -> None:
        try:
            import ray  # noqa: F401

            pytest.skip("Ray is installed; skipping ImportError test")
        except ImportError:
            with pytest.raises(ImportError, match="Ray Data not installed"):
                preprocess_with_ray(
                    paths=["/nonexistent"],
                    labels=[0],
                    modality="fingerprint",
                )

    def test_ray_preprocess_when_installed(self, synthetic_data: Path) -> None:
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.skip("Ray not installed")
        table = build_cache(synthetic_data, cache_path=None)
        paths, labels = get_preprocessed_items_from_cache(table, "fingerprint")
        paths, labels = paths[:2], labels[:2]
        if not paths:
            pytest.skip("No fingerprint paths")
        results = preprocess_with_ray(paths, labels, "fingerprint", num_cpus_per_task=0.5)
        assert len(results) == len(paths)
        for tensor, label in results:
            assert isinstance(tensor, torch.Tensor)
            assert label in labels
