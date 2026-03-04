"""Unit tests for biometric.data.arrow_cache."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from biometric.data.arrow_cache import (
    CACHE_METADATA_KEY,
    build_cache,
    cache_exists,
    filter_by_modality,
    filter_by_subjects,
    get_cache_path,
    is_cache_stale,
    load_cache,
)


class TestBuildCache:
    def test_build_returns_table(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        assert isinstance(table, pa.Table)
        assert table.num_rows > 0

    def test_build_saves_parquet(self, synthetic_data: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "test_cache.parquet"
            table = build_cache(synthetic_data, cache_path=cache_path)
            assert cache_path.exists()
            loaded = load_cache(cache_path)
            assert loaded.num_rows == table.num_rows

    def test_build_empty_root_returns_empty_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = build_cache(root, cache_path=None)
            assert table.num_rows == 0

    def test_null_values_for_iris_rows(self, synthetic_data: Path) -> None:
        """Iris rows have null gender/hand/finger_type; fingerprint has null sequence."""
        table = build_cache(synthetic_data, cache_path=None)
        iris_rows = filter_by_modality(table, "iris_left")
        if iris_rows.num_rows > 0:
            gender_col = iris_rows.column("gender")
            assert gender_col.null_count == iris_rows.num_rows

        fp_rows = filter_by_modality(table, "fingerprint")
        if fp_rows.num_rows > 0:
            seq_col = fp_rows.column("sequence")
            assert seq_col.null_count == fp_rows.num_rows

    def test_source_fingerprint_in_metadata(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        metadata = table.schema.metadata or {}
        assert CACHE_METADATA_KEY in metadata
        assert len(metadata[CACHE_METADATA_KEY]) > 0


class TestLoadCache:
    def test_load_returns_table(self, synthetic_data: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.parquet"
            build_cache(synthetic_data, cache_path=cache_path)
            table = load_cache(cache_path)
            assert isinstance(table, pa.Table)
            assert table.num_rows > 0

    def test_load_nonexistent_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "nonexistent.parquet"
            with pytest.raises((FileNotFoundError, OSError)):
                load_cache(bad_path)


class TestCacheStaleness:
    def test_fresh_cache_not_stale(self, synthetic_data: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.parquet"
            build_cache(synthetic_data, cache_path=cache_path)
            assert is_cache_stale(cache_path, synthetic_data) is False

    def test_missing_cache_is_stale(self, synthetic_data: Path) -> None:
        assert is_cache_stale("/nonexistent/cache.parquet", synthetic_data) is True

    def test_stale_after_file_added(self, synthetic_data: Path) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.parquet"
            build_cache(synthetic_data, cache_path=cache_path)
            new_img = Image.new("RGB", (320, 240))
            new_img.save(synthetic_data / "1" / "left" / "newl1.bmp")
            assert is_cache_stale(cache_path, synthetic_data) is True


class TestFilterBySubjects:
    def test_filter_reduces_rows(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        filtered = filter_by_subjects(table, [1, 2])
        assert filtered.num_rows < table.num_rows
        subject_ids = {int(x) for x in filtered.column("subject_id")}
        assert subject_ids <= {1, 2}

    def test_filter_empty_list_returns_all(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        filtered = filter_by_subjects(table, [])
        assert filtered.num_rows == table.num_rows


class TestFilterByModality:
    def test_filter_fingerprint(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        filtered = filter_by_modality(table, "fingerprint")
        assert filtered.num_rows == 30  # 3 subjects * 10
        modalities = {str(x) for x in filtered.column("modality")}
        assert modalities == {"fingerprint"}

    def test_filter_iris_left(self, synthetic_data: Path) -> None:
        table = build_cache(synthetic_data, cache_path=None)
        filtered = filter_by_modality(table, "iris_left")
        assert filtered.num_rows == 15  # 3 subjects * 5
        modalities = {str(x) for x in filtered.column("modality")}
        assert modalities == {"iris_left"}


class TestCacheExists:
    def test_exists_true_when_file_present(self, synthetic_data: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.parquet"
            build_cache(synthetic_data, cache_path=cache_path)
            assert cache_exists(cache_path) is True

    def test_exists_false_when_absent(self) -> None:
        assert cache_exists("/nonexistent/cache.parquet") is False


class TestGetCachePath:
    def test_returns_root_plus_filename(self) -> None:
        path = get_cache_path("/data/root", "custom.parquet")
        assert path == Path("/data/root/custom.parquet")

    def test_default_filename(self) -> None:
        from biometric.data.arrow_cache import DEFAULT_CACHE_FILENAME

        path = get_cache_path("/data")
        assert path.name == DEFAULT_CACHE_FILENAME
