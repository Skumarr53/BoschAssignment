"""Unit tests for biometric.utils.discovery."""

from pathlib import Path

import pytest

from biometric.utils.discovery import discover_subjects, validate_path


class TestDiscoverSubjects:
    def test_finds_all_subjects(self, synthetic_data: Path) -> None:
        ids = discover_subjects(synthetic_data)
        assert ids == [1, 2, 3]

    def test_empty_directory(self, tmp_path: Path) -> None:
        ids = discover_subjects(tmp_path)
        assert ids == []

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        ids = discover_subjects(tmp_path / "nonexistent")
        assert ids == []

    def test_ignores_non_numeric_dirs(self, synthetic_data: Path) -> None:
        (synthetic_data / "not_a_number").mkdir()
        ids = discover_subjects(synthetic_data)
        assert ids == [1, 2, 3]

    def test_partial_data(self, tmp_path: Path) -> None:
        """Subject with only iris but no fingerprint data is still discovered."""
        sid_dir = tmp_path / "99" / "left"
        sid_dir.mkdir(parents=True)
        from PIL import Image

        Image.new("RGB", (10, 10)).save(sid_dir / "testl1.bmp")
        ids = discover_subjects(tmp_path)
        assert 99 in ids


class TestValidatePath:
    def test_valid_path(self, tmp_path: Path) -> None:
        child = tmp_path / "a" / "b.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = validate_path(child, tmp_path)
        assert result == child.resolve()

    def test_traversal_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="outside allowed root"):
            validate_path(outside, tmp_path / "subdir")

    def test_symlink_resolution(self, tmp_path: Path) -> None:
        real = tmp_path / "real.txt"
        real.touch()
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        result = validate_path(link, tmp_path)
        assert result == real.resolve()
