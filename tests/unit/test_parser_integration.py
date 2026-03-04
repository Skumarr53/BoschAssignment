"""Integration tests for parser against real dataset files."""

from pathlib import Path

import pytest

from biometric.utils.parser import parse_fingerprint_filename, parse_iris_path
from biometric.utils.types import BiometricModality, Gender, Hand


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "Data" / "1" / "Fingerprint").exists(),
    reason="Dataset not present",
)
class TestParserWithRealData:
    """Parser tests using actual dataset files."""

    def test_parse_real_fingerprint(self, data_root: Path) -> None:
        """Parse a real fingerprint file from the dataset."""
        path = data_root / "1" / "Fingerprint" / "1__M_Left_index_finger.BMP"
        if not path.exists():
            pytest.skip("Fingerprint file not found")
        sample = parse_fingerprint_filename(path, label=0)
        assert sample.subject_id == 1
        assert sample.gender == Gender.M
        assert sample.hand == Hand.LEFT
        assert sample.file_path.exists()

    def test_parse_real_iris_left(self, data_root: Path) -> None:
        """Parse a real left iris file from the dataset."""
        left_dir = data_root / "1" / "left"
        if not left_dir.exists():
            pytest.skip("Iris left dir not found")
        bmp_files = list(left_dir.glob("*.bmp"))
        if not bmp_files:
            pytest.skip("No iris bmp files found")
        path = bmp_files[0]
        sample = parse_iris_path(path, subject_id=1, is_left=True, label=0)
        assert sample.subject_id == 1
        assert sample.modality == BiometricModality.IRIS_LEFT
        assert sample.file_path.exists()
