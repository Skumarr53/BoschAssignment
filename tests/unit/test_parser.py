"""Unit tests for biometric.utils.parser."""

from pathlib import Path

import pytest

from biometric.utils.parser import parse_fingerprint_filename, parse_iris_path
from biometric.utils.types import (
    BiometricModality,
    FingerType,
    Gender,
    Hand,
)


class TestParseFingerprintFilename:
    """Tests for parse_fingerprint_filename."""

    def test_valid_male_left_index(self) -> None:
        """Parse male left index fingerprint."""
        path = Path("/data/1/Fingerprint/1__M_Left_index_finger.BMP")
        sample = parse_fingerprint_filename(path, label=0)
        assert sample.subject_id == 1
        assert sample.gender == Gender.M
        assert sample.hand == Hand.LEFT
        assert sample.finger_type == FingerType.INDEX
        assert sample.modality == BiometricModality.FINGERPRINT
        assert sample.label == 0

    def test_valid_female_right_thumb(self) -> None:
        """Parse female right thumb fingerprint."""
        path = Path("/data/42/Fingerprint/42__F_Right_thumb_finger.BMP")
        sample = parse_fingerprint_filename(path, label=41)
        assert sample.subject_id == 42
        assert sample.gender == Gender.F
        assert sample.hand == Hand.RIGHT
        assert sample.finger_type == FingerType.THUMB
        assert sample.label == 41

    def test_invalid_format_raises(self) -> None:
        """Invalid filename raises ValueError."""
        path = Path("/data/bad_filename.jpg")
        with pytest.raises(ValueError, match="Invalid fingerprint filename format"):
            parse_fingerprint_filename(path, label=0)


class TestParseIrisPath:
    """Tests for parse_iris_path."""

    def test_valid_left_eye(self) -> None:
        """Parse left iris image."""
        path = Path("/data/1/left/aeval1.bmp")
        sample = parse_iris_path(path, subject_id=1, is_left=True, label=0)
        assert sample.subject_id == 1
        assert sample.modality == BiometricModality.IRIS_LEFT
        assert sample.sequence == 1
        assert sample.label == 0

    def test_valid_right_eye_sequence_5(self) -> None:
        """Parse right iris image with sequence 5."""
        path = Path("/data/42/right/winr5.bmp")
        sample = parse_iris_path(path, subject_id=42, is_left=False, label=41)
        assert sample.subject_id == 42
        assert sample.modality == BiometricModality.IRIS_RIGHT
        assert sample.sequence == 5
        assert sample.label == 41

    def test_invalid_format_raises(self) -> None:
        """Invalid filename raises ValueError."""
        path = Path("/data/invalid.bmp")
        with pytest.raises(ValueError, match="Invalid iris filename format"):
            parse_iris_path(path, subject_id=1, is_left=True, label=0)
