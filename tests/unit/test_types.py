"""Unit tests for biometric.utils.types."""

from biometric.utils.types import (
    BiometricModality,
    FingerType,
    Gender,
    Hand,
)


def test_biometric_modality_enum() -> None:
    """BiometricModality has expected values."""
    assert BiometricModality.FINGERPRINT.value == "fingerprint"
    assert BiometricModality.IRIS_LEFT.value == "iris_left"
    assert BiometricModality.IRIS_RIGHT.value == "iris_right"


def test_gender_enum() -> None:
    """Gender has expected values."""
    assert Gender.M.value == "M"
    assert Gender.F.value == "F"


def test_hand_enum() -> None:
    """Hand has expected values."""
    assert Hand.LEFT.value == "Left"
    assert Hand.RIGHT.value == "Right"


def test_finger_type_enum() -> None:
    """FingerType has expected values."""
    assert FingerType.INDEX.value == "index_finger"
    assert FingerType.THUMB.value == "thumb_finger"
