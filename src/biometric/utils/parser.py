"""Filename and path parsers for biometric dataset."""

import re
from pathlib import Path

from biometric.utils.types import (
    BiometricModality,
    FingerprintSample,
    FingerType,
    Gender,
    Hand,
    IrisSample,
)

# Fingerprint: {id}__{M|F}_{Left|Right}_{index|little|middle|ring|thumb}_finger.BMP
_FINGERPRINT_PATTERN = re.compile(
    r"^(\d+)__([MF])_(Left|Right)_(index|little|middle|ring|thumb)_finger\.BMP$",
    re.IGNORECASE,
)

# Iris: {prefix}{l|r}{1-5}.bmp
_IRIS_PATTERN = re.compile(r"^(.+)([lr])([1-5])\.bmp$", re.IGNORECASE)

_FINGER_TYPE_MAP: dict[str, FingerType] = {
    "index": FingerType.INDEX,
    "little": FingerType.LITTLE,
    "middle": FingerType.MIDDLE,
    "ring": FingerType.RING,
    "thumb": FingerType.THUMB,
}


def parse_fingerprint_filename(file_path: Path, label: int) -> FingerprintSample:
    """Parse fingerprint filename into typed FingerprintSample.

    Expected format: {subject_id}__{M|F}_{Left|Right}_{finger_type}_finger.BMP
    Example: 42__F_Left_index_finger.BMP

    Args:
        file_path: Path to the fingerprint image (used for filename parsing).
        label: 0-based class label for training.

    Returns:
        FingerprintSample with parsed metadata.

    Raises:
        ValueError: If filename does not match expected pattern.
    """
    name = file_path.name
    match = _FINGERPRINT_PATTERN.match(name)
    if match is None:
        msg = f"Invalid fingerprint filename format: {name!r}"
        raise ValueError(msg)

    subject_id_str, gender_str, hand_str, finger_str = match.groups()
    subject_id = int(subject_id_str)
    gender = Gender.M if gender_str.upper() == "M" else Gender.F
    hand = Hand.LEFT if hand_str == "Left" else Hand.RIGHT
    finger_type = _FINGER_TYPE_MAP[finger_str.lower()]

    return FingerprintSample(
        subject_id=subject_id,
        file_path=file_path.resolve(),
        modality=BiometricModality.FINGERPRINT,
        label=label,
        gender=gender,
        hand=hand,
        finger_type=finger_type,
    )


def parse_iris_path(
    file_path: Path,
    subject_id: int,
    is_left: bool,
    label: int,
) -> IrisSample:
    """Parse iris file path into typed IrisSample.

    Iris filenames vary by subject (e.g. aeval1.bmp, winl1.bmp, christinel1.bmp).
    Format: {prefix}{l|r}{1-5}.bmp where l=left, r=right.

    Args:
        file_path: Path to the iris image.
        subject_id: Subject ID from parent directory (Data/{subject_id}/left|right/).
        is_left: True if left eye (from 'left' folder), False if right.
        label: 0-based class label for training.

    Returns:
        IrisSample with parsed metadata.

    Raises:
        ValueError: If filename does not match expected pattern.
    """
    name = file_path.name
    match = _IRIS_PATTERN.match(name)
    if match is None:
        msg = f"Invalid iris filename format: {name!r}"
        raise ValueError(msg)

    _prefix, _lr, seq_str = match.groups()
    sequence = int(seq_str)
    modality = BiometricModality.IRIS_LEFT if is_left else BiometricModality.IRIS_RIGHT

    return IrisSample(
        subject_id=subject_id,
        file_path=file_path.resolve(),
        modality=modality,
        label=label,
        sequence=sequence,
    )
