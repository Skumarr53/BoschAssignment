"""Shared Pydantic models and type aliases for the biometric package."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class BiometricModality(StrEnum):
    """Supported biometric modalities."""

    FINGERPRINT = "fingerprint"
    IRIS_LEFT = "iris_left"
    IRIS_RIGHT = "iris_right"


class Gender(StrEnum):
    """Subject gender from fingerprint filename."""

    M = "M"
    F = "F"


class Hand(StrEnum):
    """Hand side from fingerprint filename."""

    LEFT = "Left"
    RIGHT = "Right"


class FingerType(StrEnum):
    """Finger type from fingerprint filename."""

    INDEX = "index_finger"
    LITTLE = "little_finger"
    MIDDLE = "middle_finger"
    RING = "ring_finger"
    THUMB = "thumb_finger"


class BiometricSample(BaseModel):
    """Base model for a single biometric sample."""

    subject_id: int = Field(..., description="Subject identifier (1-based)")
    file_path: Path = Field(..., description="Absolute path to image file")
    modality: BiometricModality = Field(..., description="Biometric modality")
    label: int = Field(..., description="Class label for training (0-based subject index)")


class FingerprintSample(BiometricSample):
    """Fingerprint sample with metadata parsed from filename."""

    modality: BiometricModality = Field(
        default=BiometricModality.FINGERPRINT, description="Always fingerprint"
    )
    gender: Gender = Field(..., description="Subject gender from filename")
    hand: Hand = Field(..., description="Left or right hand")
    finger_type: FingerType = Field(..., description="Finger type")

    model_config = {"frozen": True}


class IrisSample(BiometricSample):
    """Iris sample with metadata from path and filename."""

    sequence: int = Field(..., ge=1, le=5, description="Capture sequence number 1-5")

    model_config = {"frozen": True}
