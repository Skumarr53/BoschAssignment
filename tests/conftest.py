"""Shared pytest fixtures for biometric tests.

Provides both the real Data/ path and a synthetic data tree that works in CI.
"""

import os
from pathlib import Path

import pytest
from PIL import Image


def pytest_configure() -> None:
    """Reduce log noise during tests."""
    os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def data_root(project_root: Path) -> Path:
    """Return the real Data directory path (dataset root)."""
    return project_root / "Data"


@pytest.fixture
def synthetic_data(tmp_path: Path) -> Path:
    """Create a minimal synthetic biometric data tree for testing.

    Structure per subject (3 subjects: 1, 2, 3):
        {root}/{id}/Fingerprint/  — 10 BMP files per subject
        {root}/{id}/left/         — 5 BMP files per subject
        {root}/{id}/right/        — 5 BMP files per subject

    Returns:
        Path to the synthetic Data root.
    """
    genders = ["M", "F"]
    hands = ["Left", "Right"]
    fingers = ["index", "little", "middle", "ring", "thumb"]

    for sid in [1, 2, 3]:
        fp_dir = tmp_path / str(sid) / "Fingerprint"
        fp_dir.mkdir(parents=True)
        left_dir = tmp_path / str(sid) / "left"
        left_dir.mkdir(parents=True)
        right_dir = tmp_path / str(sid) / "right"
        right_dir.mkdir(parents=True)

        gender = genders[sid % len(genders)]
        for hand in hands:
            for finger in fingers:
                fname = f"{sid}__{gender}_{hand}_{finger}_finger.BMP"
                img = Image.new("L", (96, 103), color=sid * 10)
                img.save(fp_dir / fname)

        prefix = f"subj{sid}"
        for i in range(1, 6):
            img_left = Image.new("RGB", (320, 240), color=(sid * 20, 100, 50))
            img_left.save(left_dir / f"{prefix}l{i}.bmp")
            img_right = Image.new("RGB", (320, 240), color=(sid * 20, 50, 100))
            img_right.save(right_dir / f"{prefix}r{i}.bmp")

    return tmp_path
