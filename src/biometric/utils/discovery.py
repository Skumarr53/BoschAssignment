"""Centralized subject discovery and path validation for biometric dataset.

Single source of truth for finding subjects in the Data directory structure.
All modules must use these functions instead of rolling their own.
"""

from pathlib import Path

from biometric.utils.logging import get_logger

logger = get_logger(__name__)


def discover_subjects(root: Path | str) -> list[int]:
    """Discover subject IDs from directory structure.

    A valid subject has at least one of:
    - ``{root}/{id}/Fingerprint/*.BMP`` (fingerprint data)
    - ``{root}/{id}/left/*.bmp`` or ``{root}/{id}/right/*.bmp`` (iris data)

    Args:
        root: Path to the Data directory containing numbered subject folders.

    Returns:
        Sorted list of integer subject IDs found on disk.
    """
    root = Path(root).resolve()
    ids: list[int] = []
    if not root.exists():
        return ids
    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        has_fp = _has_files(entry / "Fingerprint", "*.BMP")
        has_iris = _has_files(entry / "left", "*.bmp") or _has_files(entry / "right", "*.bmp")
        if has_fp or has_iris:
            ids.append(int(entry.name))
    return sorted(ids)


def _has_files(directory: Path, pattern: str) -> bool:
    """Return True if directory exists and contains at least one file matching pattern."""
    if not directory.exists():
        return False
    return any(directory.glob(pattern))


def validate_path(path: Path, root: Path) -> Path:
    """Validate that path is within root directory (prevents path traversal).

    Args:
        path: File path to validate (will be resolved).
        root: Allowed root directory (will be resolved).

    Returns:
        Resolved path if valid.

    Raises:
        ValueError: If path is not within root.
    """
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        msg = f"Path {resolved} is outside allowed root {root_resolved}"
        raise ValueError(msg)
    return resolved
