"""PyArrow metadata cache for biometric dataset.

Scans the filesystem once, builds an Arrow table of metadata (subject_id,
modality, filepath, gender, hand, finger_type, sequence, filesize, label),
saves as Parquet. Subsequent loads skip filesystem traversal.

Uses native Arrow nulls for missing fields (e.g. iris rows have null gender/hand).
Includes staleness detection via stored file-count hash.
"""

import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from biometric.utils.discovery import discover_subjects
from biometric.utils.logging import get_logger
from biometric.utils.parser import parse_fingerprint_filename, parse_iris_path
from biometric.utils.types import BiometricModality

logger = get_logger(__name__)

CACHE_SCHEMA = pa.schema(
    [
        ("subject_id", pa.int32()),
        ("modality", pa.string()),
        ("filepath", pa.string()),
        ("gender", pa.string()),
        ("hand", pa.string()),
        ("finger_type", pa.string()),
        ("sequence", pa.int32()),
        ("filesize", pa.int64()),
        ("label", pa.int32()),
    ]
)

CACHE_METADATA_KEY = b"source_fingerprint"

DEFAULT_CACHE_FILENAME = "biometric_metadata.parquet"


def _compute_source_fingerprint(root: Path) -> str:
    """Compute a fast fingerprint of the data directory for staleness detection.

    Hashes the sorted list of image file paths + their mtime.
    This catches added/removed/modified files without reading content.
    """
    entries: list[str] = []
    for _ext, pattern in [
        ("BMP", "**/Fingerprint/*.BMP"),
        ("bmp", "**/left/*.bmp"),
        ("bmp", "**/right/*.bmp"),
    ]:
        for p in sorted(root.glob(pattern)):
            try:
                stat = p.stat()
                entries.append(f"{p.relative_to(root)}:{stat.st_mtime_ns}")
            except OSError:
                continue
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]


def _scan_fingerprints(root: Path, subject_to_label: dict[int, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in sorted(subject_to_label.keys()):
        label = subject_to_label[sid]
        fp_dir = root / str(sid) / "Fingerprint"
        if not fp_dir.exists():
            continue
        for path in sorted(fp_dir.glob("*.BMP")):
            try:
                sample = parse_fingerprint_filename(path, label=label)
                stat = path.stat()
                rows.append(
                    {
                        "subject_id": sample.subject_id,
                        "modality": BiometricModality.FINGERPRINT.value,
                        "filepath": str(sample.file_path),
                        "gender": sample.gender.value,
                        "hand": sample.hand.value,
                        "finger_type": sample.finger_type.value,
                        "sequence": None,
                        "filesize": stat.st_size,
                        "label": sample.label,
                    }
                )
            except ValueError as e:
                logger.warning("parse_skipped", path=str(path), reason=str(e), subject_id=sid)
    return rows


def _scan_iris(root: Path, subject_to_label: dict[int, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in sorted(subject_to_label.keys()):
        label = subject_to_label[sid]
        for is_left, subdir in [(True, "left"), (False, "right")]:
            sub_path = root / str(sid) / subdir
            if not sub_path.exists():
                continue
            modality = BiometricModality.IRIS_LEFT if is_left else BiometricModality.IRIS_RIGHT
            for path in sorted(sub_path.glob("*.bmp")):
                try:
                    sample = parse_iris_path(path, sid, is_left=is_left, label=label)
                    stat = path.stat()
                    rows.append(
                        {
                            "subject_id": sample.subject_id,
                            "modality": modality.value,
                            "filepath": str(sample.file_path),
                            "gender": None,
                            "hand": None,
                            "finger_type": None,
                            "sequence": sample.sequence,
                            "filesize": stat.st_size,
                            "label": sample.label,
                        }
                    )
                except ValueError as e:
                    logger.warning("parse_skipped", path=str(path), reason=str(e), subject_id=sid)
    return rows


def build_cache(
    root: Path | str,
    cache_path: Path | str | None = None,
) -> pa.Table:
    """Scan filesystem and build Arrow metadata table.

    Args:
        root: Path to Data directory.
        cache_path: Optional path to save Parquet. If None, uses root / DEFAULT_CACHE_FILENAME.

    Returns:
        Arrow table with cached metadata. Iris rows have null gender/hand/finger_type;
        fingerprint rows have null sequence.
    """
    root = Path(root).resolve()
    subject_ids = discover_subjects(root)
    subject_to_label = {sid: i for i, sid in enumerate(subject_ids)}

    rows: list[dict[str, Any]] = []
    rows.extend(_scan_fingerprints(root, subject_to_label))
    rows.extend(_scan_iris(root, subject_to_label))

    table = pa.Table.from_pylist(rows, schema=CACHE_SCHEMA)

    fingerprint = _compute_source_fingerprint(root)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), CACHE_METADATA_KEY: fingerprint.encode()}
    )

    if cache_path is not None:
        cache_path = Path(cache_path).resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, cache_path)
        logger.info(
            "arrow_cache_built",
            row_count=table.num_rows,
            cache_path=str(cache_path),
            root=str(root),
            source_fingerprint=fingerprint,
        )

    return table


def load_cache(cache_path: Path | str) -> pa.Table:
    """Load metadata cache from Parquet file.

    Args:
        cache_path: Path to Parquet file.

    Returns:
        Arrow table with cached metadata.
    """
    path = Path(cache_path).resolve()
    table = pq.read_table(path)
    logger.info("arrow_cache_loaded", row_count=table.num_rows, cache_path=str(path))
    return table


def is_cache_stale(cache_path: Path | str, root: Path | str) -> bool:
    """Check if the cache is stale by comparing source fingerprints.

    Args:
        cache_path: Path to existing Parquet cache.
        root: Data root directory.

    Returns:
        True if cache is stale (source data changed since cache was built).
    """
    path = Path(cache_path).resolve()
    if not path.exists():
        return True
    table = pq.read_table(path)
    stored: bytes = (table.schema.metadata or {}).get(CACHE_METADATA_KEY, b"")
    current = _compute_source_fingerprint(Path(root).resolve())
    stale: bool = stored.decode() != current
    return stale


def filter_by_subjects(table: pa.Table, subject_ids: list[int]) -> pa.Table:
    """Filter table to rows with subject_id in subject_ids."""
    if not subject_ids:
        return table
    mask = pa.compute.is_in(table.column("subject_id"), pa.array(subject_ids))
    return table.filter(mask)


def filter_by_modality(table: pa.Table, modality: str) -> pa.Table:
    """Filter table to rows with given modality."""
    mask = pa.compute.equal(table.column("modality"), modality)
    return table.filter(mask)


def cache_exists(cache_path: Path | str) -> bool:
    """Return True if cache file exists and is readable."""
    return Path(cache_path).resolve().exists()


def get_cache_path(root: Path | str, filename: str = DEFAULT_CACHE_FILENAME) -> Path:
    """Return canonical cache path for a given data root."""
    return Path(root).resolve() / filename
