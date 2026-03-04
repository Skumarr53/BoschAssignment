#!/usr/bin/env -S uv run python
"""Build and save PyArrow metadata cache for biometric dataset.

Usage:
    uv run python scripts/preprocess_cache.py [--root Data] [--output path]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biometric.data.arrow_cache import build_cache, get_cache_path


def main() -> int:
    """Build Arrow cache and save to Parquet."""
    parser = argparse.ArgumentParser(
        description="Pre-cache biometric metadata to Parquet for faster dataset init."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Data"),
        help="Path to Data directory (default: Data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: root/cache_filename from config)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="biometric_metadata.parquet",
        help="Cache filename when --output not set (default: biometric_metadata.parquet)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"Error: Data root not found: {root}")
        return 1

    cache_path = args.output.resolve() if args.output else get_cache_path(root, args.filename)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    table = build_cache(root, cache_path=cache_path)
    print(f"Built cache: {table.num_rows} rows -> {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
