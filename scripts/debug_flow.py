#!/usr/bin/env -S uv run python
"""Debug script to exercise the full biometric data pipeline.

Creates synthetic data, builds cache, runs DataModule, and loads one batch.
Use with VS Code debugger (F5) to step through the flow.

Usage:
    uv run python scripts/debug_flow.py [--data-root PATH]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image


def _create_synthetic_data(root: Path) -> None:
    """Create minimal synthetic biometric tree (3 subjects, fingerprint + iris)."""
    genders = ["M", "F"]
    hands = ["Left", "Right"]
    fingers = ["index", "little", "middle", "ring", "thumb"]

    for sid in [1, 2, 3]:
        fp_dir = root / str(sid) / "Fingerprint"
        fp_dir.mkdir(parents=True, exist_ok=True)
        left_dir = root / str(sid) / "left"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir = root / str(sid) / "right"
        right_dir.mkdir(parents=True, exist_ok=True)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug the biometric data pipeline end-to-end.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root (default: temp dir with synthetic data)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Keep synthetic data in /tmp/debug_biometric (don't delete)",
    )
    args = parser.parse_args()

    if args.data_root is not None:
        data_root = args.data_root.resolve()
        if not data_root.exists():
            print(f"Error: Data root not found: {data_root}")
            return 1
        print(f"Using existing data root: {data_root}")
    else:
        if args.persist:
            data_root = Path("/tmp/debug_biometric")
            data_root.mkdir(parents=True, exist_ok=True)
        else:
            data_root = Path(tempfile.mkdtemp(prefix="biometric_debug_"))
        _create_synthetic_data(data_root)
        print(f"Created synthetic data at: {data_root}")

    # --- 1. Discovery ---
    from biometric.utils.discovery import discover_subjects

    subject_ids = discover_subjects(data_root)
    print(f"Discovered {len(subject_ids)} subjects: {subject_ids}")

    # --- 2. Cache build ---
    from biometric.data.arrow_cache import build_cache, get_cache_path

    cache_path = get_cache_path(data_root, "biometric_metadata.parquet")
    table = build_cache(data_root, cache_path=cache_path)
    print(f"Cache built: {table.num_rows} rows -> {cache_path}")

    # --- 3. DataModule setup ---
    from biometric.data.datamodule import BiometricDataModule

    dm = BiometricDataModule(
        data_root,
        batch_size=4,
        num_workers=0,
        fingerprint_size=(96, 96),
        iris_size=(224, 224),
        use_cache=True,
    )
    dm.setup(stage="fit")
    print("DataModule setup complete (train + val)")

    # --- 4. Load one batch ---
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    fp_tensor = batch["fingerprint"]
    iris_tensor = batch["iris"]
    labels = batch["label"]
    print(
        f"First batch: fingerprint {fp_tensor.shape}, iris {iris_tensor.shape}, "
        f"labels {labels.shape}"
    )

    if not args.persist and args.data_root is None:
        import shutil

        shutil.rmtree(data_root, ignore_errors=True)
        print("Cleaned up temp data")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
