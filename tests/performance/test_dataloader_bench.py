"""Performance test: DataLoader throughput above threshold.

[Phase 5a] Asserts DataLoader delivers samples at acceptable rate.
Uses synthetic data; threshold tuned for CI (low) vs local (higher).
"""

import time
from pathlib import Path

import pytest

from biometric.data.datamodule import BiometricDataModule
from biometric.training import seed_everything

# Minimum samples/sec to pass. Synthetic data (3 subjects, ~30 pairs) with num_workers=0.
# CI may be slower; 30 is conservative. Local typically 500+.
MIN_SAMPLES_PER_SEC = 30


def _consume_loader(loader, num_batches: int) -> tuple[int, float]:
    """Iterate loader, return (samples_seen, wall_seconds)."""
    start = time.perf_counter_ns()
    samples = 0
    for i, batch in enumerate(loader):
        samples += batch["label"].shape[0]
        if i >= num_batches - 1:
            break
    elapsed_ns = time.perf_counter_ns() - start
    return samples, elapsed_ns / 1e9


@pytest.mark.performance
def test_dataloader_throughput_above_threshold(synthetic_data: Path) -> None:
    """DataLoader delivers samples at >= MIN_SAMPLES_PER_SEC."""
    seed_everything(42)

    dm = BiometricDataModule(
        synthetic_data,
        batch_size=4,
        num_workers=0,
        use_cache=False,
        train_ratio=0.7,
        val_ratio=0.2,
    )
    dm.setup(stage="fit")

    loader = dm.train_dataloader()
    warmup_batches = 2
    measure_batches = 5

    _consume_loader(loader, warmup_batches)
    samples, seconds = _consume_loader(loader, measure_batches)
    samples_per_sec = samples / seconds if seconds > 0 else 0.0

    assert samples_per_sec >= MIN_SAMPLES_PER_SEC, (
        f"DataLoader throughput {samples_per_sec:.1f} samples/sec "
        f"below threshold {MIN_SAMPLES_PER_SEC}"
    )
