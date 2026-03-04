"""Unit tests for training callbacks.

[Phase 3b] Tests each callback in isolation; no training loop.
"""

from pathlib import Path

import torch
from torch import nn

from biometric.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricLoggerCallback,
)


class MockTrainer:
    """Minimal trainer-like object for callback tests."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.model = nn.Linear(2, 3)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.checkpoint_dir = checkpoint_dir
        self.should_stop = False


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_saves_checkpoint_when_metric_improves_min(self, tmp_path: Path) -> None:
        """Saves checkpoint when val_loss decreases (mode=min)."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path, monitor="val_loss", mode="min")
        trainer = MockTrainer(tmp_path)

        cb.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        assert (tmp_path / "best.pt").exists()

        cb.on_epoch_end(trainer, 1, {"val_loss": 0.5})
        ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
        assert ckpt["epoch"] == 1
        assert ckpt["metrics"]["val_loss"] == 0.5

    def test_does_not_save_when_metric_worsens_min(self, tmp_path: Path) -> None:
        """Does not overwrite when val_loss increases (mode=min)."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path, monitor="val_loss", mode="min")
        trainer = MockTrainer(tmp_path)

        cb.on_epoch_end(trainer, 0, {"val_loss": 0.5})
        cb.on_epoch_end(trainer, 1, {"val_loss": 1.0})

        ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
        assert ckpt["epoch"] == 0
        assert ckpt["metrics"]["val_loss"] == 0.5

    def test_saves_checkpoint_when_metric_improves_max(self, tmp_path: Path) -> None:
        """Saves checkpoint when val_accuracy increases (mode=max)."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path, monitor="val_accuracy", mode="max")
        trainer = MockTrainer(tmp_path)

        cb.on_epoch_end(trainer, 0, {"val_accuracy": 0.5})
        cb.on_epoch_end(trainer, 1, {"val_accuracy": 0.8})

        ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
        assert ckpt["epoch"] == 1
        assert ckpt["metrics"]["val_accuracy"] == 0.8

    def test_skips_when_monitor_missing(self, tmp_path: Path) -> None:
        """No save when monitored metric is absent from metrics dict."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path, monitor="val_loss", mode="min")
        trainer = MockTrainer(tmp_path)

        cb.on_epoch_end(trainer, 0, {"train_loss": 1.0})
        assert not (tmp_path / "best.pt").exists()

    def test_on_epoch_start_no_op(self, tmp_path: Path) -> None:
        """on_epoch_start is a no-op (does not raise)."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path)
        trainer = MockTrainer(tmp_path)
        cb.on_epoch_start(trainer, 0)


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_sets_should_stop_after_patience_exceeded(self) -> None:
        """Sets trainer.should_stop when metric does not improve for patience epochs."""
        cb = EarlyStoppingCallback(monitor="val_loss", mode="min", patience=2)
        trainer = MockTrainer(Path("/tmp"))

        cb.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        assert not trainer.should_stop

        cb.on_epoch_end(trainer, 1, {"val_loss": 1.1})
        assert not trainer.should_stop

        cb.on_epoch_end(trainer, 2, {"val_loss": 1.2})
        assert trainer.should_stop

    def test_resets_patience_when_metric_improves(self) -> None:
        """Patience counter resets when metric improves."""
        cb = EarlyStoppingCallback(monitor="val_loss", mode="min", patience=2)
        trainer = MockTrainer(Path("/tmp"))

        cb.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        cb.on_epoch_end(trainer, 1, {"val_loss": 1.1})
        cb.on_epoch_end(trainer, 2, {"val_loss": 0.5})
        assert not trainer.should_stop

        cb.on_epoch_end(trainer, 3, {"val_loss": 0.6})
        cb.on_epoch_end(trainer, 4, {"val_loss": 0.7})
        assert trainer.should_stop

    def test_mode_max_for_accuracy(self) -> None:
        """Early stop with mode=max (higher is better)."""
        cb = EarlyStoppingCallback(monitor="val_accuracy", mode="max", patience=2)
        trainer = MockTrainer(Path("/tmp"))

        cb.on_epoch_end(trainer, 0, {"val_accuracy": 0.9})
        cb.on_epoch_end(trainer, 1, {"val_accuracy": 0.8})
        cb.on_epoch_end(trainer, 2, {"val_accuracy": 0.7})
        assert trainer.should_stop

    def test_skips_when_monitor_missing(self) -> None:
        """Does not set should_stop when monitor key absent."""
        cb = EarlyStoppingCallback(monitor="val_loss", mode="min", patience=1)
        trainer = MockTrainer(Path("/tmp"))

        cb.on_epoch_end(trainer, 0, {"train_loss": 1.0})
        cb.on_epoch_end(trainer, 1, {"train_loss": 1.1})
        assert not trainer.should_stop


class TestMetricLoggerCallback:
    """Tests for MetricLoggerCallback."""

    def test_on_epoch_start_and_end_no_raise(self, tmp_path: Path) -> None:
        """on_epoch_start and on_epoch_end do not raise (logging tested via integration)."""
        cb = MetricLoggerCallback()
        trainer = MockTrainer(tmp_path)

        cb.on_epoch_start(trainer, 0)
        cb.on_epoch_end(trainer, 0, {"train_loss": 0.5, "val_loss": 0.6})
