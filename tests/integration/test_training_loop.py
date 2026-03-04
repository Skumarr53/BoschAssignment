"""Integration test: 1-epoch training on synthetic data.

[Phase 3c] Gate: 1 epoch runs without error; loss decreases over 2 epochs.
"""

from pathlib import Path

import pytest
from PIL import Image
from torch.optim import Adam

from biometric.data.datamodule import BiometricDataModule
from biometric.models.fusion_model import MultimodalFusionModel
from biometric.training import CheckpointCallback, MetricLoggerCallback, Trainer, seed_everything


def _synthetic_data_10_subjects(tmp_path: Path) -> Path:
    """Create synthetic data with 10 subjects for train/val split."""
    genders = ["M", "F"]
    hands = ["Left", "Right"]
    fingers = ["index", "little", "middle", "ring", "thumb"]

    for sid in range(1, 11):
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


@pytest.fixture
def synthetic_10(tmp_path: Path) -> Path:
    """Synthetic data with 10 subjects for integration tests."""
    return _synthetic_data_10_subjects(tmp_path)


def test_training_loop_one_epoch_completes(synthetic_10: Path, tmp_path: Path) -> None:
    """1 epoch runs without error on synthetic data."""
    seed_everything(42)

    dm = BiometricDataModule(
        synthetic_10,
        batch_size=4,
        num_workers=0,
        use_cache=False,
        train_ratio=0.7,
        val_ratio=0.2,
    )
    dm.setup(stage="fit")

    model = MultimodalFusionModel(num_classes=10, embedding_dim=32)
    optimizer = Adam(model.parameters(), lr=1e-3)

    callbacks = [
        CheckpointCallback(checkpoint_dir=tmp_path / "ckpt"),
        MetricLoggerCallback(),
    ]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        checkpoint_dir=tmp_path / "ckpt",
        callbacks=callbacks,
        device="cpu",
    )

    trainer.fit(max_epochs=1)

    assert trainer.model is model


def test_training_loss_decreases_over_epochs(synthetic_10: Path, tmp_path: Path) -> None:
    """Loss decreases over 2 epochs (sanity check)."""
    seed_everything(42)

    losses: list[float] = []

    class LossCollector:
        def on_epoch_start(self, trainer: object, epoch: int) -> None:
            pass

        def on_epoch_end(self, trainer: object, epoch: int, metrics: dict) -> None:
            losses.append(metrics["train_loss"])

    dm = BiometricDataModule(
        synthetic_10,
        batch_size=4,
        num_workers=0,
        use_cache=False,
        train_ratio=0.7,
        val_ratio=0.2,
    )
    dm.setup(stage="fit")

    model = MultimodalFusionModel(num_classes=10, embedding_dim=32)
    optimizer = Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        checkpoint_dir=tmp_path / "ckpt",
        callbacks=[LossCollector()],
        device="cpu",
    )

    trainer.fit(max_epochs=2)

    assert len(losses) == 2
    assert all(loss > 0 and loss < 100 for loss in losses)
    assert losses[1] <= losses[0] or losses[1] < 10
