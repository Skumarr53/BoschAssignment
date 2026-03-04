"""Integration test: end-to-end inference pipeline.

[Phase 5a] Load checkpoint, run prediction on batch, assert output shape and range.
"""

from pathlib import Path

import pytest
from PIL import Image
from torch.optim import Adam

from biometric.data.datamodule import BiometricDataModule
from biometric.inference.pipeline import InferencePipeline, load_model, predict
from biometric.models.fusion_model import MultimodalFusionModel
from biometric.training import CheckpointCallback, MetricLoggerCallback, Trainer, seed_everything


def _synthetic_data_10_subjects(tmp_path: Path) -> Path:
    """Create synthetic data with 10 subjects."""
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


def test_inference_pipeline_end_to_end(synthetic_10: Path, tmp_path: Path) -> None:
    """Train 1 epoch, save checkpoint, load via pipeline, run predict."""
    seed_everything(42)
    num_classes = 10
    embedding_dim = 32
    ckpt_dir = tmp_path / "ckpt"

    dm = BiometricDataModule(
        synthetic_10,
        batch_size=4,
        num_workers=0,
        use_cache=False,
        train_ratio=0.7,
        val_ratio=0.2,
    )
    dm.setup(stage="fit")

    model = MultimodalFusionModel(num_classes=num_classes, embedding_dim=embedding_dim)
    optimizer = Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        checkpoint_dir=ckpt_dir,
        callbacks=[
            CheckpointCallback(checkpoint_dir=ckpt_dir),
            MetricLoggerCallback(),
        ],
        device="cpu",
    )
    trainer.fit(max_epochs=1)

    ckpt_path = ckpt_dir / "best.pt"
    assert ckpt_path.exists(), "Checkpoint should be saved after 1 epoch"

    pipeline = InferencePipeline(
        checkpoint_path=ckpt_path,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        device="cpu",
    )

    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))

    logits = pipeline.predict(batch)
    assert logits.shape == (batch["label"].shape[0], num_classes)
    assert logits.isfinite().all()

    probs = pipeline.predict_proba(batch)
    assert probs.shape == logits.shape
    assert (probs >= 0).all() and (probs <= 1).all()
    assert probs.sum(dim=1).allclose(probs.new_ones(probs.shape[0]))

    pred_classes = pipeline.predict_classes(batch)
    assert pred_classes.shape == (batch["label"].shape[0],)
    assert (pred_classes >= 0).all() and (pred_classes < num_classes).all()


def test_load_model_and_predict(synthetic_10: Path, tmp_path: Path) -> None:
    """Test load_model and predict functions directly."""
    seed_everything(42)
    num_classes = 10
    embedding_dim = 32

    model = MultimodalFusionModel(num_classes=num_classes, embedding_dim=embedding_dim)
    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    loaded = load_model(
        checkpoint_path=ckpt_path,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        device="cpu",
    )

    dm = BiometricDataModule(
        synthetic_10,
        batch_size=2,
        num_workers=0,
        use_cache=False,
    )
    dm.setup()
    batch = next(iter(dm.test_dataloader()))

    logits = predict(loaded, batch, device="cpu")
    assert logits.shape == (2, num_classes)
    assert logits.isfinite().all()
