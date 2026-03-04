#!/usr/bin/env -S uv run python
"""Inference CLI for multimodal biometric model.

[Phase 5a] Load checkpoint and run prediction on data directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from biometric.data.datamodule import BiometricDataModule
from biometric.inference.pipeline import InferencePipeline
from biometric.training import seed_everything
from biometric.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Run inference from checkpoint on test data."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    seed_everything(42)

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / str(cfg.data.data_root)
    if not data_root.exists():
        logger.error("data_root_not_found", path=str(data_root))
        return 1

    ckpt_path = Path(cfg.get("inference", {}).get("checkpoint_path", "checkpoints/best.pt"))
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    if not ckpt_path.exists():
        logger.error("checkpoint_not_found", path=str(ckpt_path))
        return 1

    model_cfg = cfg.model
    num_classes = int(model_cfg.get("num_classes", 45))
    embedding_dim = int(model_cfg.get("embedding_dim", 128))

    dm = BiometricDataModule(
        data_root=data_root,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        use_cache=cfg.data.get("use_cache", True),
    )
    dm.setup()

    pipeline = InferencePipeline(
        checkpoint_path=ckpt_path,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        device="cpu",
    )

    correct = 0
    total = 0
    for batch in dm.test_dataloader():
        pred = pipeline.predict_classes(batch)
        labels = batch["label"]
        correct += (pred == labels).sum().item()
        total += labels.shape[0]

    acc = correct / total if total > 0 else 0.0
    logger.info("inference_complete", correct=correct, total=total, accuracy=round(acc, 4))
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
