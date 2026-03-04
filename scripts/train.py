#!/usr/bin/env -S uv run python
"""Hydra entry point for training the multimodal biometric model.

[Phase 3d] Composes data, model, training configs; instantiates DataModule,
model, Trainer; runs trainer.fit(). Config snapshot saved by Hydra.
[Scalability] Supports DDP via torchrun when infrastructure=cluster.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py training.epochs=2 data.batch_size=8
    torchrun --nproc_per_node=4 scripts/train.py infrastructure=cluster
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.optim import Adam

from biometric.data.datamodule import BiometricDataModule
from biometric.training import (
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricLoggerCallback,
    MLflowCallback,
    Trainer,
    seed_everything,
)
from biometric.utils.distributed import (
    init_process_group_if_needed,
    is_main_process,
)


def _resolve_path(base: Path, raw: str) -> Path:
    """Resolve path: absolute if raw starts with /, else relative to base."""
    p = Path(raw)
    return p if p.is_absolute() else (base / raw).resolve()


def main() -> int:
    """Run training from Hydra config."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    project_root = Path(__file__).resolve().parent.parent
    infra = cfg.get("infrastructure", {})
    use_distributed = infra.get("distributed", False)

    dist_info = init_process_group_if_needed()
    if use_distributed and not dist_info.is_distributed:
        use_distributed = False

    seed_everything(cfg.training.seed)

    data_root_raw = os.environ.get("DATA_ROOT") or infra.get("data_root") or cfg.data.data_root
    ckpt_dir_raw = (
        os.environ.get("CHECKPOINT_DIR")
        or infra.get("checkpoint_dir")
        or cfg.training.checkpoint_dir
    )
    data_root = _resolve_path(project_root, str(data_root_raw))
    ckpt_dir = _resolve_path(project_root, str(ckpt_dir_raw))

    dm = BiometricDataModule(
        data_root=str(data_root),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        fingerprint_size=tuple(cfg.data.fingerprint_size),
        iris_size=tuple(cfg.data.iris_size),
        seed=cfg.training.seed,
        split_by_sample=cfg.data.get("split_by_sample", False),
        cache_filename=cfg.data.get("cache_filename"),
        use_cache=True,
        use_parallel_preprocess=cfg.data.get("use_parallel_preprocess", False),
        parallel_config={
            "backend": cfg.data.get("backend", "ray"),
            "max_workers": cfg.data.get("max_workers"),
            "num_cpus_per_task": cfg.data.get("num_cpus_per_task", 1.0),
            "transform_config": dict(cfg.data.get("transform_config") or {}),
        },
    )
    dm.setup(stage="fit")

    model = instantiate(cfg.model)
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    main_proc = is_main_process(dist_info)
    if main_proc:
        config_snapshot = ckpt_dir / "config.yaml"
        with config_snapshot.open("w") as f:
            OmegaConf.save(cfg, f)

    callbacks: list[Any] = [
        CheckpointCallback(
            checkpoint_dir=ckpt_dir, monitor="val_loss", mode="min", is_main_process=main_proc
        ),
        EarlyStoppingCallback(monitor="val_loss", mode="min", patience=3),
        MetricLoggerCallback(is_main_process=main_proc),
    ]
    try:
        import mlflow  # noqa: F401

        callbacks.append(
            MLflowCallback(
                experiment_name="biometric",
                params={
                    "epochs": cfg.training.epochs,
                    "learning_rate": cfg.training.learning_rate,
                    "batch_size": cfg.data.batch_size,
                    "use_amp": cfg.training.get("use_amp", False),
                    "gradient_accumulation_steps": cfg.training.get(
                        "gradient_accumulation_steps", 1
                    ),
                },
                is_main_process=main_proc,
            )
        )
    except ImportError:
        pass

    train_loader = dm.train_dataloader(
        rank=dist_info.rank if use_distributed else None,
        world_size=dist_info.world_size if use_distributed else None,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=dm.val_dataloader(),
        checkpoint_dir=ckpt_dir,
        callbacks=callbacks,
        use_amp=cfg.training.get("use_amp", False),
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        use_ddp=use_distributed,
        local_rank=dist_info.local_rank,
    )

    trainer.fit(max_epochs=cfg.training.epochs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
