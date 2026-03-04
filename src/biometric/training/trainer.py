"""Custom Trainer with forward, loss, backward, epoch loop and callbacks.

[Phase 3c] Implements TrainerLike for callback integration.
[Phase 3e] AMP + gradient accumulation.
[AMD] Device-agnostic AMP (torch.amp) for NVIDIA CUDA and AMD ROCm.
[Scalability] DDP wrap for multi-GPU/multi-node training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from biometric.training.callbacks import TrainerLike
from biometric.utils.logging import get_logger

logger = get_logger(__name__)


def _get_amp_scaler(device_type: str) -> Any:
    """Return GradScaler for AMP. Prefer torch.amp (CUDA/ROCm), fallback to torch.cuda.amp."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type)
    if device_type == "cuda" and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()
    return None


def _amp_autocast_context(device_type: str) -> Any:
    """Return autocast context manager for AMP. Works with NVIDIA CUDA and AMD ROCm."""
    if hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    if device_type == "cuda" and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast()
    from contextlib import nullcontext

    return nullcontext()


class Trainer(TrainerLike):
    """Custom training loop: forward, loss, backward, epoch loop, callbacks.

    [Phase 3c] Core loop. [Phase 3e] AMP + gradient accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        checkpoint_dir: Path | str,
        callbacks: list[Any] | None = None,
        device: str | torch.device | None = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        use_ddp: bool = False,
        local_rank: int = 0,
    ) -> None:
        """Initialize Trainer.

        Args:
            model: MultimodalFusionModel or compatible nn.Module.
            optimizer: Optimizer (e.g. Adam, SGD).
            train_loader: Training DataLoader (batch dict: fingerprint, iris, label).
            val_loader: Validation DataLoader.
            checkpoint_dir: Directory for checkpoints (used by CheckpointCallback).
            callbacks: List of callbacks with on_epoch_start/on_epoch_end.
            device: Device for model and data. Auto-detects CUDA/ROCm if available.
            use_amp: Enable mixed precision (CUDA/AMD ROCm only).
            gradient_accumulation_steps: Accumulate gradients over N steps before optimizer.step().
            use_ddp: Wrap model in DistributedDataParallel for multi-GPU/multi-node.
            local_rank: GPU device index for this process (when use_ddp).
        """
        self.model = model
        self.optimizer = optimizer
        self._train_loader = train_loader
        self._val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self._callbacks = list(callbacks) if callbacks else []
        self._use_ddp = bool(use_ddp)
        self._local_rank = int(local_rank)
        self._device = self._resolve_device(device, local_rank)
        self.should_stop = False

        self._use_amp = bool(use_amp) and self._device.type == "cuda"
        self._gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        scaler = _get_amp_scaler("cuda") if self._use_amp else None
        self._scaler = scaler

        self.model.to(self._device)
        if self._use_ddp:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self._device.index] if self._device.type == "cuda" else None,
            )
        self._criterion = nn.CrossEntropyLoss()

    def _resolve_device(
        self,
        device: str | torch.device | None,
        local_rank: int = 0,
    ) -> torch.device:
        """Resolve device: CUDA/ROCm (NVIDIA or AMD) if available, else CPU."""
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("cpu")

    def fit(self, max_epochs: int) -> None:
        """Run training for up to max_epochs. Stops early if callback sets should_stop."""
        logger.info(
            "training_start",
            max_epochs=max_epochs,
            device=str(self._device),
            train_batches=len(self._train_loader),
            val_batches=len(self._val_loader),
            use_amp=self._use_amp,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            use_ddp=self._use_ddp,
        )
        final_epoch = 0
        for epoch in range(max_epochs):
            final_epoch = epoch
            if hasattr(self._train_loader.sampler, "set_epoch"):
                self._train_loader.sampler.set_epoch(epoch)
            self._invoke_callbacks("on_epoch_start", epoch)

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)

            metrics = {**train_metrics, **val_metrics}
            self._invoke_callbacks("on_epoch_end", epoch, metrics)

            if self.should_stop:
                logger.info("training_stopped_early", epoch=epoch)
                break

        self._invoke_callbacks("on_fit_end", final_epoch)

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch with gradient accumulation. Returns train_loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self._train_loader):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            if (batch_idx + 1) % self._gradient_accumulation_steps == 0:
                self._optimizer_step()
                self.optimizer.zero_grad()

        if num_batches % self._gradient_accumulation_steps != 0:
            self._optimizer_step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}

    def _optimizer_step(self) -> None:
        """Perform optimizer step (with GradScaler when AMP enabled)."""
        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

    def _train_step(self, batch: dict[str, Any]) -> float:
        """Single training step: forward, loss, backward (with optional AMP)."""
        fp = batch["fingerprint"].to(self._device)
        iris = batch["iris"].to(self._device)
        labels = batch["label"].to(self._device)

        loss_scale = 1.0 / self._gradient_accumulation_steps

        if self._scaler is not None:
            with _amp_autocast_context("cuda"):
                logits = self.model(fingerprint=fp, iris=iris)
                loss = self._criterion(logits, labels) * loss_scale
            self._scaler.scale(loss).backward()
        else:
            logits = self.model(fingerprint=fp, iris=iris)
            loss = self._criterion(logits, labels) * loss_scale
            loss.backward()

        return float(loss.item() * self._gradient_accumulation_steps)

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        """Run one validation epoch. Returns val_loss, val_accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self._val_loader:
                fp = batch["fingerprint"].to(self._device)
                iris = batch["iris"].to(self._device)
                labels = batch["label"].to(self._device)

                logits = self.model(fingerprint=fp, iris=iris)
                loss = self._criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self._val_loader) if len(self._val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def _invoke_callbacks(
        self,
        method: str,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Invoke callback method on all callbacks."""
        for cb in self._callbacks:
            if not hasattr(cb, method):
                continue
            fn = getattr(cb, method)
            if metrics is not None:
                fn(self, epoch, metrics)
            else:
                fn(self, epoch)
