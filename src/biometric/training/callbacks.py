"""Training callbacks: checkpoint, early stopping, metric logging.

[Phase 3b] Callbacks receive trainer ref for model/optimizer access.
Interface: on_epoch_start(trainer, epoch), on_epoch_end(trainer, epoch, metrics).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from biometric.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class TrainerLike(Protocol):
    """Minimal trainer interface for callbacks. Trainer (Phase 3c) will implement this."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    checkpoint_dir: Path
    should_stop: bool


@runtime_checkable
class TrainerCallback(Protocol):
    """Protocol for training callbacks."""

    def on_epoch_start(self, trainer: TrainerLike, epoch: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch with metrics."""
        ...


class CheckpointCallback:
    """Saves best model checkpoint when monitored metric improves.

    [Phase 3b] Saves model state_dict only (optimizer optional for full resume).
    [Scalability] When is_main_process=False (DDP rank != 0), skips saving.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        file_prefix: str = "best",
        is_main_process: bool = True,
    ) -> None:
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            monitor: Metric name to monitor (e.g. 'val_loss', 'val_accuracy').
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
            file_prefix: Prefix for checkpoint filename (e.g. best -> best.pt).
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._monitor = monitor
        self._mode = mode
        self._file_prefix = file_prefix
        self._best: float | None = None
        self._is_main_process = is_main_process

    def on_epoch_start(self, trainer: TrainerLike, epoch: int) -> None:
        """No-op at epoch start."""
        pass

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, metrics: dict[str, float]) -> None:
        """Save checkpoint if monitored metric improves."""
        if not self._is_main_process or self._monitor not in metrics:
            return
        value = metrics[self._monitor]
        is_better = (
            self._best is None
            or (self._mode == "min" and value < self._best)
            or (self._mode == "max" and value > self._best)
        )
        if is_better:
            self._best = value
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self._checkpoint_dir / f"{self._file_prefix}.pt"
            raw = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            if not isinstance(raw, nn.Module):
                raise TypeError("trainer.model must be nn.Module")
            model = raw
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                path,
            )
            logger.info(
                "checkpoint_saved",
                path=str(path),
                epoch=epoch,
                monitor=self._monitor,
                value=value,
            )


class EarlyStoppingCallback:
    """Stops training when monitored metric does not improve for N epochs.

    [Phase 3b] Sets trainer.should_stop = True when patience is exceeded.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 3,
    ) -> None:
        """Initialize early stopping.

        Args:
            monitor: Metric name to monitor.
            mode: 'min' for loss, 'max' for accuracy.
            patience: Number of epochs without improvement before stopping.
        """
        self._monitor = monitor
        self._mode = mode
        self._patience = patience
        self._best: float | None = None
        self._patience_counter = 0

    def on_epoch_start(self, trainer: TrainerLike, epoch: int) -> None:
        """No-op at epoch start."""
        pass

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, metrics: dict[str, float]) -> None:
        """Check if metric improved; set trainer.should_stop if patience exceeded."""
        if self._monitor not in metrics:
            return
        value = metrics[self._monitor]
        is_better = (
            self._best is None
            or (self._mode == "min" and value < self._best)
            or (self._mode == "max" and value > self._best)
        )
        if is_better:
            self._best = value
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            if self._patience_counter >= self._patience:
                trainer.should_stop = True
                logger.info(
                    "early_stopping_triggered",
                    epoch=epoch,
                    monitor=self._monitor,
                    patience=self._patience,
                    best=self._best,
                )


class MetricLoggerCallback:
    """Logs epoch metrics via structlog at INFO level.

    [Phase 3b] Structured logging for ELK/Loki compatibility.
    [Scalability] When is_main_process=False, skips logging to avoid duplicate output.
    """

    def __init__(self, is_main_process: bool = True) -> None:
        self._is_main_process = is_main_process

    def on_epoch_start(self, trainer: TrainerLike, epoch: int) -> None:
        """Log epoch start."""
        if self._is_main_process:
            logger.info("epoch_start", epoch=epoch)

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch end with metrics."""
        if self._is_main_process:
            logger.info("epoch_end", epoch=epoch, **metrics)


class MLflowCallback:
    """Logs params, metrics, and artifacts to MLflow when mlflow is installed.

    [Phase 3e] Optional dep — no-op when mlflow not installed.
    [Scalability] When is_main_process=False (DDP rank != 0), skips logging.
    """

    def __init__(
        self,
        experiment_name: str = "biometric",
        params: dict[str, str | int | float] | None = None,
        is_main_process: bool = True,
    ) -> None:
        """Initialize MLflow callback.

        Args:
            experiment_name: MLflow experiment name.
            params: Hyperparameters to log at run start.
        """
        self._experiment_name = experiment_name
        self._params = dict(params) if params else {}
        self._run_id: str | None = None
        self._is_main_process = is_main_process

    def on_epoch_start(self, trainer: TrainerLike, epoch: int) -> None:
        """Start MLflow run on first epoch; log params."""
        if not self._is_main_process or epoch != 0:
            return
        self._start_run(trainer)

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics to MLflow."""
        if not self._is_main_process or self._run_id is None:
            return
        try:
            import mlflow

            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=epoch)
        except ImportError:
            pass

    def on_fit_end(self, trainer: TrainerLike, final_epoch: int) -> None:
        """Log best checkpoint as artifact and end MLflow run."""
        if not self._is_main_process or self._run_id is None:
            return
        try:
            import mlflow

            best_ckpt = trainer.checkpoint_dir / "best.pt"
            if best_ckpt.exists():
                mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints")
            mlflow.end_run()
        except ImportError:
            pass
        except Exception as e:
            logger.warning("mlflow_fit_end_failed", error=str(e))

    def _start_run(self, trainer: TrainerLike) -> None:
        """Start MLflow run and log params. Run stays open until process exits."""
        try:
            import mlflow

            mlflow.set_experiment(self._experiment_name)
            mlflow.start_run()  # no with: run stays open for whole training
            run = mlflow.active_run()
            self._run_id = run.info.run_id if run else None
            for k, v in self._params.items():
                mlflow.log_param(k, str(v))
        except ImportError:
            pass
        except Exception as e:
            logger.warning("mlflow_start_failed", error=str(e))
