"""Distributed training utilities for multi-GPU and multi-node scaling.

[Scalability] Detects torchrun/horovod env vars (RANK, WORLD_SIZE, LOCAL_RANK)
and provides init_process_group for PyTorch DDP. Single-process when not distributed.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

from biometric.utils.logging import get_logger

logger = get_logger(__name__)


class DistributedInfo(NamedTuple):
    """Distributed training context from environment."""

    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool


def get_distributed_info() -> DistributedInfo:
    """Read RANK, WORLD_SIZE, LOCAL_RANK from env (set by torchrun, horovodrun).

    Returns:
        DistributedInfo with rank=0, world_size=1 when not distributed.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    return DistributedInfo(
        rank=rank, world_size=world_size, local_rank=local_rank, is_distributed=is_distributed
    )


def init_process_group_if_needed(
    backend: str = "nccl",
) -> DistributedInfo:
    """Initialize torch.distributed when WORLD_SIZE > 1. No-op when single process.

    Call once at startup before creating model/DataLoader. torchrun sets
    MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK.

    Args:
        backend: "nccl" (GPU) or "gloo" (CPU). nccl preferred for multi-GPU.

    Returns:
        DistributedInfo for the current process.
    """
    info = get_distributed_info()
    if not info.is_distributed:
        return info

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
        logger.info(
            "distributed_init",
            rank=info.rank,
            world_size=info.world_size,
            local_rank=info.local_rank,
        )
    return info


def is_main_process(info: DistributedInfo | None = None) -> bool:
    """True if this process should perform checkpointing, logging, etc."""
    if info is None:
        info = get_distributed_info()
    return info.rank == 0
