# Scaling Training to Production Clusters

[Scalability] Guide for adapting the training pipeline from local/dev to production clusters with varying capacities.

## Current Capabilities

| Aspect | Local/Dev | Cluster/Production |
|--------|-----------|---------------------|
| **Config** | `infrastructure=local` (default) | `infrastructure=cluster` |
| **Paths** | Project-relative (`Data/`, `checkpoints/`) | Env vars: `DATA_ROOT`, `CHECKPOINT_DIR` |
| **Distributed** | Single process | DDP via `torchrun` |
| **Data loading** | Configurable `num_workers`, `batch_size` | Same + `DistributedSampler` |
| **Checkpointing** | All processes | Rank 0 only |
| **MLflow** | All processes | Rank 0 only |

## Local / Single-Node (Default)

```bash
uv run python scripts/train.py
uv run python scripts/train.py training.epochs=2 data.batch_size=32
```

Uses `configs/infrastructure/local.yaml`: project-relative paths, no DDP.

## Multi-GPU Single-Node (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train.py infrastructure=cluster
```

- `torchrun` sets `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`
- Model wrapped in `DistributedDataParallel`
- Training data sharded via `DistributedSampler`
- Only rank 0 saves checkpoints and logs to MLflow

## Multi-Node Cluster

```bash
# Node 0 (master)
torchrun \
  --nnodes=4 \
  --nproc_per_node=4 \
  --rdzv_id=biometric-run \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  scripts/train.py infrastructure=cluster

# Nodes 1–3: same command (torchrun discovers nodes via etcd/static)
```

Set `DATA_ROOT` and `CHECKPOINT_DIR` to shared storage (NFS, Azure Files, etc.) so all nodes access the same data and checkpoint directory.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATA_ROOT` | Override data path (e.g. `/mnt/data/biometric`) |
| `CHECKPOINT_DIR` | Override checkpoint path (e.g. `/mnt/checkpoints`) |
| `RANK`, `WORLD_SIZE`, `LOCAL_RANK` | Set by `torchrun`; do not set manually |

## Path Resolution Order

1. `DATA_ROOT` env var (cluster)
2. `infrastructure.data_root` (from config)
3. `data.data_root` (default: `Data`)

Same order for `CHECKPOINT_DIR`.

## Benchmarking for Capacity Planning

Use `scripts/benchmark.py` to find optimal `num_workers` and `batch_size` for your hardware:

```bash
uv run python scripts/benchmark.py
```

Results in `docs/performance_benchmarks.md`. Scale `batch_size` with GPU memory; scale `num_workers` with CPU cores.

## Stress Test: 10,000 Subjects

- **Arrow cache**: Essential; filesystem scan would dominate init time.
- **Sharding**: Consider partitioning metadata Parquet by subject range for very large datasets.
- **Ray**: `max_workers` and `num_cpus_per_task` scale with cluster size; tune via `data.max_workers`.
