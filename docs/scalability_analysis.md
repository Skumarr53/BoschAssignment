# Scalability Analysis

[Phase 4b] Bottleneck analysis and scaling strategies for the multimodal biometric pipeline.

## Executive Summary

The pipeline is designed for incremental scaling: Arrow cache for metadata, DDP for multi-GPU, Ray for parallel preprocessing. Primary bottlenecks are **I/O (BMP decode)** and **memory (batch size)**; compute (convolutions) is typically GPU-bound and well-utilized at moderate batch sizes.

---

## 1. Subject Scale: 45 → 45,000

### Current Behavior

- **45 subjects**: ~449 samples (fingerprint + iris pairs). Arrow cache init ~100–500 ms; Dataset build ~50 ms.
- **45,000 subjects**: ~450,000 samples. Without cache, filesystem scan would dominate (minutes to seconds).

### Scaling Strategies

| Component | Strategy | Effort |
|-----------|----------|--------|
| **Arrow cache** | Essential. Single Parquet file works up to ~10⁶ rows; beyond that, use **sharded cache** (e.g. `metadata_0000.parquet`, `metadata_0001.parquet` by subject_id range). | Medium |
| **DataLoader** | `DistributedSampler` shards per rank; no per-subject changes. | Done |
| **Memory** | `MultimodalBiometricDataset` holds pairs in memory; at 450K pairs, ~2–5 MB for indices. Acceptable. | Low |
| **Preprocessing** | Ray `max_workers` scales with cluster size; tune via `data.max_workers`.

### Sharding Example (Future)

```python
# Partition metadata by subject_id range for parallel load
def get_cache_shards(root: Path, subject_ids: list[int]) -> list[Path]:
    shard_size = 10_000
    return [root / f"metadata_{i:04d}.parquet" for i in range(0, len(subject_ids), shard_size)]
```

---

## 2. Image Resolution: 10× Increase

### Current

- Fingerprint: 96×103 → 96×96 (resize)
- Iris: 320×240 → 224×224 (resize)

### 10× Resolution (e.g. 3200×2400)

| Bottleneck | Impact | Mitigation |
|------------|--------|------------|
| **I/O** | Larger BMP files; decode time per image increases. | Streaming decode (e.g. `PIL.Image.open` with lazy load; decode only when needed). |
| **Memory** | Batch memory grows with H×W. | **Tiled loading**: load patches on demand; reduce `batch_size`; use gradient checkpointing. |
| **Compute** | Conv2d ops scale with H×W. | Lower resolution for training; use higher resolution only at inference. |

### Recommendation

- **Training**: Keep 224×224 or 256×256; resolution scaling is rarely linear for accuracy.
- **Inference**: Use higher resolution only if needed; consider multi-scale or pyramid fusion.

---

## 3. Third Modality (e.g. Face)

### Current Architecture

- `FingerprintEncoder` + `IrisEncoder` → `MultimodalFusionModel` (concatenate embeddings → classifier).
- `MultimodalBiometricDataset` pairs fingerprint + iris by subject.

### Adding Face

| Layer | Change | Effort |
|-------|--------|--------|
| **Types** | Add `BiometricModality.FACE` to `utils/types.py`. | Low |
| **Parser** | Add `parse_face_path()` or similar; extend `arrow_cache` scan. | Low |
| **Dataset** | Add `FaceDataset`; extend `MultimodalBiometricDataset` to accept triplets (fp, iris, face) or use a **registry** of modality datasets. | Medium |
| **Model** | Add `FaceEncoder`; change `classifier = nn.Linear(3 * embedding_dim, num_classes)`. | Low |
| **Config** | Add `model/face_encoder.yaml`; extend `fusion.yaml`. | Low |

### Registry Pattern (Optional)

```python
# Encoder registry: add modality without modifying core fusion logic
ENCODER_REGISTRY = {"fingerprint": FingerprintEncoder, "iris": IrisEncoder, "face": FaceEncoder}
```

No changes to existing encoders; only new encoder + config entry.

---

## 4. Single-Node vs Multi-Node Training

### Current Support

| Mode | Command | Notes |
|------|---------|-------|
| Single-node | `uv run python scripts/train.py` | Single process; `infrastructure=local`. |
| Multi-GPU | `torchrun --nproc_per_node=4 scripts/train.py infrastructure=cluster` | DDP; `DistributedSampler`. |
| Multi-node | Same `torchrun` with `--nnodes`, `--rdzv_*` | Shared `DATA_ROOT`/`CHECKPOINT_DIR` required. |

### Scaling Beyond DDP

| Option | When | Use Case |
|--------|------|----------|
| **DDP** | 1–8 nodes, 1–32 GPUs | Current; sufficient for most cases. |
| **FSDP** | Very large models; shard parameters | Not needed for current encoder sizes. |
| **Ray Train** | Heterogeneous cluster; fault tolerance | Alternative to `torchrun`; same DataLoader patterns. |

### Multi-Node Checklist

- Shared storage for `DATA_ROOT` and `CHECKPOINT_DIR` (NFS, Azure Files, etc.).
- `MASTER_ADDR` and `MASTER_PORT` set on all nodes.
- Same `torchrun` command on each node.

---

## 5. Inference at 1000 RPS

### Current State

- No inference pipeline in `src/biometric/inference/`; training loop exists.

### Scaling to 1000 RPS

| Requirement | 1000 RPS | Approach |
|-------------|----------|----------|
| **Latency** | ~1–10 ms per request | Batched inference; ONNX Runtime or TorchServe. |
| **Throughput** | 1000 samples/sec | Batch size 32–64; ~15–30 batches/sec. |
| **Model** | Optimized | `torch.compile()` or ONNX export; FP16/INT8. |

### Recommended Stack

1. **ONNX export** for deployment (CPU/GPU, multiple runtimes).
2. **TorchServe** or **Triton** for batching, dynamic batching, and autoscaling.
3. **Batch size** 32–64; dynamic batching with max latency ~50 ms.

### Rough Capacity

- GPU: ~500–2000 samples/sec on V100/A100 depending on model size.
- CPU: ~50–200 samples/sec; scale horizontally for 1000 RPS.

---

## 6. Bottleneck Analysis

### I/O Bound

| Component | Symptom | Mitigation |
|-----------|---------|------------|
| **BMP decode** | High CPU in worker; GPU underutilized. | Arrow cache (metadata); `num_workers` 4–8; parallel preprocessing (Ray). |
| **Filesystem scan** | Slow Dataset init. | Arrow cache (essential at scale). |
| **Disk throughput** | Saturates at high `num_workers`. | SSD/NVMe; reduce `num_workers` if I/O saturated. |

### Compute Bound

| Component | Symptom | Mitigation |
|-----------|---------|------------|
| **Convolutions** | GPU utilization > 90% | Increase `batch_size`; `torch.compile()`; mixed precision. |
| **Transforms** | CPU-bound in workers | Move transforms to GPU (if supported); use `num_workers` to parallelize. |

### Memory Bound

| Component | Symptom | Mitigation |
|-----------|---------|------------|
| **Batch size** | OOM on GPU | Reduce `batch_size`; gradient accumulation. |
| **Dataset** | High RAM for preloaded tensors | Use lazy loading; avoid `PreloadedMultimodalDataset` at scale. |

### Profiling Commands

```bash
# Benchmark DataLoader (includes pin_memory, prefetch_factor, persistent_workers, init time)
uv run python scripts/benchmark.py

# With cProfile (writes docs/benchmark_profile.txt)
uv run python scripts/benchmark.py benchmark.profile=true

# torch.profiler (add to training loop)
from biometric.utils.profiling import get_torch_profiler
with get_torch_profiler() as prof:
    train_one_epoch(...)
prof.export_chrome_trace("trace.json")
```

---

## 7. Summary Table

| Scale Factor | Primary Bottleneck | Recommended Action |
|--------------|--------------------|--------------------|
| 45 → 45K subjects | Init time, metadata size | Arrow cache; sharded Parquet at 100K+ |
| 10× resolution | Memory, decode time | Keep current resolution for training; tiled loading if needed |
| 3rd modality | Code changes | Add encoder + dataset; registry pattern | 
| Multi-node | Coordination, paths | DDP; shared storage; `DATA_ROOT`/`CHECKPOINT_DIR` |
| 1000 RPS inference | Batch + latency | ONNX; TorchServe; dynamic batching |

---

## References

- [Phase 4a] `docs/performance_benchmarks.md` — DataLoader benchmarks
- [Scalability] `docs/scaling_training.md` — Training cluster setup
- [Plan] Phase 4 — Performance benchmarking and scalability analysis
