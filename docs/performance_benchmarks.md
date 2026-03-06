# Performance Benchmarks

[Phase 4] DataLoader parameter sweep for multimodal biometric pipeline.

## Configuration

- **Warmup batches**: 5
- **Measure batches**: 50
- **Data source**: Arrow cache (when available) + filesystem

## Results

| num_workers | batch_size | samples | seconds | samples/sec |
|-------------|------------|---------|---------|-------------|
| 0 | 8 | 328 | 0.256 | 1283.4 |
| 0 | 16 | 328 | 0.243 | 1348.7 |
| 0 | 32 | 328 | 0.254 | 1293.2 |
| 2 | 8 | 328 | 0.193 | 1700.1 |
| 2 | 16 | 328 | 0.184 | 1781.1 |
| 2 | 32 | 328 | 0.188 | 1746.0 |
| 4 | 8 | 328 | 0.119 | 2746.5 |
| 4 | 16 | 328 | 0.123 | 2668.0 |
| 4 | 32 | 328 | 0.134 | 2453.6 |
| 8 | 8 | 328 | 0.111 | 2947.3 |
| 8 | 16 | 328 | 0.101 | 3240.3 |
| 8 | 32 | 328 | 0.118 | 2790.4 |

## Extended Comparisons (nw=4, bs=16)

| Config | samples/sec |
|--------|-------------|
| pin_memory=True | 2771.2 |
| pin_memory=False | 2870.2 |
| prefetch_factor=2 | 2769.0 |
| prefetch_factor=4 | 2643.1 |
| prefetch_factor=8 | 2645.7 |
| persistent_workers=True | 2529.9 |
| persistent_workers=False | 2711.6 |


## Init Time: Arrow Cache vs Filesystem

| Data source | init_seconds |
|-------------|-------------|
| use_cache=True | 0.058 |
| use_cache=False | 0.047 |


## Charts

Run `uv run python scripts/benchmark.py` to generate `benchmark_heatmap.png` and `benchmark_bar.png` in the output directory.


## Notes

- `num_workers=0` uses main-process loading (no worker processes).
- Higher `num_workers` typically improves throughput until I/O or CPU saturates.
- `batch_size` affects GPU utilization; larger batches reduce per-sample overhead.
- Arrow cache reduces init time vs filesystem discovery.
