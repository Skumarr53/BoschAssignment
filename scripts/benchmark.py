#!/usr/bin/env -S uv run python
"""DataLoader performance benchmark for multimodal biometric pipeline.

[Phase 4a] Parameter sweep over num_workers, batch_size, pin_memory,
prefetch_factor, persistent_workers. Measures samples/sec and init time.
[Phase 4 Part 3] Extended params, data source comparison, matplotlib charts.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py benchmark.save_charts=false
    uv run python scripts/benchmark.py benchmark.profile=true
"""

from __future__ import annotations

import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir

from biometric.data.datamodule import BiometricDataModule
from biometric.training import seed_everything
from biometric.utils.logging import get_logger

logger = get_logger(__name__)


def _consume_loader(loader: Any, num_batches: int) -> tuple[int, float]:
    """Iterate loader for num_batches, return (samples_seen, wall_seconds)."""
    start = time.perf_counter_ns()
    samples = 0
    for i, batch in enumerate(loader):
        samples += batch["label"].shape[0]
        if i >= num_batches - 1:
            break
    elapsed_ns = time.perf_counter_ns() - start
    return samples, elapsed_ns / 1e9


def _measure_init_time(
    project_root: Path,
    cfg: Any,
    use_cache: bool,
) -> float:
    """Measure DataModule setup time in seconds."""
    data_root = project_root / str(cfg.data.data_root)
    start = time.perf_counter_ns()
    dm = BiometricDataModule(
        data_root=data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        fingerprint_size=tuple(cfg.data.fingerprint_size),
        iris_size=tuple(cfg.data.iris_size),
        seed=42,
        split_by_sample=cfg.data.get("split_by_sample", False),
        cache_filename=cfg.data.get("cache_filename"),
        use_cache=use_cache,
        use_parallel_preprocess=False,
    )
    dm.setup(stage="fit")
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / 1e9


def run_benchmark(
    dm: BiometricDataModule,
    num_workers: list[int],
    batch_sizes: list[int],
    warmup_batches: int,
    measure_batches: int,
    *,
    pin_memory: bool = True,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
) -> list[dict[str, Any]]:
    """Run DataLoader sweep and return list of result dicts."""
    results: list[dict[str, Any]] = []
    for nw in num_workers:
        for bs in batch_sizes:
            loader = dm.train_dataloader(
                batch_size=bs,
                num_workers=nw,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers if nw > 0 else None,
            )
            _consume_loader(loader, warmup_batches)
            samples, seconds = _consume_loader(loader, measure_batches)
            samples_per_sec = samples / seconds if seconds > 0 else 0.0
            results.append(
                {
                    "num_workers": nw,
                    "batch_size": bs,
                    "samples": samples,
                    "seconds": round(seconds, 3),
                    "samples_per_sec": round(samples_per_sec, 1),
                }
            )
            logger.info(
                "benchmark_run",
                num_workers=nw,
                batch_size=bs,
                samples_per_sec=round(samples_per_sec, 1),
                seconds=round(seconds, 3),
            )
    return results


def run_extended_comparisons(
    dm: BiometricDataModule,
    warmup_batches: int,
    measure_batches: int,
    bench_cfg: Any,
) -> dict[str, Any]:
    """Run pin_memory, prefetch_factor, persistent_workers comparisons."""
    extended: dict[str, Any] = {}
    nw, bs = 4, 16

    if bench_cfg.get("compare_pin_memory", False):
        for pm in [True, False]:
            loader = dm.train_dataloader(batch_size=bs, num_workers=nw, pin_memory=pm)
            _consume_loader(loader, warmup_batches)
            samples, seconds = _consume_loader(loader, measure_batches)
            key = f"pin_memory={pm}"
            extended[key] = round(samples / seconds, 1) if seconds > 0 else 0.0

    if bench_cfg.get("compare_prefetch_factor", False):
        for pf in [2, 4, 8]:
            loader = dm.train_dataloader(
                batch_size=bs,
                num_workers=nw,
                prefetch_factor=pf,
            )
            _consume_loader(loader, warmup_batches)
            samples, seconds = _consume_loader(loader, measure_batches)
            extended[f"prefetch_factor={pf}"] = round(samples / seconds, 1) if seconds > 0 else 0.0

    if bench_cfg.get("compare_persistent_workers", False):
        for pw in [True, False]:
            loader = dm.train_dataloader(
                batch_size=bs,
                num_workers=nw,
                persistent_workers=pw,
            )
            _consume_loader(loader, warmup_batches)
            samples, seconds = _consume_loader(loader, measure_batches)
            extended[f"persistent_workers={pw}"] = (
                round(samples / seconds, 1) if seconds > 0 else 0.0
            )

    return extended


def _to_markdown_table(results: list[dict[str, Any]]) -> str:
    """Format results as Markdown table."""
    lines = [
        "| num_workers | batch_size | samples | seconds | samples/sec |",
        "|-------------|------------|---------|---------|-------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['num_workers']} | {r['batch_size']} | {r['samples']} | "
            f"{r['seconds']} | {r['samples_per_sec']} |"
        )
    return "\n".join(lines)


def _save_charts(
    results: list[dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    """Generate matplotlib heatmap and bar chart. Returns list of saved paths."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib_not_available", msg="Skipping charts")
        return []

    saved: list[str] = []
    nw_vals = sorted({r["num_workers"] for r in results})
    bs_vals = sorted({r["batch_size"] for r in results})
    matrix = np.zeros((len(nw_vals), len(bs_vals)))
    for r in results:
        i = nw_vals.index(r["num_workers"])
        j = bs_vals.index(r["batch_size"])
        matrix[i, j] = r["samples_per_sec"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(bs_vals)))
    ax.set_xticklabels(bs_vals)
    ax.set_yticks(range(len(nw_vals)))
    ax.set_yticklabels(nw_vals)
    ax.set_xlabel("batch_size")
    ax.set_ylabel("num_workers")
    ax.set_title("Samples/sec by num_workers × batch_size")
    plt.colorbar(im, ax=ax, label="samples/sec")
    heatmap_path = out_dir / "benchmark_heatmap.png"
    fig.savefig(heatmap_path, dpi=100, bbox_inches="tight")
    plt.close()
    saved.append(str(heatmap_path))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    labels = [f"nw={r['num_workers']}, bs={r['batch_size']}" for r in results]
    ax2.bar(range(len(results)), [r["samples_per_sec"] for r in results])
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("samples/sec")
    ax2.set_title("DataLoader Throughput")
    bar_path = out_dir / "benchmark_bar.png"
    fig2.savefig(bar_path, dpi=100, bbox_inches="tight")
    plt.close()
    saved.append(str(bar_path))

    logger.info("charts_saved", paths=saved)
    return saved


def main() -> int:
    """Run benchmark and write results to docs/performance_benchmarks.md."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="benchmark", overrides=overrides)

    seed_everything(42)

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / str(cfg.data.data_root)
    if not data_root.exists():
        logger.error("data_root_not_found", path=str(data_root))
        return 1

    bench = cfg.benchmark
    warmup_batches = int(bench.warmup_batches)
    measure_batches = int(bench.measure_batches)

    do_profile = bench.get("profile", False)
    prof_ctx: AbstractContextManager[Any]
    if do_profile:
        from biometric.utils.profiling import cprofile_context

        prof_ctx = cprofile_context(
            output_path=project_root / "docs" / "benchmark_profile.txt",
        )
    else:
        from contextlib import nullcontext

        prof_ctx = nullcontext()

    with prof_ctx:
        dm = BiometricDataModule(
            data_root=data_root,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            fingerprint_size=tuple(cfg.data.fingerprint_size),
            iris_size=tuple(cfg.data.iris_size),
            seed=42,
            split_by_sample=cfg.data.get("split_by_sample", False),
            cache_filename=cfg.data.get("cache_filename"),
            use_cache=True,
            use_parallel_preprocess=False,
        )
        dm.setup(stage="fit")

        results = run_benchmark(
            dm=dm,
            num_workers=list(bench.num_workers),
            batch_sizes=list(bench.batch_sizes),
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
        )

        extended = run_extended_comparisons(dm, warmup_batches, measure_batches, bench)

    init_times: dict[str, float] = {}
    if bench.get("compare_data_source", False):
        for use_cache in [True, False]:
            t = _measure_init_time(project_root, cfg, use_cache=use_cache)
            init_times[f"use_cache={use_cache}"] = round(t, 3)

    out_dir = project_root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_refs: list[str] = []
    if bench.get("save_charts", True):
        _save_charts(results, out_dir)
        chart_refs = [
            "![heatmap](benchmark_heatmap.png)",
            "![bar](benchmark_bar.png)",
        ]

    ext_md = ""
    if extended:
        ext_md = "\n## Extended Comparisons (nw=4, bs=16)\n\n"
        ext_md += "| Config | samples/sec |\n|--------|-------------|\n"
        for k, v in extended.items():
            ext_md += f"| {k} | {v} |\n"

    init_md = ""
    if init_times:
        init_md = "\n## Init Time: Arrow Cache vs Filesystem\n\n"
        init_md += "| Data source | init_seconds |\n|-------------|-------------|\n"
        for k, v in init_times.items():
            init_md += f"| {k} | {v} |\n"

    charts_md = "\n".join(chart_refs) + "\n" if chart_refs else ""

    content = f"""# Performance Benchmarks

[Phase 4] DataLoader parameter sweep for multimodal biometric pipeline.

## Configuration

- **Warmup batches**: {warmup_batches}
- **Measure batches**: {measure_batches}
- **Data source**: Arrow cache (when available) + filesystem

## Results

{_to_markdown_table(results)}
{ext_md}
{init_md}

## Charts

{charts_md}

## Notes

- `num_workers=0` uses main-process loading (no worker processes).
- Higher `num_workers` typically improves throughput until I/O or CPU saturates.
- `batch_size` affects GPU utilization; larger batches reduce per-sample overhead.
- Arrow cache reduces init time vs filesystem discovery.
"""
    out_path = out_dir / "performance_benchmarks.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("benchmark_complete", output_path=str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
