"""Profiling utilities for performance analysis.

[Phase 4] Wrappers for torch.profiler, cProfile, and tracemalloc.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import tracemalloc
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from biometric.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def cprofile_context(
    output_path: Path | str | None = None,
    sort_by: str = "cumulative",
    top_n: int = 20,
) -> Generator[cProfile.Profile, None, None]:
    """Context manager for cProfile with optional output to file.

    Args:
        output_path: If set, write stats to file. Default: None (no file).
        sort_by: Sort key for stats (cumulative, time, calls).
        top_n: Number of top functions to log.

    Yields:
        cProfile.Profile instance.
    """
    prof = cProfile.Profile()
    prof.enable()
    try:
        yield prof
    finally:
        prof.disable()
        stream = io.StringIO()
        ps = pstats.Stats(prof, stream=stream).sort_stats(sort_by)
        ps.print_stats(top_n)
        stats_str = stream.getvalue()
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(stats_str, encoding="utf-8")
            logger.info("cprofile_output", path=str(path))
        logger.debug("cprofile_top", stats=stats_str[:1000])


def run_cprofile(
    func: Any,
    *args: Any,
    output_path: Path | str | None = None,
    **kwargs: Any,
) -> Any:
    """Run a function under cProfile and return its result.

    Args:
        func: Callable to profile.
        *args: Positional args for func.
        output_path: Optional path to write stats.
        **kwargs: Keyword args for func.

    Returns:
        Result of func(*args, **kwargs).
    """
    with cprofile_context(output_path=output_path):
        return func(*args, **kwargs)


@contextmanager
def tracemalloc_context(
    output_path: Path | str | None = None,
    n_frames: int = 10,
) -> Generator[None, None, None]:
    """Context manager for memory tracking via tracemalloc.

    Args:
        output_path: If set, write top memory allocations to file.
        n_frames: Number of frames in traceback for allocations.

    Yields:
        None. Use inside the block to profile memory.
    """
    tracemalloc.start(n_frames)
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot.statistics("lineno")
        lines: list[str] = [
            f"Peak memory: {peak / 1024 / 1024:.2f} MiB",
            f"Current memory: {current / 1024 / 1024:.2f} MiB",
            "",
            "Top 10 allocations:",
        ]
        for stat in top_stats[:10]:
            tb_str = "".join(stat.traceback.format())
            lines.append(f"  {stat.size / 1024:.1f} KiB\n{tb_str}")

        report = "\n".join(lines)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            logger.info("tracemalloc_output", path=str(path), peak_mib=peak / 1024 / 1024)
        else:
            logger.info("tracemalloc_peak_mib", peak_mib=peak / 1024 / 1024)


def get_torch_profiler(
    *,
    activities: list[Any] | None = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
) -> Any:
    """Return a torch.profiler.profile context manager.

    Args:
        activities: Profiler activities. Default: CPU, CUDA if available.
        record_shapes: Record tensor shapes.
        profile_memory: Profile memory allocation.
        with_stack: Record Python stack traces.

    Returns:
        torch.profiler.profile context manager.
    """
    import torch

    if activities is None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )
