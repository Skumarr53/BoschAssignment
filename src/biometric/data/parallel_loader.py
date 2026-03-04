"""Parallel preprocessing for biometric images.

Config-driven toggle between backends (default: Ray):
- Ray Data: fault-tolerant, scales to cluster (requires ray[data] extra).
- ProcessPoolExecutor: stdlib, zero deps, sufficient for single-node.

Use preprocess_with_backend() or preprocess_from_config() with data.backend.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pyarrow as pa
import torch

from biometric.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for optional Ray
_ray_data = None


def _get_ray_data() -> Any:
    """Lazy import of ray.data; returns None if not installed."""
    global _ray_data
    if _ray_data is None:
        try:
            import ray.data as rd  # noqa: F401  # type: ignore[import-not-found]

            _ray_data = rd
        except ImportError:
            pass
    return _ray_data


def _build_transform_from_config(modality: str, config: dict[str, Any]) -> Any:
    """Build transform from config dict (used in worker processes)."""
    from biometric.data.preprocessing import get_fingerprint_transform, get_iris_transform

    size = tuple(config.get("size", (96, 96) if modality == "fingerprint" else (224, 224)))
    train = config.get("train", False)
    if modality == "fingerprint":
        return get_fingerprint_transform(size=size, train=train)
    return get_iris_transform(size=size, train=train)


def _load_and_preprocess(
    path: str,
    label: int,
    modality: str,
    transform_config: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Load image, apply transform, return (tensor as numpy, label).

    Top-level function for pickling in ProcessPoolExecutor.
    """
    from PIL import Image

    transform = _build_transform_from_config(modality, transform_config)
    mode = "L" if modality == "fingerprint" else "RGB"
    img = Image.open(path).convert(mode)
    tensor = transform(img)
    arr = tensor.numpy()
    return arr, label


def _preprocess_ray_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ray map function: preprocess a single row."""
    path = row["path"]
    label = int(row["label"])
    modality = row.get("modality", "fingerprint")
    config = row.get("transform_config") or {}
    try:
        arr, lab = _load_and_preprocess(path, label, modality, config)
        return {"data": arr, "label": lab}
    except Exception as e:
        logger.warning("ray_preprocess_skipped", path=path, error=str(e))
        return {"data": None, "label": None}


def preprocess_with_pool(
    paths: list[str],
    labels: list[int],
    modality: str,
    transform_config: dict[str, Any] | None = None,
    max_workers: int | None = None,
    *,
    ordered: bool = False,
) -> list[tuple[torch.Tensor, int]]:
    """Preprocess images in parallel using ProcessPoolExecutor.

    Args:
        paths: List of image file paths.
        labels: List of labels (same length as paths).
        modality: "fingerprint" or "iris" (determines transform).
        transform_config: Optional dict with size, train. Defaults from preprocessing.
        max_workers: Max parallel workers. Defaults to os.cpu_count() - 1.
        ordered: If True, output order matches input order (for dataset alignment).

    Returns:
        List of (tensor, label) tuples. Order matches input when ordered=True.
    """
    if len(paths) != len(labels):
        msg = f"paths and labels length mismatch: {len(paths)} vs {len(labels)}"
        raise ValueError(msg)
    if not paths:
        return []
    config = transform_config or {}
    workers = max_workers or max(1, (os.cpu_count() or 4) - 1)

    results: list[tuple[torch.Tensor, int]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_load_and_preprocess, p, lab, modality, config)
            for p, lab in zip(paths, labels, strict=True)
        ]
        if ordered:
            for i, future in enumerate(futures):
                try:
                    arr, label = future.result()
                    results.append((torch.from_numpy(arr), label))
                except Exception as e:
                    logger.warning("pool_preprocess_failed", path=paths[i], error=str(e))
        else:
            future_to_path = {f: paths[i] for i, f in enumerate(futures)}
            for future in as_completed(future_to_path):
                try:
                    arr, label = future.result()
                    results.append((torch.from_numpy(arr), label))
                except Exception as e:
                    logger.warning(
                        "pool_preprocess_failed",
                        path=future_to_path[future],
                        error=str(e),
                    )

    logger.info(
        "pool_preprocess_done",
        input_count=len(paths),
        output_count=len(results),
        max_workers=workers,
        ordered=ordered,
    )
    return results


def preprocess_with_backend(
    paths: list[str],
    labels: list[int],
    modality: str,
    transform_config: dict[str, Any] | None = None,
    backend: str = "ray",
    *,
    max_workers: int | None = None,
    num_cpus_per_task: float = 1.0,
    ordered: bool = False,
) -> list[tuple[torch.Tensor, int]]:
    """Preprocess images in parallel using configurable backend.

    Dispatches to Ray Data (default) or ProcessPoolExecutor based on backend.

    Args:
        paths: List of image file paths.
        labels: List of labels (same length as paths).
        modality: "fingerprint" or "iris".
        transform_config: Optional dict with size, train.
        backend: "ray" (default) or "multiprocessing" / "pool".
        max_workers: For multiprocessing only. Defaults to cpu_count - 1.
        num_cpus_per_task: For Ray only. CPUs per Ray task.

    Returns:
        List of (tensor, label) tuples.

    Raises:
        ImportError: If backend is "ray" and Ray is not installed.
        ValueError: If backend is unknown.
    """
    backend_lower = backend.lower().strip()
    if backend_lower in ("ray",):
        return preprocess_with_ray(
            paths=paths,
            labels=labels,
            modality=modality,
            transform_config=transform_config,
            num_cpus_per_task=num_cpus_per_task,
        )
    if backend_lower in ("multiprocessing", "pool", "processpool"):
        return preprocess_with_pool(
            paths=paths,
            labels=labels,
            modality=modality,
            transform_config=transform_config,
            max_workers=max_workers,
            ordered=ordered,
        )
    msg = f"Unknown parallel backend: {backend}. Use 'ray' or 'multiprocessing'."
    raise ValueError(msg)


def preprocess_from_config(
    paths: list[str],
    labels: list[int],
    modality: str,
    config: dict[str, Any],
    *,
    ordered: bool = False,
) -> list[tuple[torch.Tensor, int]]:
    """Preprocess images using parallel config dict (e.g. from Hydra data.parallel).

    Reads backend, max_workers, num_cpus_per_task, transform_config from config.

    Args:
        paths: List of image file paths.
        labels: List of labels.
        modality: "fingerprint" or "iris".
        config: Dict with keys backend, max_workers, num_cpus_per_task, transform_config.
        ordered: If True, output order matches input (for dataset alignment).

    Returns:
        List of (tensor, label) tuples.
    """
    backend = config.get("backend", "ray")
    transform_config = config.get("transform_config") or {}
    return preprocess_with_backend(
        paths=paths,
        labels=labels,
        modality=modality,
        transform_config=transform_config,
        backend=backend,
        max_workers=config.get("max_workers"),
        num_cpus_per_task=float(config.get("num_cpus_per_task", 1.0)),
        ordered=ordered,
    )


def preprocess_with_ray(
    paths: list[str],
    labels: list[int],
    modality: str,
    transform_config: dict[str, Any] | None = None,
    num_cpus_per_task: float = 1.0,
) -> list[tuple[torch.Tensor, int]]:
    """Preprocess images using Ray Data (requires ray[data] extra).

    Args:
        paths: List of image file paths.
        labels: List of labels.
        modality: "fingerprint" or "iris".
        transform_config: Optional transform config.
        num_cpus_per_task: CPUs per Ray task.

    Returns:
        List of (tensor, label) tuples.

    Raises:
        ImportError: If ray is not installed.
    """
    rd = _get_ray_data()
    if rd is None:
        msg = "Ray Data not installed. Install with: uv sync --extra ray"
        raise ImportError(msg)

    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    config = transform_config or {}
    items = [
        {"path": p, "label": lab, "modality": modality, "transform_config": config}
        for p, lab in zip(paths, labels, strict=True)
    ]
    ds = rd.from_items(items)
    ds = ds.map(_preprocess_ray_row, num_cpus=num_cpus_per_task)
    results: list[tuple[torch.Tensor, int]] = []
    for row in ds.iter_rows():
        data = row.get("data")
        lab = row.get("label")
        if data is not None and lab is not None and isinstance(data, np.ndarray):
            results.append((torch.from_numpy(data), int(lab)))
    logger.info("ray_preprocess_done", input_count=len(paths), output_count=len(results))
    return results


def get_preprocessed_items_from_cache(
    cache_table: pa.Table,
    modality: str,
    subject_ids: list[int] | None = None,
) -> tuple[list[str], list[int]]:
    """Extract (paths, labels) from Arrow cache for a modality.

    Args:
        cache_table: PyArrow table from load_cache or build_cache.
        modality: "fingerprint", "iris_left", or "iris_right".
        subject_ids: Optional filter by subject IDs.

    Returns:
        (paths, labels) for preprocessing.
    """
    import pyarrow.compute as pc

    mask = pc.equal(cache_table.column("modality"), modality)
    filtered = cache_table.filter(mask)
    if subject_ids:
        subj_arr = pa.array(subject_ids)
        mask2 = pc.is_in(filtered.column("subject_id"), subj_arr)
        filtered = filtered.filter(mask2)
    paths = [str(x) for x in filtered.column("filepath")]
    labels = [int(x) for x in filtered.column("label")]
    return paths, labels
