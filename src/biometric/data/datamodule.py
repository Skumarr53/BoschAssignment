"""DataModule orchestrator for multimodal biometric training.

Wires datasets, transforms, and DataLoaders from config.
Supports train/val/test splits by subject ID.
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from biometric.data.arrow_cache import (
    build_cache,
    cache_exists,
    get_cache_path,
    is_cache_stale,
    load_cache,
)
from biometric.data.dataset import MultimodalBiometricDataset, PreloadedMultimodalDataset
from biometric.data.parallel_loader import preprocess_from_config
from biometric.data.preprocessing import get_multimodal_transforms
from biometric.utils.discovery import discover_subjects
from biometric.utils.logging import get_logger

logger = get_logger(__name__)


def _split_subject_ids(
    subject_ids: list[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split subject IDs into train/val/test with deterministic shuffle."""
    gen = torch.Generator().manual_seed(seed)
    n = len(subject_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    perm = torch.randperm(n, generator=gen).tolist()
    shuffled = [subject_ids[i] for i in perm]
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]
    return train_ids, val_ids, test_ids


def _split_by_sample(
    subject_ids: list[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Use all subjects for train and val; split is by sample (for closed-set val).

    Returns (train_ids, val_ids, test_ids) where train_ids == val_ids == subject_ids
    for the fit stage. The actual split is done in the dataset via pair indices.
    For test we still hold out a subject fraction to simulate open-set evaluation.
    """
    gen = torch.Generator().manual_seed(seed)
    n = len(subject_ids)
    n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
    perm = torch.randperm(n, generator=gen).tolist()
    shuffled = [subject_ids[i] for i in perm]
    fit_ids = shuffled[:-n_test]  # train+val share these subjects
    test_ids = shuffled[-n_test:]
    return fit_ids, fit_ids, test_ids


class BiometricDataModule:
    """Orchestrates datasets and DataLoaders for multimodal biometric training.

    Builds MultimodalBiometricDataset with train/val/test splits.
    Optionally uses Arrow cache for faster dataset init.
    """

    def __init__(
        self,
        data_root: Path | str,
        batch_size: int = 16,
        num_workers: int = 4,
        fingerprint_size: tuple[int, int] = (96, 96),
        iris_size: tuple[int, int] = (224, 224),
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        split_by_sample: bool = False,
        cache_filename: str | None = "biometric_metadata.parquet",
        use_cache: bool = True,
        use_parallel_preprocess: bool = False,
        parallel_config: dict[str, Any] | None = None,
    ) -> None:
        self._root = Path(data_root).resolve()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._fingerprint_size = (int(fingerprint_size[0]), int(fingerprint_size[1]))
        self._iris_size = (int(iris_size[0]), int(iris_size[1]))
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        self._split_by_sample = split_by_sample
        self._cache_filename = cache_filename
        self._use_cache = use_cache
        self._use_parallel_preprocess = use_parallel_preprocess
        self._parallel_config = parallel_config or {}

        self._train_dataset: Dataset[dict[str, Any]] | None = None
        self._val_dataset: Dataset[dict[str, Any]] | None = None
        self._test_dataset: Dataset[dict[str, Any]] | None = None

    def setup(self, stage: str | None = None) -> None:
        """Build datasets and optionally discover subjects from cache."""
        subject_ids = self._discover_subjects()
        if not subject_ids:
            msg = f"No subjects found in {self._root}"
            raise FileNotFoundError(msg)

        split_fn = _split_by_sample if self._split_by_sample else _split_subject_ids
        train_ids, val_ids, test_ids = split_fn(
            subject_ids,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
            seed=self._seed,
        )

        # Global label mapping: all subjects 0..(n-1) for consistent train/val/test
        subject_to_label = {sid: i for i, sid in enumerate(sorted(subject_ids))}

        fp_transform, iris_transform = get_multimodal_transforms(
            fingerprint_size=self._fingerprint_size,
            iris_size=self._iris_size,
            train=False,
        )
        fp_transform_train, iris_transform_train = get_multimodal_transforms(
            fingerprint_size=self._fingerprint_size,
            iris_size=self._iris_size,
            train=True,
        )

        if self._use_parallel_preprocess:
            self._setup_with_parallel_preprocess(
                stage=stage,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                subject_to_label=subject_to_label,
                fp_transform=fp_transform,
                iris_transform=iris_transform,
                fp_transform_train=fp_transform_train,
                iris_transform_train=iris_transform_train,
            )
        else:
            if stage in (None, "fit"):
                if self._split_by_sample and train_ids == val_ids:
                    # Same subjects: build full dataset, split by pair indices
                    full_ds = MultimodalBiometricDataset(
                        self._root,
                        subject_ids=train_ids,
                        subject_to_label=subject_to_label,
                        fingerprint_transform=fp_transform_train,
                        iris_transform=iris_transform_train,
                    )
                    n = len(full_ds)
                    gen = torch.Generator().manual_seed(self._seed)
                    perm = torch.randperm(n, generator=gen).tolist()
                    n_train = int(n * self._train_ratio)
                    train_idx = perm[:n_train]
                    val_idx = perm[n_train:]
                    val_ds_full = MultimodalBiometricDataset(
                        self._root,
                        subject_ids=val_ids,
                        subject_to_label=subject_to_label,
                        fingerprint_transform=fp_transform,
                        iris_transform=iris_transform,
                    )
                    self._train_dataset = Subset(full_ds, train_idx)
                    self._val_dataset = Subset(val_ds_full, val_idx)
                else:
                    self._train_dataset = MultimodalBiometricDataset(
                        self._root,
                        subject_ids=train_ids,
                        subject_to_label=subject_to_label,
                        fingerprint_transform=fp_transform_train,
                        iris_transform=iris_transform_train,
                    )
                    self._val_dataset = MultimodalBiometricDataset(
                        self._root,
                        subject_ids=val_ids,
                        subject_to_label=subject_to_label,
                        fingerprint_transform=fp_transform,
                        iris_transform=iris_transform,
                    )
            if stage in (None, "test"):
                self._test_dataset = MultimodalBiometricDataset(
                    self._root,
                    subject_ids=test_ids,
                    subject_to_label=subject_to_label,
                    fingerprint_transform=fp_transform,
                    iris_transform=iris_transform,
                )

        logger.info(
            "datamodule_setup",
            train_subjects=len(train_ids),
            val_subjects=len(val_ids),
            test_subjects=len(test_ids),
            stage=stage or "all",
        )

    def _setup_with_parallel_preprocess(
        self,
        *,
        stage: str | None,
        train_ids: list[int],
        val_ids: list[int],
        test_ids: list[int],
        subject_to_label: dict[int, int],
        fp_transform: Any,
        iris_transform: Any,
        fp_transform_train: Any,
        iris_transform_train: Any,
    ) -> None:
        """[Phase 1d] Build datasets using Ray/multiprocessing parallel preprocessing."""
        cfg = self._parallel_config
        fp_cfg = {
            **cfg,
            "transform_config": {
                **(cfg.get("transform_config") or {}),
                "size": list(self._fingerprint_size),
            },
        }
        iris_cfg = {
            **cfg,
            "transform_config": {
                **(cfg.get("transform_config") or {}),
                "size": list(self._iris_size),
            },
        }

        def _preload_for_subjects(
            ids: list[int],
        ) -> tuple[
            list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]], list[tuple[int, int]]
        ]:
            ref = MultimodalBiometricDataset(
                self._root,
                subject_ids=ids,
                subject_to_label=subject_to_label,
                fingerprint_transform=None,
                iris_transform=None,
            )
            paths_fp = [str(s.file_path) for s in ref._fp_dataset._samples]
            labels_fp = [s.label for s in ref._fp_dataset._samples]
            paths_iris = [str(s.file_path) for s in ref._iris_dataset._samples]
            labels_iris = [s.label for s in ref._iris_dataset._samples]
            fp_pre = preprocess_from_config(
                paths_fp, labels_fp, "fingerprint", fp_cfg, ordered=True
            )
            iris_pre = preprocess_from_config(
                paths_iris, labels_iris, "iris", iris_cfg, ordered=True
            )
            return fp_pre, iris_pre, ref._pairs

        if stage in (None, "fit"):
            if self._split_by_sample and train_ids == val_ids:
                fp_pre, iris_pre, pairs = _preload_for_subjects(train_ids)
                preloaded_train = PreloadedMultimodalDataset(
                    fp_pre,
                    iris_pre,
                    pairs,
                    fingerprint_transform=fp_transform_train,
                    iris_transform=iris_transform_train,
                )
                preloaded_val = PreloadedMultimodalDataset(
                    fp_pre,
                    iris_pre,
                    pairs,
                    fingerprint_transform=fp_transform,
                    iris_transform=iris_transform,
                )
                n = len(preloaded_train)
                gen = torch.Generator().manual_seed(self._seed)
                perm = torch.randperm(n, generator=gen).tolist()
                n_train = int(n * self._train_ratio)
                self._train_dataset = Subset(preloaded_train, perm[:n_train])
                self._val_dataset = Subset(preloaded_val, perm[n_train:])
            else:
                fp_pre_t, iris_pre_t, pairs_t = _preload_for_subjects(train_ids)
                fp_pre_v, iris_pre_v, pairs_v = _preload_for_subjects(val_ids)
                self._train_dataset = PreloadedMultimodalDataset(
                    fp_pre_t,
                    iris_pre_t,
                    pairs_t,
                    fingerprint_transform=fp_transform_train,
                    iris_transform=iris_transform_train,
                )
                self._val_dataset = PreloadedMultimodalDataset(
                    fp_pre_v,
                    iris_pre_v,
                    pairs_v,
                    fingerprint_transform=fp_transform,
                    iris_transform=iris_transform,
                )
        if stage in (None, "test"):
            fp_pre, iris_pre, pairs = _preload_for_subjects(test_ids)
            self._test_dataset = PreloadedMultimodalDataset(
                fp_pre,
                iris_pre,
                pairs,
                fingerprint_transform=fp_transform,
                iris_transform=iris_transform,
            )

        logger.info(
            "datamodule_parallel_preprocess_done",
            backend=cfg.get("backend", "ray"),
            train_subjects=len(train_ids),
            val_subjects=len(val_ids),
            test_subjects=len(test_ids),
        )

    def _discover_subjects(self) -> list[int]:
        """Discover subject IDs, optionally from Arrow cache with staleness check."""
        if self._use_cache and self._cache_filename:
            cache_path = get_cache_path(self._root, self._cache_filename)
            if cache_exists(cache_path) and not is_cache_stale(cache_path, self._root):
                table = load_cache(cache_path)
                subject_col = table.column("subject_id")
                ids = sorted({int(x) for x in subject_col})
                logger.info("datamodule_subjects_from_cache", count=len(ids))
                return ids
            if cache_exists(cache_path):
                logger.info("cache_stale_rebuilding", cache_path=str(cache_path))
            table = build_cache(self._root, cache_path=cache_path)
            subject_col = table.column("subject_id")
            ids = sorted({int(x) for x in subject_col})
            return ids
        return discover_subjects(self._root)

    def _ensure_setup(self, stage: str) -> None:
        """Ensure setup() has been called for the given stage."""
        if stage == "fit" and (self._train_dataset is None or self._val_dataset is None):
            self.setup(stage="fit")
        elif stage == "test" and self._test_dataset is None:
            self.setup(stage="test")

    def train_dataloader(
        self,
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> DataLoader[Any]:
        """Return training DataLoader with optional parameter overrides.

        [Phase 4] Overrides enable DataLoader benchmarking without re-instantiating
        the DataModule. When overrides are None, uses instance defaults.
        [Scalability] When rank and world_size are provided, uses DistributedSampler.
        """
        self._ensure_setup("fit")
        if self._train_dataset is None:
            msg = "train_dataset is None after setup — no training subjects?"
            raise RuntimeError(msg)
        bs = batch_size if batch_size is not None else self._batch_size
        nw = num_workers if num_workers is not None else self._num_workers
        pm = pin_memory if pin_memory is not None else True

        use_distributed = rank is not None and world_size is not None and world_size > 1
        if use_distributed:
            sampler: DistributedSampler[Any] = DistributedSampler(
                self._train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            kwargs = {
                "batch_size": bs,
                "sampler": sampler,
                "num_workers": nw,
                "pin_memory": pm,
            }
        else:
            kwargs = {
                "batch_size": bs,
                "shuffle": True,
                "num_workers": nw,
                "pin_memory": pm,
            }

        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        if persistent_workers is not None and nw > 0:
            kwargs["persistent_workers"] = persistent_workers
        return DataLoader(self._train_dataset, **kwargs)  # type: ignore[arg-type]

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation DataLoader."""
        self._ensure_setup("fit")
        if self._val_dataset is None:
            msg = "val_dataset is None after setup — no validation subjects?"
            raise RuntimeError(msg)
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return test DataLoader."""
        self._ensure_setup("test")
        if self._test_dataset is None:
            msg = "test_dataset is None after setup — no test subjects?"
            raise RuntimeError(msg)
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
