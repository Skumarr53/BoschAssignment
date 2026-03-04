"""PyTorch Dataset classes for multimodal biometric data.

Provides FingerprintDataset, IrisDataset, MultimodalBiometricDataset,
and PreloadedMultimodalDataset (parallel-preprocessed, for training integration).
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from biometric.utils.discovery import discover_subjects, validate_path
from biometric.utils.logging import get_logger
from biometric.utils.parser import parse_fingerprint_filename, parse_iris_path
from biometric.utils.types import FingerprintSample, IrisSample

logger = get_logger(__name__)

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


def _load_image(path: Path, *, mode: str = "RGB") -> "Image.Image":
    """Load image from path using PIL.

    Args:
        path: Path to image file.
        mode: PIL mode — "L" for grayscale, "RGB" for color.
    """
    if Image is None:
        msg = "PIL (Pillow) is required for image loading. Install with: pip install pillow"
        raise ImportError(msg)
    img = Image.open(path)
    return img.convert(mode)


class FingerprintDataset(Dataset[tuple[Any, int]]):
    """Single-modality dataset: fingerprint images only.

    Each subject has 10 fingerprint images (5 left hand + 5 right hand).
    """

    def __init__(
        self,
        root: Path | str,
        subject_ids: list[int] | None = None,
        subject_to_label: dict[int, int] | None = None,
        transform: Any = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._transform = transform
        self._samples: list[FingerprintSample] = []
        subject_ids = subject_ids or discover_subjects(self._root)
        self._build_index(subject_ids, subject_to_label)
        logger.info(
            "fingerprint_dataset_built",
            subject_count=len(subject_ids),
            sample_count=len(self._samples),
            root=str(self._root),
        )

    @property
    def samples(self) -> list[FingerprintSample]:
        """Public read-only access to sample metadata."""
        return list(self._samples)

    def _build_index(
        self,
        subject_ids: list[int],
        subject_to_label: dict[int, int] | None = None,
    ) -> None:
        if subject_to_label is None:
            subject_to_label = {sid: i for i, sid in enumerate(sorted(subject_ids))}
        for sid in sorted(subject_ids):
            label = subject_to_label[sid]
            fp_dir = self._root / str(sid) / "Fingerprint"
            if not fp_dir.exists():
                continue
            for path in sorted(fp_dir.glob("*.BMP")):
                try:
                    validated = validate_path(path, self._root)
                    sample = parse_fingerprint_filename(validated, label=label)
                    self._samples.append(sample)
                except ValueError as e:
                    logger.warning(
                        "parse_skipped",
                        path=str(path),
                        reason=str(e),
                        subject_id=sid,
                    )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        sample = self._samples[idx]
        img = _load_image(sample.file_path, mode="L")
        if self._transform is not None:
            img = self._transform(img)
        return img, sample.label


class IrisDataset(Dataset[tuple[Any, int]]):
    """Single-modality dataset: iris images only.

    Each subject has 10 iris images (5 left eye + 5 right eye).
    """

    def __init__(
        self,
        root: Path | str,
        subject_ids: list[int] | None = None,
        subject_to_label: dict[int, int] | None = None,
        transform: Any = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._transform = transform
        self._samples: list[IrisSample] = []
        subject_ids = subject_ids or discover_subjects(self._root)
        self._build_index(subject_ids, subject_to_label)
        logger.info(
            "iris_dataset_built",
            subject_count=len(subject_ids),
            sample_count=len(self._samples),
            root=str(self._root),
        )

    @property
    def samples(self) -> list[IrisSample]:
        """Public read-only access to sample metadata."""
        return list(self._samples)

    def _build_index(
        self,
        subject_ids: list[int],
        subject_to_label: dict[int, int] | None = None,
    ) -> None:
        if subject_to_label is None:
            subject_to_label = {sid: i for i, sid in enumerate(sorted(subject_ids))}
        for sid in sorted(subject_ids):
            label = subject_to_label[sid]
            for is_left, subdir in [(True, "left"), (False, "right")]:
                sub_path = self._root / str(sid) / subdir
                if not sub_path.exists():
                    continue
                for path in sorted(sub_path.glob("*.bmp")):
                    try:
                        validated = validate_path(path, self._root)
                        sample = parse_iris_path(validated, sid, is_left=is_left, label=label)
                        self._samples.append(sample)
                    except ValueError as e:
                        logger.warning(
                            "parse_skipped",
                            path=str(path),
                            reason=str(e),
                            subject_id=sid,
                        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        sample = self._samples[idx]
        img = _load_image(sample.file_path, mode="RGB")
        if self._transform is not None:
            img = self._transform(img)
        return img, sample.label


class MultimodalBiometricDataset(Dataset[dict[str, Any]]):
    """Fused dataset: pairs fingerprint + iris samples per subject.

    Pairing strategy: for each subject, fingerprint samples are sorted by
    (hand, finger_type) and iris samples by (modality, sequence). The i-th
    fingerprint sample is paired with the i-th iris sample per subject, up to
    min(num_fp, num_iris). This ensures deterministic, semantically consistent
    pairing across runs.
    """

    def __init__(
        self,
        root: Path | str,
        subject_ids: list[int] | None = None,
        subject_to_label: dict[int, int] | None = None,
        fingerprint_transform: Any = None,
        iris_transform: Any = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._fp_transform = fingerprint_transform
        self._iris_transform = iris_transform
        self._fp_dataset = FingerprintDataset(
            self._root,
            subject_ids=subject_ids,
            subject_to_label=subject_to_label,
            transform=None,
        )
        self._iris_dataset = IrisDataset(
            self._root,
            subject_ids=subject_ids,
            subject_to_label=subject_to_label,
            transform=None,
        )
        self._pairs: list[tuple[int, int]] = []
        self._build_pairs()
        logger.info(
            "multimodal_dataset_built",
            pair_count=len(self._pairs),
            root=str(self._root),
        )

    def _build_pairs(self) -> None:
        """Build (fp_idx, iris_idx) pairs aligned by subject + sorted metadata.

        Fingerprints sorted by (hand, finger_type), iris by (modality, sequence).
        This gives a deterministic, semantically meaningful alignment.
        """
        fp_by_subject: dict[int, list[tuple[str, str, int]]] = {}
        iris_by_subject: dict[int, list[tuple[str, int, int]]] = {}

        for i, fp_sample in enumerate(self._fp_dataset._samples):
            fp_by_subject.setdefault(fp_sample.subject_id, []).append(
                (fp_sample.hand.value, fp_sample.finger_type.value, i)
            )
        for i, iris_sample in enumerate(self._iris_dataset._samples):
            iris_by_subject.setdefault(iris_sample.subject_id, []).append(
                (iris_sample.modality.value, iris_sample.sequence, i)
            )

        for sid in sorted(fp_by_subject.keys()):
            fp_sorted = sorted(fp_by_subject[sid], key=lambda x: (x[0], x[1]))
            iris_sorted = sorted(iris_by_subject.get(sid, []), key=lambda x: (x[0], x[1]))
            n = min(len(fp_sorted), len(iris_sorted))
            for j in range(n):
                self._pairs.append((fp_sorted[j][2], iris_sorted[j][2]))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fp_idx, iris_idx = self._pairs[idx]
        fp_img, _ = self._fp_dataset[fp_idx]
        iris_img, label = self._iris_dataset[iris_idx]
        if self._fp_transform is not None:
            fp_img = self._fp_transform(fp_img)
        if self._iris_transform is not None:
            iris_img = self._iris_transform(iris_img)
        return {
            "fingerprint": fp_img,
            "iris": iris_img,
            "label": label,
        }


class PreloadedMultimodalDataset(Dataset[dict[str, Any]]):
    """Multimodal dataset backed by preprocessed tensors from parallel_loader.

    [Phase 1d] Used when use_parallel_preprocess=True to integrate Ray/multiprocessing
    into the training pipeline. Base transforms applied at preprocessing; optional
    runtime transforms (e.g. augmentation) applied in __getitem__.
    """

    def __init__(
        self,
        fp_preprocessed: list[tuple[torch.Tensor, int]],
        iris_preprocessed: list[tuple[torch.Tensor, int]],
        pairs: list[tuple[int, int]],
        *,
        fingerprint_transform: Any = None,
        iris_transform: Any = None,
    ) -> None:
        self._fp = fp_preprocessed
        self._iris = iris_preprocessed
        self._pairs = pairs
        self._fp_transform = fingerprint_transform
        self._iris_transform = iris_transform
        logger.info(
            "preloaded_multimodal_dataset_built",
            pair_count=len(pairs),
            fp_count=len(fp_preprocessed),
            iris_count=len(iris_preprocessed),
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fp_idx, iris_idx = self._pairs[idx]
        fp_tensor, _ = self._fp[fp_idx]
        iris_tensor, label = self._iris[iris_idx]
        if self._fp_transform is not None:
            fp_tensor = self._fp_transform(fp_tensor)
        if self._iris_transform is not None:
            iris_tensor = self._iris_transform(iris_tensor)
        return {
            "fingerprint": fp_tensor,
            "iris": iris_tensor,
            "label": label,
        }
