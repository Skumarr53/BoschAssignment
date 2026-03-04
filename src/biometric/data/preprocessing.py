"""Per-modality image transforms for fingerprint and iris data.

Uses torchvision.transforms.v2 for composable, GPU-friendly preprocessing.
Aligns with typical multimodal biometric pipelines (resize, normalize).
"""

from typing import Any

import torch
from torchvision.transforms import v2

# Fingerprint: loaded as "L" (grayscale, 1ch) → resize, normalize.
# Iris: loaded as "RGB" (3ch) → 224x224 for pretrained backbones, ImageNet normalize.

# ImageNet normalization (for iris RGB)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Grayscale normalization (fingerprint)
GRAYSCALE_MEAN = (0.5,)
GRAYSCALE_STD = (0.5,)


def get_fingerprint_transform(
    *,
    size: tuple[int, int] = (96, 96),
    train: bool = False,
    mean: tuple[float, ...] = GRAYSCALE_MEAN,
    std: tuple[float, ...] = GRAYSCALE_STD,
) -> v2.Compose:
    """Build fingerprint preprocessing pipeline.

    Args:
        size: Target (H, W). Default 96x96 matches raw aspect.
        train: If True, add RandomHorizontalFlip for augmentation.
        mean: Normalization mean per channel (grayscale: 1 channel).
        std: Normalization std per channel.

    Returns:
        Composed transform: PIL → tensor (C, H, W) float32 normalized.
    """
    transforms_list: list[Any] = [
        v2.ToImage(),
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ]
    # Insert augmentation after Resize, before ToDtype (spatial before value transforms)
    if train:
        transforms_list.insert(3, v2.RandomHorizontalFlip(p=0.5))
    return v2.Compose(transforms_list)


def get_iris_transform(
    *,
    size: tuple[int, int] = (224, 224),
    train: bool = False,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> v2.Compose:
    """Build iris preprocessing pipeline.

    Args:
        size: Target (H, W). Default 224x224 for pretrained backbones.
        train: If True, add RandomHorizontalFlip for augmentation.
        mean: ImageNet mean for RGB (3 channels).
        std: ImageNet std for RGB.

    Returns:
        Composed transform: PIL → tensor (C, H, W) float32 normalized.
    """
    transforms_list: list[Any] = [
        v2.ToImage(),
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ]
    if train:
        transforms_list.insert(2, v2.RandomHorizontalFlip(p=0.5))
    return v2.Compose(transforms_list)


def get_multimodal_transforms(
    *,
    fingerprint_size: tuple[int, int] = (96, 96),
    iris_size: tuple[int, int] = (224, 224),
    train: bool = False,
) -> tuple[v2.Compose, v2.Compose]:
    """Return (fingerprint_transform, iris_transform) for MultimodalBiometricDataset.

    Args:
        fingerprint_size: Target size for fingerprint branch.
        iris_size: Target size for iris branch.
        train: If True, both transforms include RandomHorizontalFlip.

    Returns:
        Tuple of (fingerprint_transform, iris_transform).
    """
    fingerprint_transform = get_fingerprint_transform(size=fingerprint_size, train=train)
    iris_transform = get_iris_transform(size=iris_size, train=train)
    return fingerprint_transform, iris_transform
