"""Data loading and preprocessing for multimodal biometric datasets.

Import the specific submodules you need rather than relying on this barrel:

    from biometric.data.dataset import FingerprintDataset
    from biometric.data.datamodule import BiometricDataModule
"""

from biometric.data.datamodule import BiometricDataModule
from biometric.data.dataset import (
    FingerprintDataset,
    IrisDataset,
    MultimodalBiometricDataset,
    PreloadedMultimodalDataset,
)

__all__ = [
    "BiometricDataModule",
    "FingerprintDataset",
    "IrisDataset",
    "MultimodalBiometricDataset",
    "PreloadedMultimodalDataset",
]
