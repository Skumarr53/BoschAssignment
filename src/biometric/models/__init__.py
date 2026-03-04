"""Model layer: modality encoders and fusion.

Exports primary interfaces. Internal modules imported directly when needed.
"""

from biometric.models.base import EncoderBase
from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion_model import MultimodalFusionModel
from biometric.models.iris_encoder import IrisEncoder

__all__ = [
    "EncoderBase",
    "FingerprintEncoder",
    "IrisEncoder",
    "MultimodalFusionModel",
]
