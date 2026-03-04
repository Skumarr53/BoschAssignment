"""Inference pipeline for multimodal biometric model.

[Phase 5a] Load checkpoint and run prediction on batches.
"""

from biometric.inference.pipeline import InferencePipeline, load_model, predict

__all__ = ["InferencePipeline", "load_model", "predict"]
