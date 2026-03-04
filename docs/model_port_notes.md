# Model Port Notes — Keras → PyTorch

**Reference**: [Kaggle: omidsakaki1370/multimodal-biometric-recognition-system](https://www.kaggle.com/code/omidsakaki1370/multimodal-biometric-recognition-system)

This document maps the reference TensorFlow/Keras implementation to our PyTorch port.

---

## Architecture Overview

| Component | Keras (Reference) | PyTorch (This Project) |
|-----------|-------------------|------------------------|
| Fingerprint branch | `Sequential` / `Model` with Conv2D, BatchNorm, MaxPool | `FingerprintEncoder` — `nn.Conv2d`, `nn.BatchNorm2d`, `nn.MaxPool2d` |
| Iris branch | Similar CNN or pretrained backbone | `IrisEncoder` — 3ch→32→64→128→256 → 128-dim |
| Fusion | Concatenate → Dense → softmax | `MultimodalFusionModel` — concat(256) → `nn.Linear` → logits |

---

## Layer Mapping

| Keras | PyTorch |
|-------|---------|
| `Conv2D(filters, kernel_size, padding="same")` | `nn.Conv2d(in_ch, out_ch, kernel_size, padding=k//2)` |
| `BatchNormalization()` | `nn.BatchNorm2d(channels)` |
| `MaxPooling2D(2)` | `nn.MaxPool2d(2)` |
| `Flatten()` | `nn.Flatten()` |
| `Dense(units)` | `nn.Linear(in_features, units)` |
| `model.compile()` + `model.fit()` | Custom training loop (Phase 3) |
| `ImageDataGenerator` | PyTorch `Dataset` + `DataLoader` (Phase 1) |

---

## Input Shapes

- **Fingerprint**: `(B, 1, 96, 96)` — grayscale, matches `configs/data/default.yaml`
- **Iris**: `(B, 3, 224, 224)` — RGB, ImageNet-sized for potential pretrained backbones

---

## Notes

- The exact reference notebook architecture will be refined when the Kaggle notebook is inspected.
- Current implementation follows typical multimodal biometric patterns: per-modality CNNs → concatenated embeddings → classifier.
- No hyperparameter tuning or accuracy optimization per project instructions.
