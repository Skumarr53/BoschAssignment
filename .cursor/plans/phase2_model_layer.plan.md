---
name: ""
overview: ""
todos: []
isProject: false
---

# Phase 2: Model Layer — Phased Implementation Plan

**Reference**: [Kaggle: omidsakaki1370/multimodal-biometric-recognition-system](https://www.kaggle.com/code/omidsakaki1370/multimodal-biometric-recognition-system) (Keras → PyTorch port)

**Input shapes** (from Phase 1 DataModule):

- Fingerprint: `(B, 1, 96, 96)` grayscale
- Iris: `(B, 3, 224, 224)` RGB

---

## Phase 2a: Base + Fingerprint Encoder (This step)

**Deliverables**:

- `src/biometric/models/base.py` — Abstract `BaseEncoder` protocol
- `src/biometric/models/fingerprint_encoder.py` — CNN for 96×96 grayscale
- `src/biometric/models/__init__.py` — Barrel exports
- `docs/model_port_notes.md` — Keras→PyTorch mapping stub

**Architecture** (typical reference):

- Conv blocks: Conv2d → BatchNorm2d → ReLU → MaxPool2d
- Fingerprint: 1ch → 32 → 64 → 128 → flatten → 128-dim embedding

**Gate**: `FingerprintEncoder` forward pass produces `(B, 128)`; ruff + mypy pass.

---

## Phase 2b: Iris Encoder

**Deliverables**:

- `src/biometric/models/iris_encoder.py` — CNN for 224×224 RGB

**Architecture**:

- 3ch → 32 → 64 → 128 → 256 → flatten → 128-dim embedding
- Same block pattern as fingerprint; larger input → more conv stages

**Gate**: `IrisEncoder` forward pass produces `(B, 128)`; unit test.

---

## Phase 2c: Fusion Model

**Deliverables**:

- `src/biometric/models/fusion_model.py` — `MultimodalFusionModel`
- Concatenate embeddings → Dense → `num_classes`
- Config-driven `num_classes` (default 45)

**Gate**: End-to-end forward pass; batch from DataModule → logits `(B, num_classes)`.

---

## Phase 2d: Config + Sanity Check

**Deliverables**:

- `configs/model/default.yaml` — Hydra model config
- `scripts/debug_model.py` — Sanity forward pass with synthetic batch
- Update `configs/config.yaml` to include model defaults

**Gate**: `scripts/debug_model.py` runs without error; config instantiation works.
