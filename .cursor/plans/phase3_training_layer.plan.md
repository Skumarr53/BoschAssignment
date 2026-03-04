---
name: Phase 3 Training Layer
overview: "Incremental implementation of the training pipeline: reproducibility, callbacks, trainer loop, train script, AMP/MLflow."
todos:
  - id: p3a
    content: "Reproducibility: seed_everything, deterministic flags"
    status: completed
  - id: p3b
    content: "Callbacks: Checkpoint, EarlyStopping, MetricLogger"
    status: completed
  - id: p3c
    content: "Trainer core: forward, loss, backward, epoch loop"
    status: completed
  - id: p3d
    content: "scripts/train.py + Hydra config wiring"
    status: completed
  - id: p3e
    content: "AMP + gradient accumulation + MLflow (optional)"
    status: completed
isProject: false
---

# Phase 3: Training Pipeline — Phased Implementation Plan

**Reference**: [Bosch MLOps Plan](file:///home/sanky/.cursor/plans/bosch_mlops_evaluation_plan_7132afe9.plan.md) Phase 3

**Inputs** (from Phase 1–2):

- `BiometricDataModule` — train/val loaders
- `MultimodalFusionModel` — model from config
- `configs/` — Hydra data, model, training configs

---

## Phase 3a: Reproducibility

**Deliverables**:

- `src/biometric/training/__init__.py` — package init
- `src/biometric/training/reproducibility.py` — `seed_everything(seed)`, deterministic flags

**Implementation**:

- `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
- `os.environ["PYTHONHASHSEED"] = str(seed)`

**Gate**: Unit test verifies same seed → same random values; ruff + mypy pass.

---

## Phase 3b: Callbacks

**Deliverables**:

- `src/biometric/training/callbacks.py` — `CheckpointCallback`, `EarlyStoppingCallback`, `MetricLoggerCallback`

**Interface** (protocol or ABC):

- `on_epoch_start(trainer, epoch)`, `on_epoch_end(trainer, epoch, metrics)`
- Callbacks receive trainer ref for model/optimizer access

**Gate**: Unit tests for each callback; no training loop yet.

---

## Phase 3c: Trainer Core

**Deliverables**:

- `src/biometric/training/trainer.py` — `Trainer` class
- Forward pass, loss (CrossEntropyLoss), backward, optimizer step
- Epoch loop with train/val phases
- Callback invocation at epoch boundaries

**Gate**: 1 epoch runs without error on synthetic data; loss decreases.

---

## Phase 3d: train.py + Config Wiring

**Deliverables**:

- `scripts/train.py` — Hydra entry point
- Compose data, model, training configs
- Instantiate DataModule, model, Trainer; call `trainer.fit()`
- Config snapshot saved (Hydra default)

**Gate**: `uv run python scripts/train.py` completes 1 epoch; no hardcoded paths in src/.

---

## Phase 3e: AMP + Gradient Accumulation + MLflow

**Deliverables**:

- AMP via `torch.amp.autocast` + `GradScaler`
- Gradient accumulation for effective larger batch size
- MLflow logging (params, metrics, artifacts) — optional dep

**Gate**: Training runs with AMP; MLflow logs visible when mlflow installed.
