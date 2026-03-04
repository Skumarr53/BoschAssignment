# System Architecture

[Phase 7] C4-style architecture documentation for the Multimodal Biometric Recognition MLOps infrastructure.

---

## 1. Context (C4 Level 1)

The system provides production-grade ML infrastructure for training and inference of a multimodal biometric model (iris + fingerprint fusion). An MLOps engineer interacts with it via CLI scripts; the system consumes raw image data, trains models, and produces checkpoints and predictions.

```mermaid
graph TB
    subgraph "Users"
        Engineer[MLOps Engineer<br/>Runs training, inference, benchmarks]
    end

    subgraph "Biometric MLOps System"
        System[Biometric MLOps<br/>Trains fusion model, runs inference,<br/>manages experiments]
    end

    subgraph "External Systems"
        Dataset[Biometric Dataset<br/>Filesystem: BMP images<br/>per subject/modality]
        MLflow[MLflow<br/>Experiment tracking,<br/>model registry]
        Infra[Kubernetes / Azure<br/>Deployment target]
    end

    Engineer -->|train, infer, benchmark| System
    System -->|Read images, metadata| Dataset
    System -->|Log params, metrics, artifacts| MLflow
    System -->|Deploy container| Infra

    style System fill:#1168bd
    style Engineer fill:#08427b
    style Dataset fill:#999
    style MLflow fill:#999
    style Infra fill:#999
```

---

## 2. Containers (C4 Level 2)

The system is organized into four logical containers aligned with the `src/biometric/` package structure.

```mermaid
graph TB
    subgraph "Data Pipeline"
        ArrowCache[Arrow Cache<br/>PyArrow Parquet metadata]
        Dataset[Dataset<br/>MultimodalBiometricDataset]
        DataModule[DataModule<br/>BiometricDataModule]
        ParallelLoader[Parallel Loader<br/>Ray / multiprocessing]
    end

    subgraph "Model"
        FingerprintEnc[Fingerprint Encoder]
        IrisEnc[Iris Encoder]
        FusionModel[Multimodal Fusion Model]
    end

    subgraph "Training"
        Trainer[Trainer<br/>Custom training loop]
        Callbacks[Callbacks<br/>Checkpoint, EarlyStop, MLflow]
        Reproducibility[Reproducibility<br/>seed_everything, config snapshot]
    end

    subgraph "Inference"
        Pipeline[Inference Pipeline<br/>load_model, predict]
    end

    ArrowCache --> Dataset
    Dataset --> DataModule
    ParallelLoader --> DataModule
    DataModule --> Trainer
    FingerprintEnc --> FusionModel
    IrisEnc --> FusionModel
    FusionModel --> Trainer
    Callbacks --> Trainer
    Reproducibility --> Trainer
    FusionModel --> Pipeline

    style DataModule fill:#4ecdc4
    style FusionModel fill:#4ecdc4
    style Trainer fill:#4ecdc4
    style Pipeline fill:#4ecdc4
```

---

## 3. Components (C4 Level 3)

### 3.1 Data Layer (`src/biometric/data/`)

| Component | Responsibility |
|-----------|----------------|
| `arrow_cache` | Scans filesystem, builds Parquet metadata table, staleness detection. Enables fast dataset init without repeated glob. |
| `dataset` | `MultimodalBiometricDataset` pairs fingerprint + iris by subject; `PreloadedMultimodalDataset` for benchmarks. |
| `datamodule` | Orchestrates datasets, train/val/test splits, DataLoaders. Integrates Arrow cache and parallel preprocessing. |
| `parallel_loader` | Ray Data or multiprocessing for parallel preprocessing; config-driven via `data.use_parallel_preprocess`. |
| `preprocessing` | Per-modality transforms (resize, normalize, augment); `get_multimodal_transforms()` from config. |
| `discovery` | `discover_subjects()` — filesystem scan with sorted glob for determinism. |
| `parser` | `parse_fingerprint_filename()`, `parse_iris_path()` — extract subject_id, modality, metadata from paths. |

### 3.2 Model Layer (`src/biometric/models/`)

| Component | Responsibility |
|-----------|----------------|
| `fingerprint_encoder` | CNN branch for fingerprint images (1ch, 96×96). |
| `iris_encoder` | CNN branch for iris images (3ch, 224×224). |
| `fusion_model` | Concatenates embeddings → `nn.Linear` → logits. Port from reference Kaggle notebook (see `model_port_notes.md`). |
| `base` | Shared base classes for encoders. |

### 3.3 Training Layer (`src/biometric/training/`)

| Component | Responsibility |
|-----------|----------------|
| `trainer` | Custom training loop with AMP, gradient accumulation, `torch.compile()`. |
| `callbacks` | `CheckpointCallback`, `EarlyStoppingCallback`, `MetricLoggerCallback`, `MLflowCallback`. |
| `reproducibility` | `seed_everything()`, deterministic flags, config snapshots. |

### 3.4 Inference Layer (`src/biometric/inference/`)

| Component | Responsibility |
|-----------|----------------|
| `pipeline` | `load_model()` — load checkpoint into `MultimodalFusionModel`; `predict()` — batch inference. |

### 3.5 Scripts and Config

| Component | Responsibility |
|-----------|----------------|
| `scripts/train.py` | Hydra entry point; composes config, instantiates DataModule, model, Trainer; supports DDP via `torchrun`. |
| `scripts/infer.py` | Load checkpoint, run inference on provided images. |
| `scripts/benchmark.py` | DataLoader parameter sweep; outputs to `docs/performance_benchmarks.md`. |
| `scripts/preprocess_cache.py` | Build Arrow cache ahead of training. |
| `configs/` | Hydra configs for data, model, training, infrastructure; no hardcoded constants in `src/`. |

---

## 4. Data Flow

```mermaid
sequenceDiagram
    participant Engineer
    participant Train as train.py
    participant Hydra as Hydra Config
    participant DM as DataModule
    participant Cache as Arrow Cache
    participant Dataset as Dataset
    participant Model as Fusion Model
    participant Trainer as Trainer
    participant MLflow as MLflow

    Engineer->>Train: uv run python scripts/train.py
    Train->>Hydra: compose config
    Hydra->>DM: instantiate DataModule
    DM->>Cache: load_cache / build_cache
    Cache-->>DM: metadata table
    DM->>Dataset: MultimodalBiometricDataset(metadata)
    DM->>Trainer: fit(datamodule, model)
    loop Epoch
        Trainer->>Dataset: __getitem__
        Dataset-->>Trainer: (fp_tensor, iris_tensor, label)
        Trainer->>Model: forward
        Model-->>Trainer: logits
        Trainer->>Trainer: loss, backward, step
        Trainer->>MLflow: log metrics
    end
    Trainer->>MLflow: log checkpoint, config snapshot
```

---

## 5. Deployment View

```mermaid
graph TB
    subgraph "Local Development"
        Dev[uv run scripts/train.py<br/>or compose-dev]
    end

    subgraph "CI/CD"
        CI[GitHub Actions<br/>lint, test, build]
        CD[CD on tag<br/>Build + push image]
    end

    subgraph "Production (Optional)"
        K8s[Kubernetes<br/>Deployment, HPA, PVC]
        Helm[Helm Chart<br/>dev/staging/prod]
        TF[Terraform<br/>AKS, ACR, Storage]
    end

    Dev --> CI
    CI --> CD
    CD --> K8s
    Helm --> K8s
    TF --> K8s

    style K8s fill:#4ecdc4
```

---

## 6. Module Boundaries

- **Data ↔ Model**: DataModule yields `(fingerprint_tensor, iris_tensor, label)`; model expects `(fp, iris)` and returns logits.
- **Model ↔ Training**: Trainer receives `nn.Module`; callbacks save/load `state_dict` via checkpoint paths.
- **Training ↔ Inference**: Shared `MultimodalFusionModel`; inference loads checkpoint with `load_model()`.
- **Config**: Hydra composes `config.yaml` + overrides; all paths and hyperparameters flow from config, not hardcoded in `src/`.

---

## References

- [C4 Model](https://c4model.com/)
- [Design Decisions](./design_decisions.md)
- [Scalability Analysis](./scalability_analysis.md)
- [Model Port Notes](./model_port_notes.md)
- [Performance Benchmarks](./performance_benchmarks.md)
- [Phase 8 Verification](./phase8_verification.md)
