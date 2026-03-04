# Learning Challenge — Answer Key

**Use this file only to validate your answers.** Do not read before attempting the questions.

---

## Phase 0: Scaffolding & Tooling

### Q1: What does `uv sync` do that `pip install -r requirements.txt` does not? Where is the lockfile?

**Answer**: `uv sync` reads `pyproject.toml` and `uv.lock`, resolves dependencies deterministically, and installs into the project's virtual environment. It uses a lockfile (`uv.lock`) that pins all transitive dependencies for reproducible installs. `pip install -r requirements.txt` does not have a native lockfile; `pip freeze` is manual and does not guarantee reproducibility. The lockfile is `uv.lock` at project root.

### Q2: Why does the project use `src/biometric/` as a package instead of flat scripts? How does `pyproject.toml` declare it?

**Answer**: A package structure enables `from biometric.data.dataset import ...` and proper Python path resolution. It demonstrates packaging maturity and allows the package to be installed (`pip install -e .` or `uv sync`). `pyproject.toml` declares it via `[tool.hatch.build.targets.wheel] packages = ["src/biometric"]` (Hatch build backend).

### Q3: How does `ruff` differ from `flake8` + `black` + `isort`? What does `ruff check` actually enforce?

**Answer**: Ruff is a single Rust-based tool that replaces flake8 (linting), black (formatting), and isort (import sorting). It is 10–100× faster. `ruff check` enforces rules from `[tool.ruff.lint]` in `pyproject.toml`: select = E, F, I, N, W, UP, B, C4, SIM (errors, pyflakes, isort, naming, warnings, pyupgrade, bugbear, flake8-comprehensions, simplify). It does not format; `ruff format` does that.

### Q4: Why is `mypy` configured with `strict = true`? What happens if you add `x: int = "hello"` in `src/biometric/utils/types.py`?

**Answer**: `strict = true` enables all strict type-checking options (no implicit optional, no untyped defs, etc.) for maximum type safety. Adding `x: int = "hello"` would cause mypy to report: `Incompatible default for argument "x" (default has type "str", argument has type "int")`. The code would still run at runtime (Python is dynamically typed), but mypy would fail the build.

---

## Phase 1: Data Layer

### Q1: What is the exact directory structure the dataset expects? Where is it defined? What happens if a subject has fingerprint but no iris?

**Answer**: Structure: `{root}/{subject_id}/Fingerprint/*.BMP` and `{root}/{subject_id}/left/*.bmp`, `{root}/{subject_id}/right/*.bmp`. Defined in `discovery.py` (`discover_subjects`) and `arrow_cache.py` (`_scan_fingerprints`, `_scan_iris`). If a subject has fingerprint but no iris: `MultimodalBiometricDataset` builds pairs from the Cartesian product of fingerprint and iris samples per subject; if there are no iris samples for that subject, that subject contributes no pairs and is effectively skipped for pairing.

### Q2: Why does `discover_subjects()` use `sorted()` after collecting IDs? What breaks without it?

**Answer**: `discover_subjects()` returns `sorted(ids)` (line 38 in `discovery.py`). Without sorting, iteration order over directories is filesystem-dependent (non-deterministic). Training splits, cache building, and reproducibility would vary across runs and machines. The Plan's Rule 2 explicitly requires determinism: "Every glob or iterdir call must be followed by .sorted()."

### Q3: How does `parse_fingerprint_filename()` extract `subject_id`, `gender`, `hand`, `finger_type`? What regex or logic does it use?

**Answer**: Uses regex `_FINGERPRINT_PATTERN = re.compile(r"^(\d+)__([MF])_(Left|Right)_(index|little|middle|ring|thumb)_finger\.BMP$", re.IGNORECASE)`. Groups: (1) subject_id, (2) gender M/F, (3) hand Left/Right, (4) finger type. Maps to enums: `Gender.M`/`Gender.F`, `Hand.LEFT`/`Hand.RIGHT`, `_FINGER_TYPE_MAP` for finger type.

### Q4: Why does `MultimodalBiometricDataset` pair fingerprint and iris by subject? What would "unpaired" mean?

**Answer**: Multimodal biometric recognition assumes the same person provides both modalities; pairing by subject ensures each (fingerprint, iris) pair belongs to one identity. "Unpaired" would mean random or cross-subject pairs (e.g., subject 1's fingerprint with subject 2's iris), which would be wrong for identity verification/recognition.

---

## Phase 2: Model Layer

### Q1: What is the output shape of `FingerprintEncoder` for input `(B, 1, 96, 96)`? Of `IrisEncoder` for `(B, 3, 224, 224)`?

**Answer**: Both output `(B, embedding_dim)`. Default `embedding_dim=128`, so `(B, 128)`. FingerprintEncoder: 96→48→24→12 via 3 conv blocks + MaxPool2d(2); flat size 128×12×12. IrisEncoder: 224→112→56→28→14 via 4 conv blocks; flat size 256×14×14.

### Q2: Why does `MultimodalFusionModel` concatenate embeddings instead of averaging? What would change with averaging?

**Answer**: Concatenation preserves modality-specific information; the classifier learns how to weight each modality. Averaging would lose information (e.g., if one modality is more discriminative). With averaging, the classifier input would be `(B, embedding_dim)` instead of `(B, 2*embedding_dim)`; we would need `nn.Linear(embedding_dim, num_classes)` instead of `nn.Linear(2*embedding_dim, num_classes)`.

### Q3: How does Hydra `instantiate(cfg.model)` create the model? Trace from `configs/model/default.yaml` to the actual class.

**Answer**: `configs/model/default.yaml` has `_target_: biometric.models.fusion_model.MultimodalFusionModel` and `num_classes: 45`, `embedding_dim: 128`. Hydra's `instantiate()` resolves `_target_` to the class, passes the other keys as kwargs: `MultimodalFusionModel(num_classes=45, embedding_dim=128)`.

### Q4: Why is the reference notebook (Keras) ported to PyTorch? What is documented in `model_port_notes.md`?

**Answer**: The evaluation instructions prefer PyTorch; the reference Kaggle notebook uses TensorFlow/Keras. The port preserves architecture and process flow. `model_port_notes.md` documents the Keras→PyTorch layer mapping (Conv2D→Conv2d, BatchNorm→BatchNorm2d, Dense→Linear, etc.) and input shapes.

---

## Phase 3: Training Pipeline

### Q1: What does `seed_everything(42)` actually set? List every component.

**Answer**: Sets: (1) `random.seed(42)`, (2) `np.random.seed(42)`, (3) `torch.manual_seed(42)`, (4) `torch.cuda.manual_seed_all(42)` if CUDA available, (5) `torch.backends.cudnn.deterministic = True`, (6) `torch.backends.cudnn.benchmark = False`, (7) `os.environ["PYTHONHASHSEED"] = "42"`.

### Q2: Why is there a custom Trainer instead of PyTorch Lightning? What would Lightning hide?

**Answer**: The evaluation tests PyTorch fluency; Lightning abstracts the training loop, AMP, gradient accumulation, and callbacks. A custom Trainer makes these explicit (e.g., `torch.amp.autocast`, `GradScaler`, accumulation logic in `trainer.py`). Lightning would hide exactly what evaluators want to see.

### Q3: How does gradient accumulation work? Where is it implemented? What is the effective batch size when `accumulation_steps=4` and `batch_size=8`?

**Answer**: Implemented in `Trainer._train_epoch`: gradients are accumulated over `gradient_accumulation_steps` mini-batches before `optimizer.step()`. Effective batch size = 4 × 8 = 32 (32 samples per optimizer update).

### Q4: Why does `CheckpointCallback` save `model_state_dict` and not the full model? What would break with `torch.save(model, path)`?

**Answer**: `state_dict` is portable (no pickled class definitions); loading only requires the same model architecture. `torch.save(model, path)` pickles the full object, which can break across Python versions, file paths, or when the class definition changes. Inference loads with `model.load_state_dict(ckpt["model_state_dict"])` and a fresh model instance.

---

## Phase 4: Performance & Benchmarking

### Q1: What does `scripts/benchmark.py` measure? What parameters does it sweep?

**Answer**: Measures DataLoader throughput (samples/sec) and init time. Sweeps: `num_workers` (e.g., 0, 2, 4, 8), `batch_size` (e.g., 8, 16, 32, 64), optionally `pin_memory`, `prefetch_factor`, `persistent_workers`. Can compare filesystem vs Arrow cache init time.

### Q2: Why is `num_workers` critical for DataLoader? What happens with `num_workers=0` vs `4`?

**Answer**: `num_workers > 0` spawns worker processes that prefetch batches in parallel while the main process trains. With `num_workers=0`, the main process loads data sequentially, causing GPU starvation. With `num_workers=4`, 4 workers load batches ahead of time; throughput typically increases until I/O or CPU becomes the bottleneck.

### Q3: How does the Arrow cache speed up dataset init? Measure: time to build dataset with and without cache.

**Answer**: Without cache: every init scans the filesystem (glob, stat, parse filenames). With cache: load a single Parquet file (metadata already computed). Init time drops from seconds to milliseconds. Measure with `_measure_init_time()` in `benchmark.py` or by timing `dm.setup()` with `use_cache=True` vs `False`.

### Q4: Why does `scalability_analysis.md` discuss sharding the cache? At what scale does a single Parquet file become a bottleneck?

**Answer**: A single Parquet file works up to ~10^6 rows; beyond that, loading and filtering become slow. Sharding (e.g., `metadata_0000.parquet`, `metadata_0001.parquet` by subject_id range) enables parallel load and reduces memory. At 45,000 subjects (~450K samples), a single file is still fine; sharding is for 100K+ subjects.

---

## Phase 5: Testing & CI/CD

### Q1: What is the difference between `tests/unit/`, `tests/integration/`, and `tests/performance/`? When does each run?

**Answer**: Unit: fast, isolated, mocked I/O; runs on every CI run. Integration: slower, real data/subset, end-to-end flows; runs only on `main` and `develop` (see CI `if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'`). Performance: benchmarks with `@pytest.mark.performance`; runs on every CI run.

### Q2: Why does `conftest.py` set `LOG_LEVEL=WARNING`? What breaks if it is INFO?

**Answer**: Reduces log noise during tests; otherwise structlog would emit INFO for every dataset build, cache hit, etc., cluttering test output. With INFO, test output is harder to read; no functional break, but debugging is harder.

### Q3: How does the `synthetic_data` fixture create BMP files? What structure must it match?

**Answer**: Creates `tmp_path/{1,2,3}/Fingerprint/*.BMP` (10 per subject, format `{id}__{M|F}_{Left|Right}_{finger}_finger.BMP`) and `tmp_path/{1,2,3}/left/*.bmp`, `right/*.bmp` (5 each, format `subj{id}l{1-5}.bmp`, `subj{id}r{1-5}.bmp`). Must match the parsers' expected formats and directory structure.

### Q4: Why does CI run integration tests only on `main` and `develop`? What would happen if they ran on every PR?

**Answer**: Integration tests may require real data (`Data/`) or be slower; limiting to main/develop reduces CI time and avoids failures when `Data/` is missing on forks. If they ran on every PR: longer CI, possible failures if `Data/` is not present or paths differ.

---

## Phase 6: Infrastructure

### Q1: What is the difference between the multi-stage Dockerfile's build stage and runtime stage? Why two stages?

**Answer**: Build stage: installs build tools, runs `uv sync` (compiles deps). Runtime stage: copies only `.venv`, `src/`, `configs/`, `scripts/`; no build tools. Result: smaller image (~500MB vs ~2GB), no compilers in production, reduced attack surface.

### Q2: Why does the K8s Deployment use `runAsNonRoot` and `readOnlyRootFilesystem`? What would break if we removed them?

**Answer**: Security hardening: non-root limits privilege escalation; read-only root prevents tampering. Removing them: running as root increases risk; writable root allows malware persistence. The app must write only to mounted volumes (e.g., checkpoints PVC); it is designed for that.

### Q3: How does the Helm chart parameterize the image tag? Where is it overridden for dev vs prod?

**Answer**: `values.yaml` has `image.repository` and `image.tag`; templates use `{{ .Values.image.tag }}`. Overridden in `values-dev.yaml`, `values-staging.yaml`, `values-prod.yaml` (e.g., `tag: latest` for dev, `tag: v1.2.3` for prod).

### Q4: Why is Terraform in this repo if we are not deploying? What does it demonstrate?

**Answer**: Demonstrates infrastructure-as-code awareness: AKS, ACR, Storage Account stubs. Shows the team knows Terraform; full deployment would require Azure credentials. It is a "brownie point" for production readiness.

---

## Phase 7: Documentation

### Q1: What does the C4 model have at Level 1 (Context) vs Level 2 (Containers)? Where is this in `architecture.md`?

**Answer**: Level 1 (Context): system boundary, users, external systems—one big box. Level 2 (Containers): major subsystems (Data Pipeline, Model, Training, Inference). In `architecture.md`: Section 1 is Context (Mermaid diagram); Section 2 is Containers (four logical containers).

### Q2: Why are there 6 ADRs in `design_decisions.md`? What decision is *not* documented that you think should be?

**Answer**: The 6 ADRs cover: Trainer vs Lightning, MLflow, Hydra, Ray+multiprocessing, PyArrow, uv. Possible missing: ruff vs other linters, pytest vs other test frameworks, structlog vs logging, Docker multi-stage, K8s security context. (Open-ended; any reasonable choice counts.)

### Q3: How does `scalability_analysis.md` answer "What if we add a third modality?"

**Answer**: Describes adding `BiometricModality.FACE`, `parse_face_path()`, `FaceDataset`/registry, `FaceEncoder`, and changing classifier to `3 * embedding_dim`. Suggests a registry pattern so existing encoders stay unchanged.

### Q4: Why does `model_port_notes.md` exist? Who is the audience?

**Answer**: Documents the Keras→PyTorch port for transparency and reproducibility. Audience: reviewers, future maintainers, anyone verifying the model matches the reference notebook.

---

## Phase 8: End-to-End Verification

### Q1: What are the evaluation criteria (from the plan)? How many points each? Where is the evidence for each?

**Answer**: System Design 25, Python Quality 20, ML Workflow 15, Data Loading 15, Multimodal 10, CI/CD 10, Documentation 5. Evidence: `docs/phase8_verification.md` maps each to deliverables (architecture, ADRs, code, benchmarks, etc.).

### Q2: Why does `phase8_verification.md` list "Known Limitations"? What are they?

**Answer**: Docker build requires daemon; Terraform validate needs network; pre-commit needs git repo; some unit tests (parallel preprocessing) can be slow/flaky. These are environment-specific; CI handles them.

### Q3: How would you prove to a reviewer that the project is complete?

**Answer**: Run quality gates (lint, test, build), show evaluation criteria mapping, run training for 1 epoch, run inference. Point to `phase8_verification.md` and the evidence table.

---

## Final Exam

### 1: What is the exact flow when you run `uv run python scripts/train.py`? (Trace from entry point to first backward pass.)

**Answer**: (1) Hydra `compose("config", overrides)` → config. (2) `seed_everything()`. (3) `BiometricDataModule` setup (load/build cache, build datasets). (4) `instantiate(cfg.model)` → `MultimodalFusionModel`. (5) `Trainer.fit()`: for each epoch, iterate `train_loader` → batch dict → `model(fingerprint, iris)` → loss → `loss.backward()` (with accumulation) → `optimizer.step()` after N steps. First backward: after first accumulation step (or first step if accumulation_steps=1).

### 2: Why does `preprocess_with_pool` return `list[tuple[torch.Tensor, int]]` and not `list[tuple[np.ndarray, int]]`?

**Answer**: Workers return `(np.ndarray, int)` because numpy arrays are picklable; the main process converts with `torch.from_numpy(arr)` before appending. The Dataset and model expect tensors; returning tensors keeps the API consistent and avoids repeated conversion at the call site.

### 3: What would break if we removed `seed_everything` from the training script?

**Answer**: Training would be non-reproducible: different runs would yield different losses/accuracies. Data shuffling, weight init, dropout, and augmentation would vary. Reproducibility is a core requirement.

### 4: How does Hydra's `compose()` merge `config.yaml`, `data/default.yaml`, and CLI overrides?

**Answer**: `defaults` in `config.yaml` list `data/default`, etc. Hydra loads each file and merges (later overrides earlier for same keys). CLI overrides (e.g., `training.epochs=2`) are applied last and override merged config. Result: single OmegaConf object.

### 5: What is the purpose of `source_fingerprint` in the Arrow cache metadata?

**Answer**: Staleness detection. It is a hash of (sorted file paths + mtime) of the data directory. On load, `is_cache_stale()` recomputes the hash and compares; if the data changed (files added/removed/modified), the cache is invalid and must be rebuilt.
