# Biometric MLOps — End-to-End Learning Challenge

**Purpose**: Own this project fully. Be able to explain every component's **what**, **why**, and **how**. No spoon-feeding. Learn by doing.

**Rule**: Before each section, answer the questions. If you cannot, trace the code until you can. Do not proceed until you can explain your answers.

---

## Phase 0: Scaffolding & Tooling

### Questions (Answer Before Proceeding)

1. **What** does `uv sync` do that `pip install -r requirements.txt` does not? Where is the lockfile?
2. **Why** does the project use `src/biometric/` as a package instead of flat scripts? How does `pyproject.toml` declare it?
3. **How** does `ruff` differ from `flake8` + `black` + `isort`? What does `ruff check` actually enforce?
4. **Why** is `mypy` configured with `strict = true`? What happens if you add `x: int = "hello"` in `src/biometric/utils/types.py`?

### Challenges

1. **Trace the install path**: Run `uv sync` and observe. Then run `uv pip list`. Explain: Where does each dependency come from? What is the difference between `[project.dependencies]` and `[project.optional-dependencies]`?
2. **Break the linter**: Intentionally add a ruff violation (e.g., unused import, wrong indentation). Run `make lint`. Fix it. Repeat for mypy (add a type error).
3. **Prove package structure**: From project root, run `python -c "from biometric.data.dataset import MultimodalBiometricDataset; print('OK')"`. Why does this work? What would break if `pyproject.toml` did not have `packages = ["src/biometric"]`?

### Verification

- You can explain the difference between `uv run pytest` and `pytest` (why the former).
- You can list all config files that `ruff` and `mypy` read from.
- You can add a new optional dependency group and install it.

---

## Phase 1: Data Layer

### Questions (Answer Before Proceeding)

1. **What** is the exact directory structure the dataset expects? Where is it defined? What happens if a subject has fingerprint but no iris?
2. **Why** does `discover_subjects()` use `sorted()` after `glob`? What breaks without it?
3. **How** does `parse_fingerprint_filename()` extract `subject_id`, `gender`, `hand`, `finger_type`? What regex or logic does it use?
4. **Why** does `MultimodalBiometricDataset` pair fingerprint and iris by subject? What would "unpaired" mean?

### Challenges

1. **Trace a single sample**: Pick one BMP path from `Data/` (or create synthetic). Trace from `MultimodalBiometricDataset.__getitem__(0)` backward: which parser? which transform? what tensor shapes at each step?
2. **Break the parser**: Create a malformed filename (e.g., wrong format). Run a test that uses it. What exception? Where is it raised? How does the codebase handle parse failures?
3. **Prove preprocessing**: Without reading the preprocessing module first, run `uv run pytest tests/unit/test_preprocessing.py -v`. For each failing test (if any), fix the test or the code. Document: what does each transform do to the image?
4. **Arrow cache deep dive**: Run `build_cache()` on `Data/` (or synthetic). Inspect the Parquet file. What columns? What is `source_fingerprint` in metadata? Why is it there?

### Verification

- You can draw the data flow: `Data/` → discovery → parser → Arrow cache → Dataset → DataLoader → batch.
- You can explain why iris rows have `null` for `gender`, `hand`, `finger_type`.
- You can add a new modality (e.g., face) to the schema and document the changes required.

---

## Phase 2: Model Layer

### Questions (Answer Before Proceeding)

**What** is the output shape of `FingerprintEncoder` for input `(B, 1, 96, 96)`? Of `IrisEncoder` for `(B, 3, 224, 224)`?

1. **Why** does `MultimodalFusionModel` concatenate embeddings instead of averaging? What would change with averaging?
2. **How** does Hydra `instantiate(cfg.model)` create the model? Trace from `configs/model/default.yaml` to the actual class.
3. **Why** is the reference notebook (Keras) ported to PyTorch? What is documented in `model_port_notes.md`?

### Challenges

1. **Count parameters**: Write a small script that loads `MultimodalFusionModel` and prints total trainable parameters. Compare with a forward pass: does the output shape match `num_classes`?
2. **Break the model**: Pass a batch with wrong shapes (e.g., iris `(B, 1, 96, 96)` instead of `(B, 3, 224, 224)`). What error? Where does it originate?
3. **Trace config to model**: Change `model.embedding_dim` in config. Run training for 1 step. Prove the change took effect (e.g., param count changed).

### Verification

- You can explain the fusion architecture: two encoders → concat → linear → logits.
- You can map each Keras layer in the reference to a PyTorch equivalent.
- You can modify the model (e.g., add dropout) and verify it in a test.

---

## Phase 3: Training Pipeline

### Questions (Answer Before Proceeding)

1. **What** does `seed_everything(42)` actually set? List every component (random, numpy, torch, cudnn, env).
2. **Why** is there a custom Trainer instead of PyTorch Lightning? What would Lightning hide?
3. **How** does gradient accumulation work? Where is it implemented? What is the effective batch size when `accumulation_steps=4` and `batch_size=8`?
4. **Why** does `CheckpointCallback` save `model_state_dict` and not the full model? What would break with `torch.save(model, path)`?

### Challenges

1. **Run training**: Execute `uv run python scripts/train.py training.epochs=1 data.batch_size=4`. Before running: predict what will be logged (loss, metrics). After: verify. Where is the checkpoint saved?
2. **Reproducibility test**: Run training twice with the same seed. Compare final loss (or first 5 epochs). If they differ, find why.
3. **Trace a callback**: Pick `EarlyStoppingCallback`. Trace: when is it invoked? What does it do to `Trainer.should_stop`? How does the Trainer react?
4. **MLflow**: If MLflow is installed, run training and check `mlruns/` or MLflow UI. What is logged? Params? Metrics? Artifacts?

### Verification

- You can explain the training loop: dataloader → forward → loss → backward → optimizer step (with accumulation).
- You can explain AMP: what is `autocast`? What is `GradScaler`? Why both?
- You can add a custom callback and have it run during training.

---

## Phase 4: Performance & Benchmarking

### Questions (Answer Before Proceeding)

1. **What** does `scripts/benchmark.py` measure? What parameters does it sweep?
2. **Why** is `num_workers` critical for DataLoader? What happens with `num_workers=0` vs `4`?
3. **How** does the Arrow cache speed up dataset init? Measure: time to build dataset with and without cache.
4. **Why** does `scalability_analysis.md` discuss sharding the cache? At what scale does a single Parquet file become a bottleneck?

### Challenges

1. **Run benchmark**: Execute `uv run python scripts/benchmark.py` (or the equivalent from config). Inspect `docs/performance_benchmarks.md`. What is the throughput (samples/sec) for your config?
2. **Profile one epoch**: Use `torch.profiler` or `cProfile` to profile a single training epoch. Identify: where is most time spent? I/O or compute?
3. **Stress test**: Increase `batch_size` until you get OOM. Document the limit. What is the bottleneck (GPU memory? CPU?).

### Verification

- You can explain the DataLoader pipeline: prefetch, num_workers, pin_memory.
- You can answer: "What happens when subjects grow from 45 to 45,000?"
- You can propose a change to improve throughput and justify it.

---

## Phase 5: Testing & CI/CD

### Questions (Answer Before Proceeding)

1. **What** is the difference between `tests/unit/`, `tests/integration/`, and `tests/performance/`? When does each run?
2. **Why** does `conftest.py` set `LOG_LEVEL=WARNING`? What breaks if it is INFO?
3. **How** does the `synthetic_data` fixture create BMP files? What structure must it match?
4. **Why** does CI run integration tests only on `main` and `develop`? What would happen if they ran on every PR?

### Challenges

1. **Run all tests**: `uv run pytest tests/ -v`. Note: which tests are slow? Why? Fix any failure (or document why it is environment-specific).
2. **Write a new test**: Add a unit test for a function you have not tested yet. Follow existing patterns. Ensure it passes.
3. **Break CI locally**: Simulate CI: run the exact commands from `.github/workflows/ci.yaml` (lint, test, build). Fix any failure.
4. **Trace a failing test**: Pick a test. Make it fail (e.g., change assertion). Run it. Read the traceback. Fix it. Repeat until you understand the flow.

### Verification

- You can explain the test pyramid: unit (fast, isolated) vs integration (slower, real deps) vs performance (benchmarks).
- You can add a new fixture and use it in multiple tests.
- You can trigger a CI run (push or PR) and interpret the results.

---

## Phase 6: Infrastructure

### Questions (Answer Before Proceeding)

1. **What** is the difference between the multi-stage Dockerfile's build stage and runtime stage? Why two stages?
2. **Why** does the K8s Deployment use `runAsNonRoot` and `readOnlyRootFilesystem`? What would break if we removed them?
3. **How** does the Helm chart parameterize the image tag? Where is it overridden for dev vs prod?
4. **Why** is Terraform in this repo if we are not deploying? What does it demonstrate?

### Challenges

1. **Docker**: Build the image (`make docker-build` or equivalent). Run a container. Execute `train.py` inside it. What fails? Fix it (e.g., missing data mount).
2. **K8s manifests**: Validate all manifests (`make k8s-validate`). Then: change `replicas` in deployment. Change a resource limit. Predict: what would `kubectl apply` do?
3. **Helm**: Run `helm template biometric infrastructure/helm/biometric-mlops/ -f infrastructure/helm/biometric-mlops/values-dev.yaml`. Inspect the output. What is different from `values-prod.yaml`?
4. **Terraform**: Run `terraform init` and `terraform validate` in `infrastructure/terraform/`. What resources are defined? What would `terraform plan` show (do not run apply)?

### Verification

- You can explain the container security context: non-root, read-only fs, dropped capabilities.
- You can map: Dockerfile → K8s Deployment → Helm template → Terraform AKS.
- You can add a new ConfigMap key and have it injected into the container.

---

## Phase 7: Documentation

### Questions (Answer Before Proceeding)

1. **What** does the C4 model have at Level 1 (Context) vs Level 2 (Containers)? Where is this in `architecture.md`?
2. **Why** are there 6 ADRs in `design_decisions.md`? What decision is *not* documented that you think should be?
3. **How** does `scalability_analysis.md` answer "What if we add a third modality?"
4. **Why** does `model_port_notes.md` exist? Who is the audience?

### Challenges

1. **Update an ADR**: Pick ADR-0002 (MLflow). Add a "Superseded by" or "Related" section for a hypothetical future decision (e.g., "We will add W&B for experiment comparison"). Follow the ADR format.
2. **Draw a missing diagram**: The architecture doc has Context and Containers. Add a Component-level diagram for the Data layer (arrow_cache, dataset, datamodule, parallel_loader) using Mermaid.
3. **Explain to a newcomer**: Write a one-page "Quick Start" that gets a new engineer from clone to "training has run 1 epoch" in under 10 minutes. Do not copy from README. Make it your own.

### Verification

- You can explain the architecture to someone without showing the doc.
- You can defend each of the 6 ADR decisions (why we chose X over Y).
- You can point to the exact doc section that answers a given question.

---

## Phase 8: End-to-End Verification

### Questions (Answer Before Proceeding)

1. **What** are the evaluation criteria (from the plan)? How many points each? Where is the evidence for each?
2. **Why** does `phase8_verification.md` list "Known Limitations"? What are they?
3. **How** would you prove to a reviewer that the project is complete?

### Challenges

1. **Full pipeline run**: From a clean state (or fresh clone), run: install → lint → test → build (Docker) → (optional) 1 epoch training → inference on a sample. Document every command and every failure. Fix until it works.
2. **Evaluation self-check**: For each of the 7 evaluation objectives, write one paragraph of evidence. "We satisfy X because ..." with a specific file/line or doc reference.
3. **Stress test**: Change one thing (e.g., `data_root` to a non-existent path, remove a config file, break a type). Run the pipeline. Document the failure mode. Fix it. Repeat for 3 different breakages.

### Verification

- You can run the full project without looking at this guide.
- You can explain every evaluation criterion and where the evidence lives.
- You can debug a failure by tracing from symptom to root cause.

---

## Cross-Cutting Challenges (Pro Level)

1. **Add a feature**: Implement a simple feature (e.g., "log GPU memory every N steps"). Trace the change through: config → training → callback → log. Ensure tests pass.
2. **Refactor safely**: Extract a function from `datamodule.py` into `utils/`. Ensure no behavior change. Run full test suite.
3. **Security audit**: Run `pip-audit` and `bandit`. Fix any finding. Run `gitleaks` (if configured). Document what each tool checks.
4. **Explain the stack**: Draw the full stack on a whiteboard: Data → Model → Training → Inference → Infra → CI. Time yourself: can you do it in 5 minutes?

---

## Final Exam (Do Not Peek)

1. What is the exact flow when you run `uv run python scripts/train.py`? (Trace from entry point to first backward pass.)
2. Why does `preprocess_with_pool` return `list[tuple[torch.Tensor, int]]` and not `list[tuple[np.ndarray, int]]`?
3. What would break if we removed `seed_everything` from the training script?
4. How does Hydra's `compose()` merge `config.yaml`, `data/default.yaml`, and CLI overrides?
5. What is the purpose of `source_fingerprint` in the Arrow cache metadata?

**Answer these without looking at the code.** Then verify.

---

## How to Use This Guide

- **Do not skip questions.** If you cannot answer, trace the code until you can.
- **Do not skip challenges.** Each one is designed to force understanding.
- **Document your answers.** Keep a learning log (markdown or notes). "I learned X because Y."
- **Time yourself.** Can you explain the data flow in 2 minutes? The training loop in 3?
- **Teach someone.** The best test: explain a component to a colleague. If you stumble, you don't own it yet.

Good luck. No shortcuts.
