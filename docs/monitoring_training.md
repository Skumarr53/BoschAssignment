## Monitoring Training Progress and Metrics

\[Phase 3c\] Instructions for tracking training progress with MLflow and Ray.

## MLflow (Primary — Training Metrics)p

MLflow logs hyperparameters, per-epoch metrics, and artifacts when `mlflow` is installed.

### 1\. Start MLflow UI (before or during training)

```plaintext
cd /home/sanky/projects/BoschAssignment
uv run mlflow ui --backend-store-uri mlruns
```

Open **http://127.0.0.1:5000** in a browser.

### 2\. Run Training (metrics auto-logged)

```plaintext
uv run python scripts/train.py
```

Each run logs:

*   **Params**: `epochs`, `learning_rate`, `batch_size`, `use_amp`, `gradient_accumulation_steps`
*   **Metrics**: `train_loss`, `val_loss`, `val_accuracy` (per epoch)
*   **Artifacts**: config snapshot, best checkpoint path (when configured)

### 3\. View Runs

*   **Experiments**: Select `biometric` experiment
*   **Runs**: Compare runs, filter by params, view metric curves
*   **Metrics**: `val_loss` (minimize), `val_accuracy` (maximize)

### 4\. Compare Runs

Use the MLflow UI to compare multiple runs, or via CLI:

```plaintext
uv run mlflow runs list --experiment-name biometric
```

## Ray (Data Preprocessing — Phase 1d)

**Ray is used for parallel preprocessing**, not for the training loop. Training uses PyTorch `DataLoader`.

### When Ray Is Used

*   `configs/data/parallel.yaml`: `backend: ray` (default); use `data.backend=multiprocessing` to switch
*   Scripts that call `preprocess_with_ray()` (e.g. batch preprocessing, benchmarks)

### Start Ray Dashboard (when using Ray)

```plaintext
ray start --head
# Or if Ray is already initialized by your script:
# Dashboard runs at http://127.0.0.1:8265
```

### View Ray Metrics

*   **Dashboard**: http://127.0.0.1:8265
*   **Tasks**: Preprocessing tasks, CPU/memory usage
*   **Actors**: Ray worker processes

### Current Training Pipeline

The `scripts/train.py` pipeline uses:

*   **Data**: `BiometricDataModule` → PyTorch `DataLoader` (no Ray in the main path)
*   **Preprocessing**: `torchvision.transforms` applied in the DataLoader workers
*   **Integrated**: Set `data.use_parallel_preprocess: true` to use Ray/multiprocessing in the training pipeline (Task 3).
*   **Optional**: `preprocess_with_ray()` or `preprocess_with_pool()` for offline batch preprocessing (see `src/biometric/data/parallel_loader.py`)

To use Ray for preprocessing in a custom script:

```python
from biometric.data.parallel_loader import preprocess_with_ray

results = preprocess_with_ray(paths, labels, modality="fingerprint")
```

## Quick Reference

| Tool | Purpose | When to Use |
| --- | --- | --- |
| MLflow | Training metrics, params | Always (runs with `train.py`) |
| Ray | Parallel data preprocessing | When `backend: ray` or custom scripts |

## Troubleshooting

*   **MLflow not logging**: Ensure `mlflow` is installed (`uv sync`).
*   **Ray not found**: Install with `uv sync --extra ray`.
*   **Metrics missing**: Check that `MLflowCallback` is in the trainer callbacks (see `scripts/train.py`).