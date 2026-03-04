# VS Code Debugging Guide — Bosch Biometric MLOps

This guide explains how to debug and test the biometric data pipeline in VS Code.

---

## Prerequisites

- VS Code (or Cursor) with the **Python** and **Pylance** extensions
- Project dependencies installed: `uv sync`

---

## 1. Select Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Run **Python: Select Interpreter**
3. Choose `.venv/bin/python` (project virtual environment)

The workspace `.vscode/settings.json` already points to this interpreter.

---

## 2. Launch Configurations

Open the **Run and Debug** panel (`Ctrl+Shift+D` / `Cmd+Shift+D`) and choose a configuration:

| Configuration | Purpose |
|---------------|---------|
| **Debug Flow (End-to-End)** | Run the full pipeline with synthetic data. Best for understanding the flow. |
| **Debug Flow (Real Data)** | Same as above, but uses `Data/` directory (requires real dataset). |
| **Debug Preprocess Cache** | Build Arrow metadata cache only. |
| **Debug Current File** | Run the currently open Python file. |
| **Pytest: Current File** | Debug the currently open test file. |
| **Pytest: All Unit Tests** | Run all unit tests under the debugger. |

---

## 3. Debug the End-to-End Flow

### Step 1: Set Breakpoints

Open `scripts/debug_flow.py` and set breakpoints (click in the gutter) at:

- **Line ~55** — `discover_subjects()` — subject discovery
- **Line ~62** — `build_cache()` — cache build
- **Line ~68** — `dm.setup()` — DataModule setup
- **Line ~75** — `next(iter(train_loader))` — first batch load

### Step 2: Start Debugging

1. Select **Debug Flow (End-to-End)** from the dropdown
2. Press **F5** (or click the green play button)
3. Execution will stop at your first breakpoint

### Step 3: Step Through

- **F10** — Step Over (run current line, don’t enter functions)
- **F11** — Step Into (enter function calls)
- **Shift+F11** — Step Out (return from current function)
- **F5** — Continue to next breakpoint

### Step 4: Inspect Variables

In the **Variables** panel you can inspect:

- `subject_ids` — list of discovered subject IDs
- `table` — PyArrow table (cache)
- `dm` — `BiometricDataModule` instance
- `batch` — dict with `fingerprint`, `iris`, `label` tensors

---

## 4. Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  scripts/debug_flow.py                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  1. _create_synthetic_data()  →  /tmp/debug_biometric/                  │
│     ├── {id}/Fingerprint/*.BMP                                          │
│     ├── {id}/left/*.bmp                                                 │
│     └── {id}/right/*.bmp                                                │
│                                                                         │
│  2. discover_subjects(root)   →  [1, 2, 3]                              │
│     (utils/discovery.py)                                                │
│                                                                         │
│  3. build_cache(root)         →  biometric_metadata.parquet              │
│     (data/arrow_cache.py)     →  60 rows (FP + Iris metadata)            │
│                                                                         │
│  4. BiometricDataModule.setup()                                          │
│     (data/datamodule.py)      →  train/val/test splits by subject        │
│     └── MultimodalBiometricDataset (data/dataset.py)                    │
│         └── FingerprintDataset + IrisDataset (semantic pairing)           │
│                                                                         │
│  5. train_dataloader()        →  DataLoader                             │
│     next(iter(loader))        →  batch: {fingerprint, iris, label}       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Key Files to Step Into

| File | What to inspect |
|------|------------------|
| `src/biometric/utils/discovery.py` | `discover_subjects()` — how subject IDs are found |
| `src/biometric/data/arrow_cache.py` | `build_cache()`, `is_cache_stale()` — cache logic |
| `src/biometric/data/datamodule.py` | `setup()`, `_discover_subjects()` — split and dataset creation |
| `src/biometric/data/dataset.py` | `MultimodalBiometricDataset._build_pairs()` — semantic pairing |
| `src/biometric/data/preprocessing.py` | `get_multimodal_transforms()` — transforms |

---

## 6. Debug a Single Test

1. Open a test file (e.g. `tests/unit/test_dataset.py`)
2. Set a breakpoint inside a test (e.g. `test_len`)
3. Select **Pytest: Current File**
4. Press **F5**

---

## 7. Run Without Debugger

From the project root:

```bash
# End-to-end flow (synthetic data)
PYTHONPATH=src uv run python scripts/debug_flow.py --persist

# With real Data/ directory
PYTHONPATH=src uv run python scripts/debug_flow.py --data-root Data

# Preprocess cache only
uv run python scripts/preprocess_cache.py --root Data

# Run all unit tests
uv run pytest tests/unit/ -v
```

---

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'biometric'` | Ensure `PYTHONPATH` includes `src`. Launch configs set this automatically. |
| Breakpoints not hit | Confirm **justMyCode: false** in launch.json so you can step into libraries. |
| Wrong Python version | Run **Python: Select Interpreter** and pick `.venv/bin/python`. |
| Pytest config not found | Ensure `cwd` is `${workspaceFolder}` in the Pytest launch config. |
