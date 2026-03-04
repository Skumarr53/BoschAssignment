# Biometric MLOps

Production-grade MLOps infrastructure for multimodal biometric recognition (iris + fingerprint).

## Setup

```bash
uv sync
uv sync --group dev   # for pytest, ruff, mypy
```

## Run Tests

```bash
uv run pytest tests/ -v
# or with PYTHONPATH if package not installed:
PYTHONPATH=src uv run pytest tests/ -v
```

## Lint

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Documentation

- [Learning Challenge](docs/LEARNING_CHALLENGE.md) — End-to-end hands-on guide to own every component (what, why, how)
- [Architecture](docs/architecture.md) — C4-style system and component diagrams
- [Design Decisions](docs/design_decisions.md) — ADRs for key technical choices
- [Scalability Analysis](docs/scalability_analysis.md) — Bottleneck and scaling strategies
- [Performance Benchmarks](docs/performance_benchmarks.md) — DataLoader benchmark results
- [Model Port Notes](docs/model_port_notes.md) — Keras → PyTorch mapping
- [Phase 8 Verification](docs/phase8_verification.md) — Evaluation criteria checklist and quality gates

## Security

See [SECURITY.md](SECURITY.md) for security practices, vulnerability reporting, and hardening (non-root containers, read-only filesystem, pip-audit, bandit, gitleaks).

## Troubleshooting

### `AttributeError: module 'triton' has no attribute 'language'`

This occurs when a ROCm-specific or minimal `triton` package (without `triton.language`) is installed alongside PyTorch. PyTorch's `torch._dynamo` expects the full triton-lang API.

**Fix:** Uninstall the incompatible triton and let `uv` reinstall from the lockfile:

```bash
uv pip uninstall triton
uv run python scripts/train.py   # uv will reinstall correct triton from lockfile
```

The PyPI `triton` wheel (triton-lang) provides the full API. ROCm builds may pull a minimal triton; using the standard wheel resolves the issue.
