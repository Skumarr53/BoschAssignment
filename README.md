# Biometric MLOps

Production-grade MLOps infrastructure for multimodal biometric recognition (iris + fingerprint).

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Runtime** | Python 3.12, PyTorch 2.x |
| **Package** | uv (lockfile, deterministic installs) |
| **Config** | Hydra (composition, CLI overrides) |
| **Data** | PyArrow/Parquet (metadata cache), Ray/multiprocessing (parallel preprocessing), torchvision.transforms |
| **Training** | Custom Trainer (AMP, gradient accumulation, callbacks), MLflow (experiment tracking) |
| **Quality** | ruff, mypy, pytest, pre-commit, pip-audit, bandit, gitleaks |
| **Infra** | Docker, Kubernetes, Helm, Terraform (Azure AKS/ACR/Storage) |
| **CI/CD** | GitHub Actions (lint, test, build, security scan) |

## Quick Start

```bash
uv sync
make docker-build
make docker-run ARGS="training.epochs=1"
```

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

- [Architecture](docs/architecture.md) — C4-style system and component diagrams
- [Tech Stack](docs/TECH_STACK.md) — Consolidated tools, frameworks, and practices
- [Design Decisions](docs/design_decisions.md) — ADRs for key technical choices
- [Scalability Analysis](docs/scalability_analysis.md) — Bottleneck and scaling strategies
- [Performance Benchmarks](docs/performance_benchmarks.md) — DataLoader benchmark results

## Security

See [SECURITY.md](SECURITY.md) for security practices, vulnerability reporting, and hardening (non-root containers, read-only filesystem, pip-audit, bandit, gitleaks).

adding comment to test cache usage in build process 
