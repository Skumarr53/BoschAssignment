# Tech Stack Reference

Consolidated reference for tools, frameworks, and practices used in the Multimodal Biometric MLOps pipeline.

## Summary

| Layer | Component | Tool / Framework |
|-------|-----------|------------------|
| **Runtime** | Python | 3.12 |
| **ML** | Deep learning | PyTorch 2.x, torchvision |
| **Package** | Dependency management | uv (lockfile, deterministic installs) |
| **Config** | Hyperparameters | Hydra (composition, CLI overrides) |
| **Data** | Metadata cache | PyArrow, Parquet |
| **Data** | Parallel preprocessing | Ray, multiprocessing |
| **Data** | Transforms | torchvision.transforms.v2 |
| **Model** | Encoders, fusion | PyTorch nn.Module |
| **Training** | Loop, AMP, callbacks | Custom Trainer |
| **Training** | Experiment tracking | MLflow |
| **Quality** | Lint, format | ruff |
| **Quality** | Type checking | mypy |
| **Quality** | Tests | pytest |
| **Quality** | Security | pip-audit, bandit, gitleaks |
| **Infra** | Container | Docker (multi-stage) |
| **Infra** | Orchestration | Kubernetes, Helm |
| **Infra** | IaC | Terraform (Azure AKS/ACR/Storage) |
| **CI/CD** | Pipeline | GitHub Actions |

## Practices

- **Structured logging**: `structlog` with JSON output
- **Typed models**: Pydantic for config and data schemas
- **Pre-commit**: ruff, mypy, pytest, bandit, gitleaks
- **Reproducibility**: uv lockfile, Hydra config, MLflow artifact logging

## See Also

- [Architecture](architecture.md) — System design and component boundaries
- [Design Decisions](design_decisions.md) — ADRs for Trainer, MLflow, Hydra, Ray, PyArrow, uv
