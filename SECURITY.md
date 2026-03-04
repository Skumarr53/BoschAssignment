# Security Policy

[Phase 6e] Biometric MLOps security practices. Plan: bosch_mlops_evaluation_plan_7132afe9

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| develop | :white_check_mark: |

## Security Measures

### No Secrets in Code

- All credentials via environment variables or Kubernetes Secrets
- ConfigMap used only for non-sensitive config (e.g. `LOG_LEVEL`)
- Never commit API keys, passwords, or tokens

### Dependency Scanning

- **pip-audit** in CI for known CVEs
- **bandit** for Python security anti-patterns
- **gitleaks** pre-commit hook for secrets detection

### Container Security

- Non-root user in Docker (`appuser`, uid 1000)
- Read-only root filesystem in Kubernetes
- `allowPrivilegeEscalation: false`, `capabilities: drop ALL`

### Reporting a Vulnerability

Please report security issues privately. Do not open public issues for vulnerabilities.
