# [Phase 6a] Multi-stage Dockerfile for biometric MLOps
# Plan: bosch_mlops_evaluation_plan_7132afe9
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Minimal deps for wheel builds (PyTorch provides wheels; some deps may need build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN pip install uv && uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="bosch-biometric-mlops"
LABEL org.opencontainers.image.description="Biometric MLOps training and inference"

WORKDIR /app

# [Phase 6e] Security: non-root user
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

RUN chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

USER appuser

ENTRYPOINT ["python", "scripts/train.py"]
