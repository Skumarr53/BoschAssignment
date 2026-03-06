.PHONY: install lint test test-cov run pre-commit docker-build docker-run compose-dev compose-infer k8s-validate k8s-apply helm-lint helm-template helm-template-dev helm-template-prod tf-init tf-validate tf-plan tf-fmt

install:
	uv sync --all-extras

pre-commit:
	pre-commit run --all-files

lint:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/
	uv run mypy src/ scripts/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/unit/ -v --cov=src/biometric --cov-report=term --cov-report=html --cov-fail-under=70

run:
	python -m biometric

# [Phase 6a] Container build and compose (Podman rootless per system_context)
# Run without sudo: make docker-build
docker-build:
	podman build -f infrastructure/docker/Dockerfile -t bosch-biometric:latest .

# Run container with Data and checkpoints mounted (ENTRYPOINT already set)
# Ensures checkpoints is writable (fixes root-owned dirs from prior sudo runs)
# Examples: make docker-run
#           make docker-run ARGS="training.epochs=2"
docker-run:
	@mkdir -p checkpoints && chmod -R a+rwX checkpoints 2>/dev/null || true
	podman run --rm \
		--shm-size=2g \
		--user $(shell id -u):$(shell id -g) \
		-v $(CURDIR)/Data:/app/Data \
		-v $(CURDIR)/checkpoints:/app/checkpoints \
		bosch-biometric:latest $(ARGS)

compose-dev:
	podman-compose -f infrastructure/compose.yaml --profile dev run --rm app

compose-infer:
	podman-compose -f infrastructure/compose.yaml --profile dev run --rm infer

# [Phase 6b] Kubernetes (validate=YAML syntax; apply=requires cluster)
# k8s-validate: Python YAML parse only—works offline, no kubeconfig needed
# k8s-apply: requires AKS credentials (az aks get-credentials); see infrastructure/README.md
k8s-validate:
	@python -c "import yaml; from pathlib import Path; [list(yaml.safe_load_all(f.open())) for f in Path('infrastructure/k8s').glob('*.yaml')]" && echo "K8s manifests: YAML valid"

k8s-apply:
	kubectl apply -f infrastructure/k8s/

# [Phase 6c] Helm chart
helm-lint:
	helm lint infrastructure/helm/biometric-mlops/

helm-template:
	@helm template biometric infrastructure/helm/biometric-mlops/ > /dev/null && echo "Helm chart: template OK"

# Output rendered manifests for inspection (Phase 6 challenge)
helm-template-dev:
	helm template biometric infrastructure/helm/biometric-mlops/ -f infrastructure/helm/biometric-mlops/values-dev.yaml

helm-template-prod:
	helm template biometric infrastructure/helm/biometric-mlops/ -f infrastructure/helm/biometric-mlops/values-prod.yaml

# [Phase 6d] Terraform Azure stubs
tf-init:
	cd infrastructure/terraform && terraform init

tf-validate:
	cd infrastructure/terraform && terraform init && terraform validate

# Requires az login; shows planned resources (do not apply)
tf-plan:
	cd infrastructure/terraform && terraform init && terraform plan

tf-fmt:
	cd infrastructure/terraform && terraform fmt -recursive -check
