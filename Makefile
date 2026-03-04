.PHONY: install lint test test-cov run pre-commit docker-build compose-dev compose-infer k8s-validate k8s-apply helm-lint helm-template tf-init tf-validate tf-fmt

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

# [Phase 6a] Docker and compose
docker-build:
	docker build -f infrastructure/docker/Dockerfile -t bosch-biometric:latest .

compose-dev:
	docker compose -f infrastructure/compose.yaml --profile dev run --rm app

compose-infer:
	docker compose -f infrastructure/compose.yaml --profile dev run --rm infer

# [Phase 6b] Kubernetes (validate=YAML syntax; apply=requires cluster)
k8s-validate:
	@python -c "import yaml; from pathlib import Path; [list(yaml.safe_load_all(f.open())) for f in Path('infrastructure/k8s').glob('*.yaml')]" && echo "K8s manifests: YAML valid"

k8s-apply:
	kubectl apply -f infrastructure/k8s/

# [Phase 6c] Helm chart
helm-lint:
	helm lint infrastructure/helm/biometric-mlops/

helm-template:
	@helm template biometric infrastructure/helm/biometric-mlops/ > /dev/null && echo "Helm chart: template OK"

# [Phase 6d] Terraform Azure stubs
tf-init:
	cd infrastructure/terraform && terraform init

tf-validate:
	cd infrastructure/terraform && terraform init && terraform validate

tf-fmt:
	cd infrastructure/terraform && terraform fmt -recursive -check
