# Infrastructure

This project uses **Azure** (AKS, ACR, Storage). Terraform provisions the cluster; kubectl uses your **local kubeconfig**.

## Why "AWS" errors when project uses Azure?

`kubectl` reads `~/.kube/config` (or `KUBECONFIG`). If your current context points to an **EKS** (AWS) cluster, kubectl will try to reach that cluster—and fail if `aws` CLI is missing or credentials are invalid.

**Fix**: Switch to the AKS context after Terraform apply:

```bash
cd infrastructure/terraform && terraform output -raw aks_get_credentials
# Then run the printed command, e.g.:
# az aks get-credentials --resource-group <rg> --name <aks> --overwrite-existing
```

## Validation

| Target | What it does |
|--------|--------------|
| `make k8s-validate` | Python YAML parse only—**works offline**, no kubeconfig |
| `kubectl apply -f infrastructure/k8s/ --dry-run=client` | Schema validation—**requires cluster** (kubectl fetches OpenAPI from server) |

Use `make k8s-validate` when your kubeconfig points to EKS or you have no cluster. Use `kubectl apply --dry-run=server` for full server-side validation after configuring AKS.

## Layout

- **terraform/** – Azure AKS, ACR, Storage (IaC stubs)
- **k8s/** – Raw manifests (Deployment, Service, ConfigMap, HPA, PVC, Namespace)
- **helm/** – Chart wrapping k8s manifests with env-specific values
- **docker/** – Container build
- **compose.yaml** – Local dev with podman-compose
