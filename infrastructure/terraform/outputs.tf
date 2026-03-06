# [Phase 6d] Terraform outputs for downstream use
# Plan: bosch_mlops_evaluation_plan_7132afe9

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.this.name
}

output "aks_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.aks.name
}

output "aks_fqdn" {
  description = "AKS API server FQDN"
  value       = azurerm_kubernetes_cluster.aks.fqdn
}

output "acr_login_server" {
  description = "ACR login server URL"
  value       = azurerm_container_registry.acr.login_server
}

output "storage_account_name" {
  description = "Storage account name"
  value       = azurerm_storage_account.data.name
}

output "storage_primary_connection_string" {
  description = "Storage account primary connection string"
  value       = azurerm_storage_account.data.primary_connection_string
  sensitive   = true
}

output "aks_get_credentials" {
  description = "az aks get-credentials command to configure kubectl for AKS"
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.this.name} --name ${azurerm_kubernetes_cluster.aks.name} --overwrite-existing"
}
