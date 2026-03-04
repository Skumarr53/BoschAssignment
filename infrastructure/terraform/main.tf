# [Phase 6d] Azure infrastructure stubs: AKS, ACR, Storage
# Plan: bosch_mlops_evaluation_plan_7132afe9
# Why stubs: Demonstrates IaC awareness without requiring Azure credentials for apply

provider "azurerm" {
  features {}
}

# Resource group (required parent for all resources)
resource "azurerm_resource_group" "this" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.location

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# [Phase 6d] AKS - Kubernetes cluster for biometric workload
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.project_name}-${var.environment}-aks"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  dns_prefix          = "${var.project_name}-${var.environment}"

  default_node_pool {
    name       = "default"
    node_count = var.aks_node_count
    vm_size    = var.aks_vm_size
  }

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# [Phase 6d] ACR - Container registry for biometric images
resource "azurerm_container_registry" "acr" {
  name                = replace("${var.project_name}${var.environment}acr", "-", "")
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  sku                 = var.acr_sku
  admin_enabled       = false

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# [Phase 6d] Storage - Data and checkpoint storage
resource "azurerm_storage_account" "data" {
  name                     = replace("${var.project_name}${var.environment}data", "-", "")
  resource_group_name      = azurerm_resource_group.this.name
  location                 = azurerm_resource_group.this.location
  account_tier             = var.storage_account_tier
  account_replication_type = var.storage_account_replication

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
