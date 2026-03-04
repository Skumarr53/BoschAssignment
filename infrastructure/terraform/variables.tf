# [Phase 6d] Terraform variables for Azure stubs
# Plan: bosch_mlops_evaluation_plan_7132afe9

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "biometric-mlops"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "aks_node_count" {
  description = "AKS default node pool count"
  type        = number
  default     = 1
}

variable "aks_vm_size" {
  description = "AKS node VM size"
  type        = string
  default     = "Standard_DS2_v2"
}

variable "acr_sku" {
  description = "ACR SKU (Basic, Standard, Premium)"
  type        = string
  default     = "Basic"
}

variable "storage_account_tier" {
  description = "Storage account tier (Standard, Premium)"
  type        = string
  default     = "Standard"
}

variable "storage_account_replication" {
  description = "Storage replication (LRS, GRS, RAGRS, ZRS)"
  type        = string
  default     = "LRS"
}
