# [Phase 6d] Terraform and provider versions
# Plan: bosch_mlops_evaluation_plan_7132afe9
# Stubs: Demonstrates IaC awareness without requiring Azure credentials
terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  # Uncomment for remote state (requires backend config)
  # backend "azurerm" {}
}
