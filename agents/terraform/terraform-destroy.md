---
description: Safely destroy Terraform-managed infrastructure
---

# Terraform Destroy

Safely destroy Terraform-managed infrastructure with proper planning and review.

## Steps

### 1. Plan Destroy
```bash
terraform plan -destroy -out=destroy.tfplan
```
Generate destruction plan to review what will be removed.

### 2. Review Destroy Plan
```bash
terraform show destroy.tfplan
```
Review what resources will be destroyed before proceeding.

### 3. Destroy
```bash
terraform apply destroy.tfplan
```
⚠️ **Requires Confirmation** - Execute destruction of infrastructure.
