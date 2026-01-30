---
description: Complete Terraform workflow from initialization to apply
---

# Terraform Init Plan Apply

Complete workflow for Terraform infrastructure deployment with security scanning and validation.

## Steps

### 1. Format Code
```bash
terraform fmt -recursive
```
Format all Terraform files to ensure consistent style.

### 2. Initialize
```bash
terraform init -upgrade
```
Initialize Terraform and upgrade providers to latest versions.

### 3. Validate
```bash
terraform validate
```
Validate Terraform configuration syntax and structure.

### 4. Security Scan
```bash
checkov -d . --framework terraform
```
Run security and compliance scan with Checkov.

### 5. Plan
```bash
terraform plan -out=tfplan
```
Generate execution plan and save to file.

### 6. Show Plan
```bash
terraform show -json tfplan | jq
```
Display plan in readable JSON format.

### 7. Apply
```bash
terraform apply tfplan
```
⚠️ **Requires Confirmation** - Apply the execution plan to create/update infrastructure.
