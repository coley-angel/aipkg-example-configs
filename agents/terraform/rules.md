# Terraform Development Rules

## Code Style & Structure

### File Organization
- Use consistent file naming: `main.tf`, `variables.tf`, `outputs.tf`, `versions.tf`
- Group resources logically by service or function
- Keep modules focused and single-purpose
- Maximum file size: 500 lines (split into modules if larger)

### Naming Conventions
- Use snake_case for all resource names
- Prefix resources with project/environment: `prod_web_server`
- Use descriptive names that indicate purpose: `user_data_bucket` not `bucket1`
- Tags should follow: `Name`, `Environment`, `Project`, `ManagedBy`

### Resource Definitions
```hcl
resource "aws_instance" "web_server" {
  ami           = var.ami_id
  instance_type = var.instance_type
  
  tags = merge(
    var.common_tags,
    {
      Name = "${var.project}-${var.environment}-web"
      Role = "web-server"
    }
  )
}
```

## Best Practices

### State Management
- Always use remote state (S3 + DynamoDB for AWS)
- Enable state locking to prevent concurrent modifications
- Use workspaces for environment separation
- Never commit `.tfstate` files to version control

### Variables & Outputs
- Define all variables in `variables.tf` with descriptions
- Use type constraints: `string`, `number`, `list(string)`, `map(any)`
- Provide sensible defaults where appropriate
- Mark sensitive variables: `sensitive = true`
- Output all important resource attributes for downstream use

### Modules
- Create reusable modules for common patterns
- Version your modules using Git tags
- Document module inputs/outputs in README
- Use module composition over monolithic configurations

## Security

### Secrets Management
- NEVER hardcode credentials or secrets
- Use AWS Secrets Manager or Parameter Store
- Reference secrets via data sources, not variables
- Rotate credentials regularly

### IAM & Permissions
- Follow principle of least privilege
- Use IAM roles over access keys
- Enable MFA for sensitive operations
- Audit permissions regularly

### Network Security
- Use security groups with minimal required ports
- Implement network segmentation (VPC, subnets)
- Enable VPC Flow Logs for monitoring
- Use private subnets for databases and internal services

## Validation & Testing

### Pre-commit Checks
```bash
terraform fmt -recursive
terraform validate
tflint
checkov -d .
```

### Testing Strategy
- Use `terraform plan` to preview changes
- Test in dev/staging before production
- Use Terratest for automated testing
- Validate outputs match expectations

## Documentation

### Code Comments
- Comment complex logic and non-obvious decisions
- Document why, not what (code shows what)
- Include links to relevant documentation
- Add examples for module usage

### Required Documentation
- README.md with usage examples
- Architecture diagrams for complex setups
- Runbook for common operations
- Disaster recovery procedures

## Performance

### Resource Optimization
- Use data sources to reference existing resources
- Implement resource targeting for large infrastructures
- Use `-parallelism` flag appropriately
- Cache provider plugins

### Cost Optimization
- Tag all resources for cost tracking
- Use appropriate instance sizes
- Implement auto-scaling where applicable
- Review and remove unused resources

## Version Control

### Git Workflow
- Use feature branches for changes
- Require pull request reviews
- Run CI/CD validation on PRs
- Tag releases with semantic versioning

### Commit Messages
```
feat(ec2): add auto-scaling configuration
fix(rds): correct backup retention period
docs(readme): update module usage examples
```

## Error Handling

### Common Issues
- Always check provider version compatibility
- Handle resource dependencies explicitly with `depends_on`
- Use `lifecycle` blocks to prevent resource recreation
- Implement proper error messages in validation

### Debugging
```bash
export TF_LOG=DEBUG
terraform plan -out=plan.tfplan
terraform show -json plan.tfplan | jq
```

## Compliance

### Standards
- Follow CIS AWS Foundations Benchmark
- Implement encryption at rest and in transit
- Enable CloudTrail for audit logging
- Regular security scanning with tools like Prowler

### Required Tags
```hcl
tags = {
  Environment = var.environment
  Project     = var.project_name
  ManagedBy   = "Terraform"
  Owner       = var.team_name
  CostCenter  = var.cost_center
}
```
