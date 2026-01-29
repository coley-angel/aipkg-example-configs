# Terraform Development Context

## Project Overview

This agent provides comprehensive Terraform development support with focus on AWS infrastructure, best practices, and security.

## Common Patterns

### Multi-Environment Setup

```hcl
# Directory structure
environments/
├── dev/
│   ├── main.tf
│   ├── terraform.tfvars
│   └── backend.tf
├── staging/
│   └── ...
└── prod/
    └── ...

# Using workspaces
terraform workspace new dev
terraform workspace select dev
```

### Module Structure

```hcl
# modules/vpc/main.tf
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-vpc"
    }
  )
}

# Root main.tf
module "vpc" {
  source = "./modules/vpc"
  
  vpc_cidr    = "10.0.0.0/16"
  name_prefix = "production"
  tags        = local.common_tags
}
```

### Remote State Configuration

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

## AWS Resource Patterns

### EC2 with Auto Scaling

```hcl
resource "aws_launch_template" "app" {
  name_prefix   = "${var.project}-app-"
  image_id      = data.aws_ami.amazon_linux_2.id
  instance_type = var.instance_type
  
  vpc_security_group_ids = [aws_security_group.app.id]
  
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    environment = var.environment
  }))
  
  tag_specifications {
    resource_type = "instance"
    tags = local.common_tags
  }
}

resource "aws_autoscaling_group" "app" {
  name                = "${var.project}-app-asg"
  vpc_zone_identifier = var.private_subnet_ids
  target_group_arns   = [aws_lb_target_group.app.arn]
  
  min_size         = var.min_size
  max_size         = var.max_size
  desired_capacity = var.desired_capacity
  
  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }
  
  health_check_type         = "ELB"
  health_check_grace_period = 300
  
  tag {
    key                 = "Name"
    value               = "${var.project}-app-instance"
    propagate_at_launch = true
  }
}
```

### RDS with Encryption

```hcl
resource "aws_db_instance" "main" {
  identifier     = "${var.project}-${var.environment}-db"
  engine         = "postgres"
  engine_version = "14.7"
  instance_class = var.db_instance_class
  
  allocated_storage     = var.allocated_storage
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn
  
  db_name  = var.database_name
  username = var.master_username
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = var.backup_retention_days
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  deletion_protection = var.environment == "prod" ? true : false
  skip_final_snapshot = var.environment != "prod"
  
  tags = local.common_tags
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "db_password" {
  name = "${var.project}/${var.environment}/db-password"
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db_password.result
}
```

### S3 with Versioning and Encryption

```hcl
resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  
  rule {
    id     = "archive-old-versions"
    status = "Enabled"
    
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }
    
    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}
```

## Data Sources

### Common Data Sources

```hcl
# Get current AWS account
data "aws_caller_identity" "current" {}

# Get available AZs
data "aws_availability_zones" "available" {
  state = "available"
}

# Get latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Reference existing VPC
data "aws_vpc" "existing" {
  tags = {
    Name = "production-vpc"
  }
}
```

## Local Values

```hcl
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Owner       = var.team_name
    Repository  = var.repository_url
  }
  
  name_prefix = "${var.project_name}-${var.environment}"
  
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}
```

## Troubleshooting

### Common Issues

**State Lock Issues**
```bash
# Force unlock (use with caution)
terraform force-unlock <LOCK_ID>

# Check DynamoDB for locks
aws dynamodb scan --table-name terraform-locks
```

**Import Existing Resources**
```bash
# Import existing resource
terraform import aws_instance.web i-1234567890abcdef0

# Generate import configuration
terraform plan -generate-config-out=generated.tf
```

**Drift Detection**
```bash
# Check for configuration drift
terraform plan -refresh-only

# Show current state
terraform show
```

### Debugging Commands

```bash
# Detailed logging
export TF_LOG=DEBUG
export TF_LOG_PATH=terraform.log

# Validate configuration
terraform validate

# Format code
terraform fmt -recursive

# Show dependency graph
terraform graph | dot -Tsvg > graph.svg

# List resources in state
terraform state list

# Show specific resource
terraform state show aws_instance.web
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Terraform
on:
  pull_request:
    paths:
      - 'terraform/**'
  push:
    branches:
      - main

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: 1.6.0
      
      - name: Terraform Init
        run: terraform init
        
      - name: Terraform Format
        run: terraform fmt -check
        
      - name: Terraform Validate
        run: terraform validate
        
      - name: Terraform Plan
        run: terraform plan -out=tfplan
        
      - name: Checkov Scan
        uses: bridgecrewio/checkov-action@master
        with:
          directory: .
          framework: terraform
```

## Performance Tips

- Use `-parallelism=n` to control concurrent operations
- Cache provider plugins in CI/CD
- Use `terraform plan -target` for specific resources
- Implement proper resource dependencies
- Use data sources instead of hardcoded values
- Leverage module outputs for cross-stack references
