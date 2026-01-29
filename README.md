# AI Agent Configurations

A curated collection of AI agent configurations for professional development workflows. These agents provide context, rules, and automation for Terraform and Python development across multiple IDEs.

## Available Agents

### 🏗️ terraform_pro
Professional Terraform development agent with comprehensive IaC best practices, security scanning, and AWS expertise.

**Features:**
- Complete Terraform coding standards and style guide
- AWS resource patterns and examples
- Security best practices and compliance
- Automated workflow definitions
- Multi-environment setup patterns
- State management and debugging tips

**Includes:**
- `rules.md` - Comprehensive coding standards and best practices
- `context.md` - Common patterns, AWS resources, troubleshooting
- `workflows.json` - Automated Terraform workflows (init, plan, apply, destroy, etc.)

### 🐍 python_dev
Modern Python development agent with type safety, testing patterns, and best practices for Python 3.11+.

**Features:**
- PEP 8 compliance with type hints
- Modern Python patterns (dataclasses, Pydantic, async/await)
- Comprehensive testing strategies with pytest
- API client patterns and database operations
- Performance optimization techniques
- Security and logging best practices

**Includes:**
- `rules.md` - Python coding standards and project structure
- `context.md` - Common patterns, async operations, testing strategies
- `snippets.json` - VSCode snippets for rapid development

## Installation

### Using aipkg CLI

```bash
# Install aipkg
uv tool install git+https://github.com/coley-angel/aipkg.git

# Add this repository
aipkg add-repo my-agents git@github.com:YOUR_USERNAME/ai-agent-configs.git

# List available agents
aipkg list-agents

# Activate an agent
aipkg activate terraform_pro
aipkg activate python_dev

# Check status
aipkg status

# View agent details
aipkg info terraform_pro
```

### Manual Installation

If you prefer to manually copy files:

1. Clone this repository
2. Copy the agent files to your IDE's configuration directories
3. Refer to `aipkg.yaml` for the correct destination paths

## Repository Structure

```
ai-agent-configs/
├── aipkg.yaml              # Agent configuration file
├── README.md               # This file
└── agents/
    ├── terraform/
    │   ├── rules.md        # Terraform coding standards
    │   ├── context.md      # Terraform patterns and examples
    │   └── workflows.json  # Terraform automation workflows
    └── python/
        ├── rules.md        # Python coding standards
        ├── context.md      # Python patterns and examples
        └── snippets.json   # VSCode code snippets
```

## Supported IDEs

- **VSCode** - Full support with rules, context, workflows, and snippets
- **Windsurf** - Full support with rules, context, and global workflows
  - Rules: `~/.windsurf/cascade/rules/`
  - Context: `~/.windsurf/cascade/context/`
  - Workflows: `~/.codeium/windsurf/global_workflows/`
- **Claude Desktop** - Compatible with standard configuration paths

## Configuration

The `aipkg.yaml` file defines:
- Agent metadata (name, description, version)
- File mappings with explicit destination paths
- IDE-specific configurations
- Source file locations

### Example Configuration

```yaml
agents:
  terraform_pro:
    name: "terraform_pro"
    description: "Professional Terraform development agent"
    version: "1.0.0"
    files:
      - source_path: "agents/terraform/rules.md"
        dest_path: "~/.cascade/rules/terraform_pro.md"
        ide: "vscode"
        description: "Terraform coding standards for VSCode"
```

## Usage Examples

### Terraform Development

Once `terraform_pro` is activated:
- Your IDE will have access to Terraform best practices
- Automated workflows for common Terraform operations
- Security scanning and validation patterns
- AWS resource templates and examples

### Python Development

Once `python_dev` is activated:
- Type-safe Python development patterns
- Modern async/await examples
- Testing strategies with pytest
- API client and database patterns
- Code snippets for rapid development

## Customization

### Adding Custom Agents

1. Create a new directory under `agents/`
2. Add your agent files (rules, context, workflows, etc.)
3. Update `aipkg.yaml` with your agent definition
4. Commit and push changes
5. Run `aipkg list-agents` to see your new agent

### Modifying Existing Agents

1. Edit the agent files in `agents/terraform/` or `agents/python/`
2. Commit your changes
3. Run `aipkg deactivate <agent>` then `aipkg activate <agent>` to reload

## Version Control

This repository uses semantic versioning for agents:
- **1.0.0** - Initial stable release
- Update version in `aipkg.yaml` when making breaking changes
- Use git tags to mark releases

## Contributing

1. Fork this repository
2. Create a feature branch
3. Add or modify agent configurations
4. Test with `aipkg activate <agent>`
5. Submit a pull request

## Best Practices

### For Terraform Agents
- Keep rules focused on security and compliance
- Include real-world AWS examples
- Document common pitfalls and solutions
- Update workflows for new Terraform versions

### For Python Agents
- Emphasize type safety and modern patterns
- Include async/await examples
- Provide testing patterns for all scenarios
- Keep snippets practical and frequently used

## Troubleshooting

### Agent Not Found
```bash
# Refresh repository
aipkg remove-repo my-agents
aipkg add-repo my-agents git@github.com:YOUR_USERNAME/ai-agent-configs.git
```

### Files Not Installing
- Check destination paths in `aipkg.yaml`
- Ensure directories exist or can be created
- Verify file permissions

### Conflicts with Existing Files
```bash
# Use force flag to override
aipkg activate terraform_pro --force

# Or check what would be installed
aipkg activate terraform_pro --dry-run
```

## License

MIT License - Feel free to use and modify these configurations for your projects.

## Support

- Issues: Open an issue in this repository
- Questions: Check the aipkg documentation
- Updates: Watch this repository for new agents and improvements

## Roadmap

- [ ] Kubernetes agent with Helm and kubectl patterns
- [ ] Go development agent
- [ ] Rust development agent
- [ ] DevOps agent with CI/CD patterns
- [ ] Security scanning agent configurations
