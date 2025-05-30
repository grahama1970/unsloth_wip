# CLI Entry Points

After installing the package with `pip install -e .` or `uv pip install -e .`, the following commands become available:

## Main Commands

### `unsloth` - Primary CLI
The main command-line interface for all operations:
```bash
# Train a model
unsloth train --model unsloth/Phi-3.5-mini-instruct --dataset qa.jsonl

# Enhance dataset
unsloth enhance --input raw.jsonl --output enhanced.jsonl --model phi-3.5

# Validate adapter
unsloth validate --adapter ./adapter --base-model phi-3.5

# RunPod commands
unsloth runpod list
unsloth runpod gpus
unsloth runpod train --model llama-70b --dataset qa.jsonl

# Generate integrations
unsloth generate-slash    # Create slash commands
unsloth generate-mcp      # Create MCP config
```

### `unsloth-cli` - Legacy Typer Interface
The original Typer-based CLI (maintained for compatibility):
```bash
unsloth-cli train --model phi-3.5 --dataset qa.jsonl
unsloth-cli enhance-thinking --input raw.jsonl --output enhanced.jsonl
```

### `unsloth-mcp` - MCP Server
Starts the Model Context Protocol server:
```bash
# Start with defaults (localhost:5555)
unsloth-mcp

# Custom host/port
unsloth-mcp --host 0.0.0.0 --port 8080

# Debug mode
unsloth-mcp --debug
```

## Installation

```bash
# From project root
pip install -e .

# Or with uv
uv pip install -e .

# Verify installation
which unsloth
which unsloth-mcp
```

## Entry Point Configuration

From `pyproject.toml`:
```toml
[project.scripts]
unsloth = "unsloth.cli.unified_cli:main"
unsloth-cli = "unsloth.cli.main:app"
unsloth-mcp = "unsloth.cli.mcp_server:main"
```

## MCP Server Usage

1. Generate MCP configuration:
   ```bash
   unsloth generate-mcp --output unsloth_mcp.json
   ```

2. Start the server:
   ```bash
   unsloth-mcp
   ```

3. Configure your MCP client to point to the server:
   - Host: localhost
   - Port: 5555
   - Config: unsloth_mcp.json

## Development Usage

If running from source without installation:
```bash
# Main CLI
python -m unsloth.cli.unified_cli

# MCP server
python -m unsloth.cli.mcp_server

# Legacy CLI
python -m unsloth.cli.main
```