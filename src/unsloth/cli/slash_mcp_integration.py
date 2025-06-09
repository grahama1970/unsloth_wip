"""Module docstring"""

import json
from pathlib import Path
from typing import Any

import click
from click.core import Command, Group


def extract_click_commands(group: click.Group, prefix: str = "") -> dict[str, Any]:
    """Recursively extract all Click commands and their metadata."""
    commands = {}

    for name, cmd in group.commands.items():
        full_name = f"{prefix}{name}" if prefix else name

        if isinstance(cmd, Group):
            # Recursively process subgroups
            sub_prefix = f"{full_name}." if full_name else ""
            sub_commands = extract_click_commands(cmd, sub_prefix)
            commands.update(sub_commands)
        elif isinstance(cmd, Command):
            # Extract command metadata
            params = []
            required = []

            for param in cmd.params:
                param_info = {
                "name": param.name,
                "type": "string",  # Default type
                "description": param.help or f"Parameter {param.name}",
                "required": param.required
                }

                # Map Click types to JSON schema types
                if isinstance(param.type, click.INT):
                    param_info["type"] = "integer"
                elif isinstance(param.type, click.FLOAT):
                    param_info["type"] = "number"
                elif isinstance(param.type, click.BOOL):
                    param_info["type"] = "boolean"
                elif isinstance(param.type, click.Path):
                    param_info["type"] = "string"
                    param_info["format"] = "path"

                    # Handle options vs arguments
                    if isinstance(param, click.Option):
                        param_info["is_option"] = True
                        param_info["opts"] = param.opts
                        if param.default is not None:
                            param_info["default"] = param.default
                        else:
                            param_info["is_argument"] = True

                            params.append(param_info)

                            if param.required:
                                required.append(param.name)

                                commands[full_name] = {
                                "description": cmd.help or f"Execute {name} command",
                                "params": params,
                                "required": required,
                                "callback": cmd.callback
                                }

                                return commands


                                def generate_slash_commands(cli: click.Group, output_dir: Path = Path(".claude/commands")):
                                    """Generate slash command files for Claude Code."""
                                    output_dir.mkdir(parents=True, exist_ok=True)

                                    commands = extract_click_commands(cli)
                                    generated = 0

                                    for cmd_name, cmd_info in commands.items():
                                        # Create slash command content
                                        content = f"""# {cmd_info['description']}

                                        ## Usage

                                        ```bash
                                        unsloth {cmd_name.replace('.', ' ')} [OPTIONS]
                                        ```

                                        ## Parameters

                                        """

                                        # Document parameters
                                        for param in cmd_info['params']:
                                            if param.get('is_option'):
                                                opts = ' '.join(param['opts'])
                                                content += f"- `{opts}`: {param['description']}"
                                                if 'default' in param:
                                                    content += f" (default: {param['default']})"
                                                    if param['required']:
                                                        content += " **[REQUIRED]**"
                                                        content += "\n"
                                                    elif param.get('is_argument'):
                                                        content += f"- `{param['name']}`: {param['description']}"
                                                        if param['required']:
                                                            content += " **[REQUIRED]**"
                                                            content += "\n"

                                                            # Add examples
                                                            content += """
                                                            ## Examples

                                                            """

                                                            # Generate example based on command type
                                                            if "train" in cmd_name:
                                                                content += """```bash
                                                                # Train a model
                                                                unsloth train --model unsloth/Phi-3.5-mini-instruct --dataset qa_data.jsonl

                                                                # Train with HuggingFace upload
                                                                unsloth train --model unsloth/Phi-3.5-mini-instruct --dataset qa_data.jsonl --hub-id username/model
                                                                ```"""
                                                            elif "runpod" in cmd_name:
                                                                if "list" in cmd_name:
                                                                    content += """```bash
                                                                    # List all RunPod pods
                                                                    unsloth runpod list
                                                                    ```"""
                                                                elif "gpus" in cmd_name:
                                                                    content += """```bash
                                                                    # Show available GPUs
                                                                    unsloth runpod gpus
                                                                    ```"""
                                                                elif "stop" in cmd_name:
                                                                    content += """```bash
                                                                    # Stop a pod
                                                                    unsloth runpod stop pod-id-here
                                                                    ```"""
                                                                elif "train" in cmd_name:
                                                                    content += """```bash
                                                                    # Train on RunPod
                                                                    unsloth runpod train --model unsloth/Llama-2-70b --dataset qa_data.jsonl
                                                                    ```"""
                                                                elif "enhance" in cmd_name:
                                                                    content += """```bash
                                                                    # Enhance dataset with student-teacher thinking
                                                                    unsloth enhance --input raw_qa.jsonl --output enhanced_qa.jsonl --model unsloth/Phi-3.5-mini-instruct
                                                                    ```"""
                                                                elif "validate" in cmd_name:
                                                                    content += """```bash
                                                                    # Validate a trained adapter
                                                                    unsloth validate --adapter ./outputs/adapter --base-model unsloth/Phi-3.5-mini-instruct
                                                                    ```"""

                                                                    content += """

                                                                    ---
                                                                    *Auto-generated slash command for Unsloth CLI*
                                                                    """

                                                                    # Write file
                                                                    safe_name = cmd_name.replace('.', '-')
                                                                    cmd_file = output_dir / f"unsloth-{safe_name}.md"
                                                                    cmd_file.write_text(content)
                                                                    generated += 1

                                                                    print(f"✅ Generated {generated} slash commands in {output_dir}")
                                                                    return generated


                                                                    def generate_mcp_config(
                                                                    cli: click.Group,
                                                                    output_file: Path = Path("unsloth_mcp_config.json"),
                                                                    host: str = "localhost",
                                                                    port: int = 5555
                                                                    ):
                                                                        """Generate MCP configuration for the Unsloth CLI."""

                                                                        commands = extract_click_commands(cli)

                                                                        # Convert to MCP tool format
                                                                        tools = {}
                                                                        for cmd_name, cmd_info in commands.items():
                                                                            # Build input schema
                                                                            properties = {}
                                                                            required = []

                                                                            for param in cmd_info['params']:
                                                                                prop_name = param['name'].replace('-', '_')
                                                                                properties[prop_name] = {
                                                                                "type": param['type'],
                                                                                "description": param['description']
                                                                                }

                                                                                if 'default' in param:
                                                                                    properties[prop_name]['default'] = param['default']

                                                                                    if param['required']:
                                                                                        required.append(prop_name)

                                                                                        # Create tool definition
                                                                                        tool_name = f"unsloth_{cmd_name.replace('.', '_')}"
                                                                                        tools[tool_name] = {
                                                                                        "description": cmd_info['description'],
                                                                                        "inputSchema": {
                                                                                        "type": "object",
                                                                                        "properties": properties,
                                                                                        "required": required
                                                                                        }
                                                                                        }

                                                                                        # Build MCP config
                                                                                        config = {
                                                                                        "name": "unsloth-mcp-server",
                                                                                        "version": "1.0.0",
                                                                                        "description": "MCP server for Unsloth enhanced training pipeline",
                                                                                        "server": {
                                                                                        "command": "unsloth-mcp",
                                                                                        "args": ["--host", host, "--port", str(port)]
                                                                                        },
                                                                                        "tools": tools,
                                                                                        "capabilities": {
                                                                                        "tools": True,
                                                                                        "prompts": False,
                                                                                        "resources": False
                                                                                        }
                                                                                        }

                                                                                        # Write config
                                                                                        output_file.write_text(json.dumps(config, indent=2))
                                                                                        print(f"✅ Generated MCP config: {output_file}")
                                                                                        print(f" Includes {len(tools)} tools")

                                                                                        # Print some key tools
                                                                                        print("\n Key tools available:")
                                                                                        key_tools = [
                                                                                        "unsloth_train",
                                                                                        "unsloth_enhance",
                                                                                        "unsloth_validate",
                                                                                        "unsloth_runpod_train",
                                                                                        "unsloth_runpod_list",
                                                                                        "unsloth_runpod_gpus"
                                                                                        ]
                                                                                        for tool in key_tools:
                                                                                            if tool in tools:
                                                                                                print(f"  - {tool}: {tools[tool]['description']}")

                                                                                                return config


                                                                                                def add_generation_commands(cli: click.Group):
                                                                                                    """Add slash command and MCP generation commands to the CLI."""

                                                                                                    @cli.command()
                                                                                                    @click.option("--output", "-o", type=click.Path(), default=".claude/commands",
                                                                                                    help="Output directory for slash commands")
                                                                                                    def generate_slash():
                                                                                                        """Generate slash commands for Claude Code."""
                                                                                                        output_dir = Path(output)
                                                                                                        count = generate_slash_commands(cli, output_dir)

                                                                                                        # Print usage instructions
                                                                                                        click.echo("\n To use in Claude Code:")
                                                                                                        click.echo("1. Make sure the commands are in your project's .claude/commands directory")
                                                                                                        click.echo("2. Use commands like: /unsloth-train, /unsloth-runpod-list, etc.")

                                                                                                        @cli.command()
                                                                                                        @click.option("--output", "-o", type=click.Path(), default="unsloth_mcp_config.json",
                                                                                                        help="Output file for MCP config")
                                                                                                        @click.option("--host", default="localhost", help="MCP server host")
                                                                                                        @click.option("--port", default=5555, help="MCP server port")
                                                                                                        def generate_mcp():
                                                                                                            """Generate MCP configuration file."""
                                                                                                            output_file = Path(output)
                                                                                                            config = generate_mcp_config(cli, output_file, host, port)

                                                                                                            # Print usage instructions
                                                                                                            click.echo("\n To use with MCP:")
                                                                                                            click.echo("1. Start the MCP server: unsloth-mcp")
                                                                                                            click.echo("2. Add the config to your MCP client")
                                                                                                            click.echo("3. Tools will be available with prefix 'unsloth_'")


                                                                                                            # Example slash commands that would be generated:
                                                                                                            EXAMPLE_SLASH_COMMANDS = """
                                                                                                            # Example slash commands that will be generated:

                                                                                                            /unsloth-train              # Run complete training pipeline
                                                                                                            /unsloth-enhance            # Enhance dataset with student-teacher
                                                                                                            /unsloth-validate           # Validate a trained adapter
                                                                                                            /unsloth-upload             # Upload to HuggingFace
                                                                                                            /unsloth-runpod-train       # Train on RunPod
                                                                                                            /unsloth-runpod-list        # List RunPod pods
                                                                                                            /unsloth-runpod-gpus        # Show available GPUs
                                                                                                            /unsloth-runpod-stop        # Stop a RunPod pod
                                                                                                            /unsloth-quickstart         # Show quickstart guide
                                                                                                            /unsloth-models             # List recommended models
                                                                                                            """


                                                                                                            if __name__ == "__main__":
                                                                                                                # Test with the unified CLI
                                                                                                                from .unified_cli import cli

                                                                                                                # Add generation commands
                                                                                                                add_generation_commands(cli)

                                                                                                                # Generate both slash commands and MCP config
                                                                                                                print(" Generating Unsloth CLI integrations...")

                                                                                                                # Generate slash commands
                                                                                                                slash_dir = Path(".claude/commands")
                                                                                                                generate_slash_commands(cli, slash_dir)

                                                                                                                # Generate MCP config
                                                                                                                mcp_file = Path("unsloth_mcp_config.json")
                                                                                                                generate_mcp_config(cli, mcp_file)

                                                                                                                print("\n✅ All integrations generated!")
                                                                                                                print(EXAMPLE_SLASH_COMMANDS)
