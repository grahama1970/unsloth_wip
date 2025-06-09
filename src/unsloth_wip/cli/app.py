"""
unsloth_wip CLI Application
"""
Module: app.py
Description: Functions for app operations
Description: Functions for app operations

import typer
from .granger_slash_mcp_mixin import add_slash_mcp_commands

app = typer.Typer(name="unsloth_wip")

# Add MCP commands
add_slash_mcp_commands(app, project_name="unsloth_wip")

@app.command()
def status():
    """Check unsloth_wip status"""
    print("✅ unsloth_wip is ready!")

if __name__ == "__main__":
    app()
