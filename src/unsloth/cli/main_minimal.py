"""Minimal working CLI for Unsloth."""

import typer
from rich import print

app = typer.Typer(

# Add slash command and MCP generation
from .slash_mcp_mixin import add_slash_mcp_commands
add_slash_mcp_commands(app)

    name="unsloth-cli",
    help="Unsloth fine-tuning pipeline for LoRA adapters",
    add_completion=False
)

@app.command()
def version():
    """Show version information."""
    print("[bold green]Unsloth CLI v0.1.0[/bold green]")
    print("Ready for fine-tuning!")

@app.command()
def train(
    dataset: str = typer.Argument(..., help="Dataset path"),
    model: str = typer.Option("llama-3.2-1b", "--model", help="Base model"),
    epochs: int = typer.Option(3, "--epochs", help="Number of epochs")
):
    """Train a model with Unsloth (placeholder)."""
    print(f"[yellow]Training would start with:[/yellow]")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print("[red]Note: Full unsloth library needs to be installed for actual training[/red]")

if __name__ == "__main__":
    app()
