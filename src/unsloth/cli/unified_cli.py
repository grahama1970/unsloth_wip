"""
Module: unified_cli.py
Description: Functions for unified cli operations

External Dependencies:
- asyncio: [Documentation URL]
- click: [Documentation URL]
- rich: [Documentation URL]
- : [Documentation URL]
- runpod: [Documentation URL]

Sample Input:
>>> # Add specific examples based on module functionality

Expected Output:
>>> # Add expected output examples

Example Usage:
>>> # Add usage examples
"""

#!/usr/bin/env python3
"""Unified CLI for Unsloth enhanced training pipeline with RunPod support."""

import asyncio
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..data.thinking_enhancer import StudentTeacherConfig, ThinkingEnhancer

# Import all necessary components
from ..pipeline.complete_training_pipeline import CompletePipeline
from ..training.runpod_training_ops import run_training_on_runpod
from ..upload.hub_uploader import HubUploader
from ..validation.model_validator import ModelValidator
from .slash_mcp_integration import add_generation_commands

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Unsloth Enhanced Training Pipeline CLI.
    
    Train LoRA adapters with student-teacher thinking enhancement.
    Supports local and RunPod training for models of all sizes.
    """
    pass


# ===== Main Commands =====

@cli.command()
@click.option("--model", "-m", required=True, help="Model name (e.g., unsloth/Phi-3.5-mini-instruct)")
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Path to Q&A dataset")
@click.option("--output", "-o", default="./outputs/pipeline", help="Output directory")
@click.option("--hub-id", help="HuggingFace Hub model ID for upload")
@click.option("--force-runpod", is_flag=True, help="Force training on RunPod")
@click.option("--skip-enhancement", is_flag=True, help="Skip student-teacher enhancement")
@click.option("--max-samples", type=int, help="Maximum samples to process")
@click.option("--grokking", is_flag=True, help="Enable grokking for better generalization")
def train(model, dataset, output, hub_id, force_runpod, skip_enhancement, max_samples, grokking):
    """Run complete training pipeline (enhancement + training + upload)."""

    console.print("[bold green] Starting Unsloth Training Pipeline[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Dataset: {dataset}")

    pipeline = CompletePipeline(
        model_name=model,
        dataset_path=dataset,
        output_dir=output,
        hub_model_id=hub_id,
        use_runpod=force_runpod
    )

    async def _run():
        results = await pipeline.run_pipeline()

        if results["status"] == "completed":
            console.print("\n[bold green] Pipeline completed successfully![/bold green]")
            _print_results(results)
        else:
            console.print(f"\n[bold red] Pipeline failed: {results.get('error', 'Unknown error')}[/bold red]")

    asyncio.run(_run())


@cli.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input Q&A dataset")
@click.option("--output", "-o", required=True, help="Output enhanced dataset")
@click.option("--model", "-m", required=True, help="Model to use as student")
@click.option("--max-samples", type=int, help="Maximum samples to enhance")
@click.option("--max-iterations", default=3, help="Max iterations per question")
@click.option("--batch-size", default=10, help="Batch size for processing")
def enhance(input, output, model, max_samples, max_iterations, batch_size):
    """Enhance Q&A dataset with student-teacher thinking."""

    console.print("[bold cyan] Enhancing dataset with student-teacher thinking[/bold cyan]")
    console.print(f"Student Model: {model}")
    console.print("Teacher Model: anthropic/max (Claude)")

    async def _enhance():
        config = StudentTeacherConfig(
            teacher_model="anthropic/max",
            judge_model="gpt-4o-mini",
            max_iterations=max_iterations,
            batch_size=batch_size,
            use_local_student=False
        )

        enhancer = ThinkingEnhancer(config, base_model_name=model)

        with console.status("[bold green]Enhancing dataset..."):
            stats = await enhancer.enhance_dataset(
                input_path=Path(input),
                output_path=Path(output),
                max_samples=max_samples
            )

        _print_enhancement_stats(stats)

    asyncio.run(_enhance())


@cli.command()
@click.option("--adapter", "-a", required=True, type=click.Path(exists=True), help="Path to adapter")
@click.option("--base-model", "-b", required=True, help="Base model name")
@click.option("--prompts", "-p", multiple=True, help="Test prompts")
@click.option("--compare-base", is_flag=True, help="Compare with base model")
@click.option("--dataset", type=click.Path(exists=True), help="Validation dataset")
def validate(adapter, base_model, prompts, compare_base, dataset):
    """Validate a trained adapter."""

    console.print("[bold yellow] Validating adapter[/bold yellow]")
    console.print(f"Adapter: {adapter}")
    console.print(f"Base Model: {base_model}")

    async def _validate():
        validator = ModelValidator()

        test_prompts = list(prompts) if prompts else None

        with console.status("[bold green]Running validation tests..."):
            results = await validator.validate_adapter(
                adapter_path=Path(adapter),
                base_model=base_model,
                test_prompts=test_prompts,
                validation_dataset=Path(dataset) if dataset else None,
                compare_base=compare_base
            )

        _print_validation_results(results)

    asyncio.run(_validate())


@cli.command()
@click.option("--adapter", "-a", required=True, type=click.Path(exists=True), help="Path to adapter")
@click.option("--model-id", "-m", required=True, help="HuggingFace model ID")
@click.option("--base-model", "-b", required=True, help="Base model name")
@click.option("--private", is_flag=True, help="Make repository private")
@click.option("--tags", "-t", multiple=True, help="Additional tags")
def upload(adapter, model_id, base_model, private, tags):
    """Upload adapter to HuggingFace Hub."""

    console.print("[bold magenta] Uploading to HuggingFace Hub[/bold magenta]")
    console.print(f"Model ID: {model_id}")

    async def _upload():
        uploader = HubUploader()

        with console.status("[bold green]Uploading adapter..."):
            result = await uploader.upload_adapter(
                adapter_path=Path(adapter),
                model_id=model_id,
                base_model=base_model,
                private=private,
                tags=list(tags) if tags else None
            )

        if result["status"] == "success":
            console.print("\n[bold green] Upload successful![/bold green]")
            console.print(f" Model available at: {result['url']}")
        else:
            console.print(f"\n[bold red] Upload failed: {result.get('error')}[/bold red]")

    asyncio.run(_upload())


# ===== RunPod Commands =====

@cli.group()
def runpod():
    """RunPod training commands."""
    pass


@runpod.command()
@click.option("--model", "-m", required=True, help="Model name")
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Dataset path")
@click.option("--hub-id", help="HuggingFace Hub ID for upload")
@click.option("--epochs", default=3, help="Training epochs")
@click.option("--batch-size", default=4, help="Batch size")
@click.option("--learning-rate", default=2e-4, help="Learning rate")
def train(model, dataset, hub_id, epochs, batch_size, learning_rate):
    """Train a model on RunPod infrastructure."""

    console.print("[bold blue]☁️ Starting RunPod training[/bold blue]")
    console.print(f"Model: {model}")

    training_config = {
        "model_name": model,
        "max_seq_length": 2048,
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "hub_model_id": hub_id
    }

    async def _train():
        with console.status("[bold green]Training on RunPod..."):
            result = await run_training_on_runpod(
                model_name=model,
                dataset_path=Path(dataset),
                training_config=training_config,
                hub_model_id=hub_id
            )

        if result["status"] == "success":
            console.print("\n[bold green] Training completed![/bold green]")
            console.print(f"Adapter: {result['adapter_path']}")
        else:
            console.print(f"\n[bold red] Training failed: {result.get('error')}[/bold red]")

    asyncio.run(_train())


@runpod.command("list")
def list_pods():
    """List all RunPod pods."""
    import runpod

    pods = runpod.get_pods()

    if not pods:
        console.print("No pods found")
        return

    table = Table(title="RunPod Pods")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("GPU", style="yellow")

    for pod in pods:
        table.add_row(
            pod["id"],
            pod["name"],
            pod.get("desiredStatus", "UNKNOWN"),
            pod.get("machine", {}).get("gpuDisplayName", "N/A")
        )

    console.print(table)


@runpod.command()
@click.argument("pod_id")
@click.option("--terminate", is_flag=True, help="Terminate instead of stop")
def stop(pod_id, terminate):
    """Stop a RunPod pod."""
    import runpod

    try:
        if terminate:
            runpod.terminate_pod(pod_id)
            console.print(f"[green]Pod {pod_id} terminated[/green]")
        else:
            runpod.stop_pod(pod_id)
            console.print(f"[green]Pod {pod_id} stopped[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@runpod.command()
def gpus():
    """List available GPUs on RunPod."""
    import runpod

    gpus = runpod.get_gpus()

    if not gpus:
        console.print("No GPUs available")
        return

    table = Table(title="Available RunPod GPUs")
    table.add_column("GPU", style="cyan")
    table.add_column("Memory", style="magenta")
    table.add_column("Price/hr", style="green")
    table.add_column("Available", style="yellow")

    for gpu in gpus:
        available = "" if gpu.get("available", True) else ""
        table.add_row(
            gpu.get("displayName", "Unknown"),
            f"{gpu.get('memoryInGb', 0)} GB",
            f"${gpu.get('securePrice', 0):.2f}",
            available
        )

    console.print(table)


# ===== Utility Commands =====

@cli.command()
def quickstart():
    """Show quickstart guide."""

    guide = """
[bold cyan] Unsloth Enhanced Training Quickstart[/bold cyan]

[bold yellow]1. Set environment variables:[/bold yellow]
   export HF_TOKEN="your_huggingface_token"
   export ANTHROPIC_API_KEY="your_claude_key"
   export OPENAI_API_KEY="your_openai_key"
   export RUNPOD_API_KEY="your_runpod_key"  # Optional

[bold yellow]2. Basic training (local):[/bold yellow]
   unsloth train --model unsloth/Phi-3.5-mini-instruct \\
                 --dataset qa_data.jsonl \\
                 --output ./outputs

[bold yellow]3. Training with upload:[/bold yellow]
   unsloth train --model unsloth/Phi-3.5-mini-instruct \\
                 --dataset qa_data.jsonl \\
                 --hub-id username/my-model

[bold yellow]4. Force RunPod for large models:[/bold yellow]
   unsloth train --model unsloth/Llama-3.2-70B \\
                 --dataset qa_data.jsonl \\
                 --force-runpod

[bold yellow]5. Enhance dataset first:[/bold yellow]
   unsloth enhance --input qa_raw.jsonl \\
                   --output qa_enhanced.jsonl \\
                   --model unsloth/Phi-3.5-mini-instruct

[bold yellow]6. Validate trained model:[/bold yellow]
   unsloth validate --adapter ./outputs/adapter \\
                    --base-model unsloth/Phi-3.5-mini-instruct \\
                    --compare-base

[bold yellow]7. RunPod management:[/bold yellow]
   unsloth runpod list
   unsloth runpod gpus
   unsloth runpod stop <pod_id>

For more help: unsloth --help
"""
    console.print(guide)


@cli.command()
def models():
    """List recommended models and their requirements."""

    table = Table(title="Recommended Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Min GPU", style="yellow")
    table.add_column("RunPod?", style="green")

    models = [
        ("unsloth/Phi-3.5-mini-instruct", "3.8B", "RTX 3090 (24GB)", "No"),
        ("unsloth/Llama-3.2-1B-Instruct", "1B", "RTX 3060 (12GB)", "No"),
        ("unsloth/Llama-3.2-3B-Instruct", "3B", "RTX 3090 (24GB)", "No"),
        ("unsloth/mistral-7b-instruct-v0.3", "7B", "RTX 4090 (24GB)", "No"),
        ("unsloth/Meta-Llama-3.1-8B-Instruct", "8B", "RTX 4090 (24GB)", "No"),
        ("unsloth/gemma-2-9b-it", "9B", "RTX A6000 (48GB)", "Optional"),
        ("meta-llama/Llama-2-13b-hf", "13B", "A100 (40GB)", "Recommended"),
        ("meta-llama/Llama-2-70b-hf", "70B", "H100 (80GB)", "Required")
    ]

    for model, size, gpu, runpod in models:
        table.add_row(model, size, gpu, runpod)

    console.print(table)


# ===== Helper Functions =====

def _print_results(results):
    """Print pipeline results in a nice format."""

    if "enhancement" in results["steps"]:
        stats = results["steps"]["enhancement"]
        console.print("\n[bold cyan] Enhancement Statistics:[/bold cyan]")
        console.print(f"  Examples: {stats.get('enhanced_examples', 0)}")
        console.print(f"  Avg Iterations: {stats.get('average_iterations', 0):.2f}")
        console.print(f"  Convergence: {stats.get('convergence_rate', 0)*100:.1f}%")

    if "training" in results["steps"]:
        training = results["steps"]["training"]
        console.print("\n[bold green] Training Results:[/bold green]")
        console.print(f"  Location: {training.get('training_location', 'N/A')}")
        console.print(f"  Adapter: {training.get('adapter_path', 'N/A')}")

    if "upload" in results["steps"]:
        upload = results["steps"]["upload"]
        console.print("\n[bold magenta] Upload Results:[/bold magenta]")
        console.print(f"  URL: {upload.get('url', 'N/A')}")


def _print_enhancement_stats(stats):
    """Print enhancement statistics."""

    console.print("\n[bold green] Enhancement complete![/bold green]")

    table = Table(title="Enhancement Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Examples", str(stats["total_examples"]))
    table.add_row("Enhanced Examples", str(stats["enhanced_examples"]))
    table.add_row("Average Iterations", f"{stats['average_iterations']:.2f}")
    table.add_row("Convergence Rate", f"{stats['convergence_rate']*100:.1f}%")
    table.add_row("Total Iterations", str(stats["total_iterations"]))

    console.print(table)


def _print_validation_results(results):
    """Print validation results."""

    status_color = "green" if results["status"] == "passed" else "red"
    console.print(f"\n[bold {status_color}]Validation Status: {results['status'].upper()}[/bold {status_color}]")

    tests = results.get("tests", {})

    # Basic inference
    if "basic_inference" in tests:
        test = tests["basic_inference"]
        status = "" if test["passed"] else ""
        console.print(f"\n{status} Basic Inference: {'Passed' if test['passed'] else 'Failed'}")

    # Prompt responses
    if "prompt_responses" in tests:
        console.print("\n[bold]Test Prompts:[/bold]")
        for resp in tests["prompt_responses"]:
            status = "" if resp["passed"] else ""
            console.print(f"  {status} {resp['prompt'][:50]}...")

    # Performance
    if "performance" in tests:
        perf = tests["performance"]
        console.print("\n[bold]Performance:[/bold]")
        console.print(f"  Tokens/sec: {perf.get('tokens_per_second', 0):.2f}")
        console.print(f"  GPU Memory: {perf.get('gpu_memory_used_gb', 0):.2f} GB")

    # File integrity
    if "file_integrity" in tests:
        integrity = tests["file_integrity"]
        console.print("\n[bold]File Integrity:[/bold]")
        console.print(f"  Required files: {all(integrity['required_files'].values())}")


def main():
    """Main entry point."""
    # Add slash command and MCP generation commands
    add_generation_commands(cli)

    # Check for API keys
    required_keys = ["HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        console.print(f"[yellow]⚠️  Missing environment variables: {', '.join(missing_keys)}[/yellow]")
        console.print("Some features may not work without these keys.\n")

    cli()


if __name__ == "__main__":
    main()
