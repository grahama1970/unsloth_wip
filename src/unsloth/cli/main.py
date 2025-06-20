"""Command-line interface for Unsloth pipeline."""
Module: main.py

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table

from ..core.config import TrainingConfig
from ..core.enhanced_config import EnhancedTrainingConfig
from ..core.grokking_config import GrokkingConfig
from ..inference.generate import GenerationConfig, InferenceEngine
from ..models.upload import ModelCard, ModelUploader
from ..training.enhanced_trainer import EnhancedUnslothTrainer
from ..training.trainer import UnslothTrainer
from ..utils.logging import setup_logging
from ..utils.memory import log_memory_usage
from .granger_slash_mcp_mixin import add_slash_mcp_commands

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="unsloth-cli",
    help="Unsloth fine-tuning pipeline for LoRA adapters",
    add_completion=False
)
console = Console()


@app.command()
def enhance_thinking(
    input_path: Path = typer.Argument(..., help="Input Q&A JSONL file"),
    output_path: Path = typer.Argument(..., help="Output enhanced JSONL file"),
    student_model: str = typer.Option("unsloth/Phi-3.5-mini-instruct", "--student", help="Student model"),
    teacher_model: str = typer.Option("gpt-4o-mini", "--teacher", help="Teacher model"),
    max_iterations: int = typer.Option(3, "--iterations", help="Max student-teacher iterations"),
    max_samples: int | None = typer.Option(None, "--samples", help="Max samples to process"),
    batch_size: int = typer.Option(10, "--batch-size", help="Batch size for processing")
):
    """Enhance Q&A thinking fields using student-teacher approach."""
    import asyncio

    from ..data.thinking_enhancer import StudentTeacherConfig, ThinkingEnhancer

    console.print("[bold green]Enhancing thinking fields...[/bold green]")

    # Configure enhancer
    config = StudentTeacherConfig(
        student_model=student_model,
        teacher_model=teacher_model,
        max_iterations=max_iterations,
        batch_size=batch_size
    )

    # Show configuration
    table = Table(title="Enhancement Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Student Model", student_model)
    table.add_row("Teacher Model", teacher_model)
    table.add_row("Max Iterations", str(max_iterations))
    table.add_row("Batch Size", str(batch_size))

    console.print(table)

    # Initialize enhancer
    enhancer = ThinkingEnhancer(config)

    # Run enhancement
    try:
        stats = asyncio.run(enhancer.enhance_dataset(
            input_path=input_path,
            output_path=output_path,
            max_samples=max_samples
        ))

        console.print("\n[bold green]Enhancement complete![/bold green]")
        console.print(f"Total examples: {stats['total_examples']}")
        console.print(f"Enhanced examples: {stats['enhanced_examples']}")
        console.print(f"Average iterations: {stats['average_iterations']:.2f}")
        console.print(f"Convergence rate: {stats['convergence_rate']:.2%}")

    except Exception as e:
        console.print(f"[bold red]Enhancement failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def train_runpod(
    model_name: str = typer.Argument(..., help="Model to train (e.g., meta-llama/Llama-2-70b-hf)"),
    dataset_path: Path = typer.Argument(..., help="Path to dataset"),
    gpu_type: str = typer.Option("RTX A6000", "--gpu", help="GPU type for RunPod"),
    epochs: int = typer.Option(3, "--epochs", help="Training epochs"),
    lora_r: int = typer.Option(128, "--lora-r", help="LoRA rank"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size"),
    pod_name: str = typer.Option("unsloth-training", "--pod-name", help="RunPod pod name"),
    volume_size: int = typer.Option(100, "--volume-size", help="Volume size in GB")
):
    """Train large models on RunPod infrastructure."""
    import asyncio

    from ..training.runpod_trainer import RunPodTrainer, RunPodTrainingConfig

    console.print(f"[bold green]Starting RunPod training for {model_name}[/bold green]")

    # Create configuration
    config = RunPodTrainingConfig(
        model_name=model_name,
        dataset_path=str(dataset_path),
        dataset_source="arangodb" if "arangodb" in str(dataset_path) else "huggingface",
        gpu_type=gpu_type,
        pod_name=pod_name,
        volume_size_gb=volume_size,
        num_train_epochs=epochs,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        per_device_train_batch_size=batch_size
    )

    # Show configuration
    table = Table(title="RunPod Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model", model_name)
    table.add_row("GPU Type", gpu_type)
    table.add_row("Pod Name", pod_name)
    table.add_row("Volume Size", f"{volume_size} GB")
    table.add_row("Epochs", str(epochs))
    table.add_row("LoRA Rank", str(lora_r))

    console.print(table)

    async def run_training():
        trainer = RunPodTrainer(config)

        try:
            # Start pod
            console.print("\n Starting RunPod training pod...")
            pod_info = await trainer.start_training_pod()
            console.print(f" Pod started: {pod_info['id']}")
            console.print(f"   GPU: {pod_info['gpu']}")
            console.print(f"   API: {pod_info['api_base']}")

            # Upload dataset
            console.print("\n Uploading dataset...")
            await trainer.upload_dataset()
            console.print(" Dataset uploaded")

            # Start training
            console.print("\n Starting training job...")
            job_id = await trainer.start_training()
            console.print(f" Training started: {job_id}")

            # Monitor progress
            console.print("\n Monitoring progress...")
            async for progress in trainer.monitor_training(job_id):
                console.print(f"   Epoch {progress['epoch']}: Loss={progress['loss']:.4f}")

                if progress['status'] == 'completed':
                    break

            # Download adapter
            console.print("\n Downloading trained adapter...")
            adapter_path = await trainer.download_adapter()
            console.print(f" Adapter saved to: {adapter_path}")

            return adapter_path

        except Exception as e:
            console.print(f"[bold red]Training failed: {e}[/bold red]")
            raise
        finally:
            # Clean up
            console.print("\n Cleaning up...")
            await trainer.stop_pod()

    try:
        adapter_path = asyncio.run(run_training())
        console.print("\n[bold green]Training completed successfully![/bold green]")
        console.print(f"Adapter location: {adapter_path}")

    except Exception as e:
        console.print(f"[bold red]RunPod training failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def train(
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    model_name: str = typer.Option(None, "--model", "-m", help="Model name"),
    dataset_path: str = typer.Option(None, "--dataset", "-d", help="Dataset path"),
    output_dir: str = typer.Option("./outputs", "--output", "-o", help="Output directory"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(5e-5, "--lr", help="Learning rate"),
    lora_r: int = typer.Option(8, "--lora-r", help="LoRA rank"),
    log_file: Path | None = typer.Option(None, "--log-file", help="Log file path"),
    use_grokking: bool = typer.Option(False, "--grokking", help="Enable grokking mode (extended training)"),
    grokking_multiplier: float = typer.Option(30.0, "--grokking-multiplier", help="Multiply epochs for grokking"),
    enhanced: bool = typer.Option(False, "--enhanced", help="Use enhanced trainer with advanced features")
):
    """Train a model with LoRA adapters."""
    setup_logging(log_file)

    console.print("[bold green]Starting Unsloth training...[/bold green]")

    # Determine which trainer to use
    if enhanced or use_grokking:
        # Use enhanced trainer for advanced features
        if config_file and config_file.exists():
            import json
            with open(config_file) as f:
                config_dict = json.load(f)
            config = EnhancedTrainingConfig(**config_dict)
            console.print(f"Loaded enhanced configuration from {config_file}")
        else:
            # Create enhanced config
            config = EnhancedTrainingConfig(
                model_name=model_name or os.getenv("DEFAULT_MODEL", "unsloth/Phi-3.5-mini-instruct"),
                dataset_path=dataset_path or os.getenv("ARANGODB_QA_PATH", "/home/graham/workspace/experiments/arangodb/qa_output"),
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                lora_r=lora_r,
                dataset_source="arangodb" if "arangodb" in (dataset_path or "") else "huggingface"
            )

            # Add grokking if requested
            if use_grokking:
                config.use_grokking = True
                config.grokking = GrokkingConfig(
                    enable_grokking=True,
                    grokking_multiplier=grokking_multiplier,
                    grokking_weight_decay=0.1,
                    disable_early_stopping=True
                )
                console.print(f"[bold yellow] Grokking mode enabled - training for {int(epochs * grokking_multiplier)} epochs![/bold yellow]")

        trainer_class = EnhancedUnslothTrainer
    else:
        # Use standard trainer
        if config_file and config_file.exists():
            import json
            with open(config_file) as f:
                config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
            console.print(f"Loaded configuration from {config_file}")
        else:
            config = TrainingConfig(
                model_name=model_name or os.getenv("DEFAULT_MODEL", "unsloth/Phi-3.5-mini-instruct"),
                dataset_path=dataset_path or os.getenv("ARANGODB_QA_PATH", "/home/graham/workspace/experiments/arangodb/qa_output"),
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                lora_r=lora_r,
                dataset_source="arangodb" if "arangodb" in (dataset_path or "") else "huggingface"
            )

        trainer_class = UnslothTrainer

    # Show configuration
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model", config.model_name)
    table.add_row("Dataset", config.dataset_path)
    table.add_row("Output Dir", config.output_dir)
    table.add_row("Epochs", str(config.num_train_epochs))
    table.add_row("Batch Size", str(config.per_device_train_batch_size))
    table.add_row("Learning Rate", str(config.learning_rate))
    table.add_row("LoRA Rank", str(config.lora_r))

    console.print(table)

    # Log memory before training
    log_memory_usage("Before training")

    # Initialize trainer
    trainer = trainer_class(config)

    # Train
    try:
        result = trainer.train()
        console.print("[bold green]Training completed![/bold green]")
        console.print(f"Adapter saved to: {result.adapter_path}")
        console.print(f"Training time: {result.training_time:.2f} seconds")

        if result.metrics:
            console.print("\n[bold]Final Metrics:[/bold]")
            for key, value in result.metrics.items():
                console.print(f"  {key}: {value}")

    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        logger.exception("Training error")
        raise typer.Exit(1)
    finally:
        trainer.cleanup()
        log_memory_usage("After cleanup")


@app.command()
def inference(
    model_path: Path = typer.Argument(..., help="Path to model or adapter"),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Input prompt"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive chat mode"),
    system_prompt: str | None = typer.Option(None, "--system", "-s", help="System prompt"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max new tokens"),
    stream: bool = typer.Option(False, "--stream", help="Stream output")
):
    """Run inference with a fine-tuned model."""
    console.print(f"[bold green]Loading model from {model_path}...[/bold green]")

    # Initialize inference engine
    engine = InferenceEngine(model_path)
    engine.load_model()

    # Generation config
    gen_config = GenerationConfig(
        temperature=temperature,
        max_new_tokens=max_tokens,
        stream=stream
    )

    if interactive:
        # Interactive chat mode
        console.print("[bold]Starting interactive chat mode[/bold]")
        engine.chat(gen_config, system_prompt)
    elif prompt:
        # Single prompt
        console.print(f"\n[bold]Prompt:[/bold] {prompt}")
        console.print("\n[bold]Response:[/bold]")
        response = engine.generate(prompt, gen_config, system_prompt)
        if not stream:
            console.print(response)
    else:
        console.print("[bold red]Please provide a prompt with -p or use -i for interactive mode[/bold red]")
        raise typer.Exit(1)


@app.command()
def upload(
    model_path: Path = typer.Argument(..., help="Path to model or adapter"),
    repo_name: str = typer.Argument(..., help="HuggingFace repository name"),
    base_model: str = typer.Option(None, "--base-model", help="Base model name"),
    dataset_name: str = typer.Option("custom", "--dataset", help="Dataset name"),
    description: str = typer.Option("", "--description", "-d", help="Model description"),
    private: bool = typer.Option(False, "--private", help="Make repository private"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags")
):
    """Upload a model or adapter to HuggingFace Hub."""
    console.print(f"[bold green]Uploading {model_path} to {repo_name}...[/bold green]")

    # Determine base model
    if not base_model:
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            import json
            with open(adapter_config_path) as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path", "unknown")
        else:
            console.print("[bold red]Please specify base model with --base-model[/bold red]")
            raise typer.Exit(1)

    # Create model card
    model_card = ModelCard(
        model_name=repo_name.split("/")[-1],
        base_model=base_model,
        dataset_name=dataset_name,
        description=description,
        tags=tags.split(",") if tags else []
    )

    # Upload
    uploader = ModelUploader()

    try:
        # Check if adapter or full model
        if (model_path / "adapter_config.json").exists():
            url = uploader.upload_adapter(model_path, repo_name, model_card, private)
        else:
            url = uploader.upload_model(model_path, repo_name, model_card, private)

        console.print(f"[bold green]Successfully uploaded to {url}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Upload failed: {e}[/bold red]")
        logger.exception("Upload error")
        raise typer.Exit(1)


@app.command()
def list_datasets(
    qa_path: Path | None = typer.Option(None, "--path", "-p", help="QA output path")
):
    """List available datasets from ArangoDB."""
    qa_path = qa_path or Path(os.getenv("ARANGODB_QA_PATH", "/home/graham/workspace/experiments/arangodb/qa_output"))

    console.print(f"[bold]Searching for datasets in {qa_path}[/bold]")

    if not qa_path.exists():
        console.print(f"[bold red]Path does not exist: {qa_path}[/bold red]")
        raise typer.Exit(1)

    # Find JSONL files
    jsonl_files = list(qa_path.glob("qa_unsloth_*.jsonl"))

    if not jsonl_files:
        console.print("[yellow]No datasets found[/yellow]")
        return

    # Create table
    table = Table(title="Available Datasets")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Modified", style="green")
    table.add_column("Lines", style="yellow")

    for file in sorted(jsonl_files):
        size = file.stat().st_size / (1024 * 1024)  # MB
        modified = file.stat().st_mtime

        # Count lines
        with open(file) as f:
            lines = sum(1 for _ in f)

        from datetime import datetime
        mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")

        table.add_row(
            file.name,
            f"{size:.2f} MB",
            mod_time,
            str(lines)
        )

    console.print(table)



# Add slash command and MCP generation capabilities
add_slash_mcp_commands(app, project_name='unsloth-wip', command_prefix="generate")

def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
