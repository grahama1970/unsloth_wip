"""RunPod CLI commands for Unsloth training."""
Module: runpod_commands.py
Description: Functions for runpod commands operations

import asyncio
from pathlib import Path

import click
from loguru import logger

from ..training.runpod_training_ops import RunPodTrainingOps, run_training_on_runpod


@click.group()
def runpod():
    """RunPod training commands."""
    pass


@runpod.command()
@click.option("--model", "-m", required=True, help="Model name (e.g., unsloth/Phi-3.5-mini-instruct)")
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Path to dataset")
@click.option("--output", "-o", default="./outputs/runpod", help="Output directory")
@click.option("--hub-id", help="HuggingFace Hub model ID for upload")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--batch-size", default=4, help="Training batch size")
@click.option("--learning-rate", default=2e-4, help="Learning rate")
@click.option("--r", default=16, help="LoRA rank")
@click.option("--alpha", default=16, help="LoRA alpha")
@click.option("--enhance-thinking", is_flag=True, help="Apply student-teacher enhancement")
def train(
    model: str,
    dataset: str,
    output: str,
    hub_id: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    r: int,
    alpha: int,
    enhance_thinking: bool
):
    """Train a model on RunPod infrastructure."""

    # Create training configuration
    training_config = {
        "model_name": model,
        "max_seq_length": 2048,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.03,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "gradient_checkpointing": True,
        "hub_model_id": hub_id
    }

    # Run training
    async def _train():
        if enhance_thinking:
            logger.info("Student-teacher enhancement would be applied before training")
            # Enhancement would happen here

        result = await run_training_on_runpod(
            model_name=model,
            dataset_path=Path(dataset),
            training_config=training_config,
            hub_model_id=hub_id
        )

        if result["status"] == "success":
            logger.success(f"Training completed! Adapter saved to: {result['adapter_path']}")
            if hub_id:
                logger.success(f"Model uploaded to: https://huggingface.co/{hub_id}")
        else:
            logger.error(f"Training failed: {result.get('error', 'Unknown error')}")

    asyncio.run(_train())


@runpod.command()
def list_pods():
    """List all RunPod pods."""
    import runpod

    pods = runpod.get_pods()

    if not pods:
        click.echo("No pods found")
        return

    click.echo("\nRunPod Pods:")
    click.echo("-" * 80)

    for pod in pods:
        status = pod.get("desiredStatus", "UNKNOWN")
        gpu = pod.get("machine", {}).get("gpuDisplayName", "N/A")

        click.echo(f"ID: {pod['id']}")
        click.echo(f"Name: {pod['name']}")
        click.echo(f"Status: {status}")
        click.echo(f"GPU: {gpu}")
        click.echo(f"Created: {pod.get('createdAt', 'N/A')}")
        click.echo("-" * 80)


@runpod.command()
@click.argument("pod_id")
@click.option("--terminate", is_flag=True, help="Terminate instead of stop")
def stop(pod_id: str, terminate: bool):
    """Stop a RunPod pod."""
    import runpod

    try:
        if terminate:
            runpod.terminate_pod(pod_id)
            click.echo(f"Pod {pod_id} terminated")
        else:
            runpod.stop_pod(pod_id)
            click.echo(f"Pod {pod_id} stopped")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@runpod.command()
def gpus():
    """List available GPUs on RunPod."""
    import runpod

    available_gpus = runpod.get_gpus()

    if not available_gpus:
        click.echo("No GPUs available")
        return

    click.echo("\nAvailable GPUs:")
    click.echo("-" * 60)

    for gpu in available_gpus:
        name = gpu.get("displayName", "Unknown")
        memory = gpu.get("memoryInGb", 0)
        price = gpu.get("securePrice", 0)

        click.echo(f"Name: {name}")
        click.echo(f"Memory: {memory} GB")
        click.echo(f"Price: ${price:.2f}/hr")
        click.echo(f"ID: {gpu.get('id', 'N/A')}")
        click.echo("-" * 60)


@runpod.command()
@click.option("--model-size", "-s", type=click.Choice(["7b", "13b", "30b", "70b"]),
              default="7b", help="Model size for GPU recommendation")
def recommend_gpu(model_size: str):
    """Recommend GPUs for a model size."""

    recommendations = {
        "7b": {
            "gpus": ["RTX 4090", "RTX A6000", "A100 PCIe"],
            "min_memory": 24,
            "notes": "Can run efficiently on consumer GPUs"
        },
        "13b": {
            "gpus": ["A100 PCIe", "A100 SXM", "RTX A6000"],
            "min_memory": 40,
            "notes": "Requires professional GPUs"
        },
        "30b": {
            "gpus": ["A100 SXM 80GB", "H100 PCIe"],
            "min_memory": 80,
            "notes": "Requires high-end datacenter GPUs"
        },
        "70b": {
            "gpus": ["H100 PCIe", "H100 SXM", "H100 NVL"],
            "min_memory": 80,
            "notes": "Requires latest generation GPUs"
        }
    }

    rec = recommendations[model_size]

    click.echo(f"\nRecommended GPUs for {model_size} model:")
    click.echo("-" * 50)
    click.echo(f"Minimum Memory: {rec['min_memory']} GB")
    click.echo(f"Recommended GPUs: {', '.join(rec['gpus'])}")
    click.echo(f"Notes: {rec['notes']}")


@runpod.command()
@click.argument("pod_id")
async def monitor(pod_id: str):
    """Monitor a running training pod."""

    ops = RunPodTrainingOps()
    ops.pod = {"id": pod_id}
    ops.api_base = f"https://{pod_id}-8888.proxy.runpod.net"

    click.echo(f"Monitoring pod {pod_id}...")
    click.echo("Press Ctrl+C to stop monitoring")

    try:
        async for progress in ops.monitor_training():
            status = progress.get("status", "unknown")
            timestamp = progress.get("timestamp", "")

            click.clear()
            click.echo(f"Pod ID: {pod_id}")
            click.echo(f"Status: {status}")
            click.echo(f"Last Update: {timestamp}")

            if "epoch" in progress:
                click.echo(f"Epoch: {progress['epoch']}")
            if "step" in progress:
                click.echo(f"Step: {progress['step']}")
            if "loss" in progress:
                click.echo(f"Loss: {progress['loss']:.4f}")

            if status in ["completed", "failed"]:
                if status == "completed":
                    click.echo("\nTraining completed successfully!")
                else:
                    click.echo(f"\nTraining failed: {progress.get('error', 'Unknown error')}")
                break

    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped")


# Add to main CLI
def add_runpod_commands(cli):
    """Add RunPod commands to main CLI."""
    cli.add_command(runpod)
