"""
Module: unified_typer_cli.py
Description: Functions for unified typer cli operations

External Dependencies:
- asyncio: [Documentation URL]
- typer: [Documentation URL]
- loguru: [Documentation URL]
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
"""Unified Typer CLI for Unsloth enhanced training pipeline with RunPod support."""

import asyncio
import os
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from ..data.thinking_enhancer import StudentTeacherConfig, ThinkingEnhancer
from ..evaluation.config import EvaluationConfig
from ..evaluation.evaluator import ModelEvaluator
from ..evaluation.litellm_evaluator import JudgeConfig, LiteLLMEvaluator
from ..evaluation.multi_model_evaluator import ModelCandidate, MultiModelEvaluator
from ..inference.generate import GenerationConfig, InferenceEngine
from ..inference.test_suite import InferenceTestSuite, interactive_test_session

# Import all necessary components
from ..pipeline.complete_training_pipeline import CompletePipeline
from ..training.runpod_training_ops import run_training_on_runpod
from ..upload.hub_uploader import HubUploader
from ..validation.model_validator import ModelValidator
from .slash_mcp_mixin import add_slash_mcp_commands

# Create Typer app
app = typer.Typer(
    name="unsloth",
    help="Unsloth Enhanced Training Pipeline CLI - Train LoRA adapters with student-teacher thinking enhancement.",
    add_completion=False
)
console = Console()

# Add slash/MCP commands
add_slash_mcp_commands(app)

# ===== Main Commands =====

@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., unsloth/Phi-3.5-mini-instruct)"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to Q&A dataset"),
    output: Path = typer.Option("./outputs/pipeline", "--output", "-o", help="Output directory"),
    hub_id: str | None = typer.Option(None, "--hub-id", help="HuggingFace Hub model ID for upload"),
    force_runpod: bool = typer.Option(False, "--force-runpod", help="Force training on RunPod"),
    skip_enhancement: bool = typer.Option(False, "--skip-enhancement", help="Skip student-teacher enhancement"),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Maximum samples to process"),
    grokking: bool = typer.Option(False, "--grokking", help="Enable grokking for better generalization")
):
    """Run complete training pipeline (enhancement + training + upload)."""

    console.print("[bold green] Starting Unsloth Training Pipeline[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Dataset: {dataset}")

    pipeline = CompletePipeline(
        model_name=model,
        dataset_path=str(dataset),
        output_dir=str(output),
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


@app.command()
def enhance(
    input: Path = typer.Option(..., "--input", "-i", help="Input Q&A dataset"),
    output: Path = typer.Option(..., "--output", "-o", help="Output enhanced dataset"),
    model: str = typer.Option(..., "--model", "-m", help="Model to use as student"),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Maximum samples to enhance"),
    max_iterations: int = typer.Option(3, "--max-iterations", help="Max iterations per question"),
    batch_size: int = typer.Option(10, "--batch-size", help="Batch size for processing")
):
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
                input_path=input,
                output_path=output,
                max_samples=max_samples
            )

        _print_enhancement_stats(stats)

    asyncio.run(_enhance())


@app.command()
def validate(
    adapter: Path = typer.Option(..., "--adapter", "-a", help="Path to adapter"),
    base_model: str = typer.Option(..., "--base-model", "-b", help="Base model name"),
    prompts: list[str] | None = typer.Option(None, "--prompts", "-p", help="Test prompts"),
    compare_base: bool = typer.Option(False, "--compare-base", help="Compare with base model"),
    dataset: Path | None = typer.Option(None, "--dataset", help="Validation dataset")
):
    """Validate a trained adapter."""

    console.print("[bold yellow] Validating adapter[/bold yellow]")
    console.print(f"Adapter: {adapter}")
    console.print(f"Base Model: {base_model}")

    async def _validate():
        validator = ModelValidator()

        test_prompts = list(prompts) if prompts else None

        with console.status("[bold green]Running validation tests..."):
            results = await validator.validate_adapter(
                adapter_path=adapter,
                base_model=base_model,
                test_prompts=test_prompts,
                validation_dataset=dataset,
                compare_base=compare_base
            )

        _print_validation_results(results)

    asyncio.run(_validate())


@app.command()
def upload(
    adapter: Path = typer.Option(..., "--adapter", "-a", help="Path to adapter"),
    model_id: str = typer.Option(..., "--model-id", "-m", help="HuggingFace model ID"),
    base_model: str = typer.Option(..., "--base-model", "-b", help="Base model name"),
    private: bool = typer.Option(True, "--private/--public", help="Make repository private"),
    tags: list[str] | None = typer.Option(None, "--tags", "-t", help="Additional tags")
):
    """Upload adapter to HuggingFace Hub."""

    console.print("[bold magenta] Uploading to HuggingFace Hub[/bold magenta]")
    console.print(f"Model ID: {model_id}")

    async def _upload():
        uploader = HubUploader()

        with console.status("[bold green]Uploading adapter..."):
            result = await uploader.upload_adapter(
                adapter_path=adapter,
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


@app.command()
def infer(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model or adapter"),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Single prompt to test"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive chat mode"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Generation temperature"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Maximum tokens to generate"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
    system_prompt: str | None = typer.Option(None, "--system", help="System prompt")
):
    """Run inference with a fine-tuned model."""

    console.print("[bold cyan] Running inference[/bold cyan]")
    console.print(f"Model: {model}")

    # Create inference engine
    engine = InferenceEngine(
        model_path=model,
        load_in_4bit=True
    )

    # Load model
    with console.status("[bold green]Loading model..."):
        engine.load_model()

    # Generation config
    config = GenerationConfig(
        temperature=temperature,
        max_new_tokens=max_tokens,
        stream=stream
    )

    if interactive:
        # Interactive mode
        console.print("\n[bold green]Starting interactive chat mode[/bold green]")
        console.print("Type 'exit' to quit\n")
        engine.chat(config, system_prompt)

    elif prompt:
        # Single prompt mode
        console.print(f"\n[bold]Prompt:[/bold] {prompt}")
        console.print("\n[bold]Response:[/bold] ", end="")

        response = engine.generate(prompt, config, system_prompt)
        if not stream:
            console.print(response)

    else:
        console.print("[red]Please provide a prompt with -p or use -i for interactive mode[/red]")
        raise typer.Exit(1)


@app.command()
def test_inference(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model or adapter"),
    output: Path = typer.Option("./inference_test_results", "--output", "-o", help="Output directory"),
    categories: list[str] | None = typer.Option(None, "--category", "-c", help="Test categories to run"),
    custom_tests: Path | None = typer.Option(None, "--custom-tests", help="JSON file with custom tests"),
    judge_model: str = typer.Option("gpt-4", "--judge", help="Judge model for evaluation"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip judge evaluation"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive test mode")
):
    """Run comprehensive inference tests on a model."""

    if interactive:
        # Interactive testing mode
        console.print("[bold cyan] Interactive Test Mode[/bold cyan]")
        asyncio.run(interactive_test_session(str(model)))
        return

    console.print("[bold cyan] Running Inference Test Suite[/bold cyan]")
    console.print(f"Model: {model}")

    # Create test suite
    test_suite = InferenceTestSuite(
        model_path=str(model),
        output_dir=str(output),
        use_judge=not no_judge,
        judge_model=judge_model
    )

    # Add custom tests if provided
    if custom_tests:
        test_suite.add_custom_tests(str(custom_tests))

    # Run tests
    async def _run_tests():
        results = await test_suite.run_tests(categories=categories)

        # Show report location
        report_path = output / "test_report.html"
        if report_path.exists():
            console.print(f"\n[bold cyan] Test Report:[/bold cyan] {report_path}")
            console.print("Open in browser to view detailed results.")

    asyncio.run(_run_tests())


# ===== RunPod Commands Group =====

runpod_app = typer.Typer(help="RunPod training commands")
app.add_typer(runpod_app, name="runpod")


@runpod_app.command("train")
def runpod_train(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Dataset path"),
    hub_id: str | None = typer.Option(None, "--hub-id", help="HuggingFace Hub ID for upload"),
    epochs: int = typer.Option(3, "--epochs", help="Training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", help="Learning rate")
):
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
                dataset_path=dataset,
                training_config=training_config,
                hub_model_id=hub_id
            )

        if result["status"] == "success":
            console.print("\n[bold green] Training completed![/bold green]")
            console.print(f"Adapter: {result['adapter_path']}")
        else:
            console.print(f"\n[bold red] Training failed: {result.get('error')}[/bold red]")

    asyncio.run(_train())


@runpod_app.command("list")
def runpod_list():
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


@runpod_app.command("stop")
def runpod_stop(
    pod_id: str = typer.Argument(..., help="Pod ID to stop"),
    terminate: bool = typer.Option(False, "--terminate", help="Terminate instead of stop")
):
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


@runpod_app.command("gpus")
def runpod_gpus():
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


# ===== Evaluation Command =====

@app.command()
def evaluate(
    base_model: str = typer.Option(..., "--base-model", "-b", help="Base model path or HF ID"),
    lora_model: str | None = typer.Option(None, "--lora-model", "-l", help="LoRA adapter path"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Evaluation dataset path"),
    output: Path = typer.Option("./evaluation_results", "--output", "-o", help="Output directory"),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Max samples to evaluate"),
    judge_model: str = typer.Option("gpt-4", "--judge-model", help="Judge model for evaluation"),
    generate_html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML dashboard"),
    metrics: list[str] | None = typer.Option(None, "--metrics", "-m", help="Specific metrics to run"),
    compare: bool = typer.Option(True, "--compare/--no-compare", help="Compare base vs LoRA")
):
    """Evaluate model accuracy before and after LoRA training."""

    console.print("[bold cyan] Starting Model Evaluation[/bold cyan]")
    console.print(f"Base Model: {base_model}")
    if lora_model:
        console.print(f"LoRA Model: {lora_model}")

    # Configure evaluation
    eval_config = EvaluationConfig(
        dataset_path=str(dataset),
        base_model_path=base_model,
        lora_model_path=lora_model,
        output_dir=str(output),
        max_samples=max_samples,
        use_judge_model=True,
        judge_config={"model_name": judge_model},
        generate_html_report=generate_html,
        compare_models=compare and lora_model is not None
    )

    # Filter metrics if specified
    if metrics:
        eval_config.metrics = [
            m for m in eval_config.metrics
            if m.name in metrics
        ]

    # Run evaluation
    evaluator = ModelEvaluator(eval_config)

    try:
        with console.status("[bold green]Running evaluation..."):
            results = asyncio.run(evaluator.evaluate_all())

        # Print summary
        console.print("\n[bold green] Evaluation Complete![/bold green]")

        # Show results table
        if results.get("comparison"):
            comparison = results["comparison"]

            # Summary stats
            table = Table(title="Evaluation Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Base Model", style="yellow")
            table.add_column("LoRA Model", style="green")
            table.add_column("Change", style="magenta")

            # Add improvements
            for metric, data in comparison.get("improvements", {}).items():
                table.add_row(
                    metric,
                    f"{data['base']:.4f}",
                    f"{data['lora']:.4f}",
                    f"+{data['improvement_pct']:.1f}%"
                )

            # Add regressions
            for metric, data in comparison.get("regressions", {}).items():
                table.add_row(
                    metric,
                    f"{data['base']:.4f}",
                    f"{data['lora']:.4f}",
                    f"-{data['regression_pct']:.1f}%"
                )

            console.print(table)

            # Overall recommendation
            summary = comparison.get("summary", {})
            console.print(f"\n[bold]Recommendation:[/bold] {summary.get('recommendation', 'N/A')}")
            console.print(f"Improvements: {summary.get('total_improvements', 0)}")
            console.print(f"Regressions: {summary.get('total_regressions', 0)}")

            # Judge score improvement
            if "judge_score_improvement" in summary:
                judge_imp = summary["judge_score_improvement"]
                console.print(f"\nJudge Score: {judge_imp['base']:.2f} → {judge_imp['lora']:.2f} ({judge_imp['improvement_pct']:+.1f}%)")

        else:
            # Single model evaluation
            base_results = results.get("base_model", {})
            metrics = base_results.get("metrics", {})

            if metrics:
                table = Table(title="Model Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")

                for metric, value in metrics.items():
                    table.add_row(metric, f"{value:.4f}")

                console.print(table)

            # Judge scores
            judge_scores = base_results.get("judge_scores", {})
            if judge_scores:
                console.print(f"\n[bold]Judge Overall Score:[/bold] {judge_scores.get('overall_score', 0):.2f}/10")

                if judge_scores.get("criteria_scores"):
                    console.print("\n[bold]Criteria Scores:[/bold]")
                    for criterion, score in judge_scores["criteria_scores"].items():
                        console.print(f"  {criterion}: {score:.2f}/10")

        # Dashboard location
        if generate_html:
            dashboard_path = Path(output) / "evaluation_dashboard.html"
            if dashboard_path.exists():
                console.print(f"\n[bold cyan] Dashboard:[/bold cyan] {dashboard_path}")
                console.print("Open in browser to view interactive charts and detailed results.")

        # Save location
        console.print(f"\n[bold]Results saved to:[/bold] {output}")

    except Exception as e:
        console.print(f"[red] Evaluation failed: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def multi_evaluate(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Evaluation dataset path"),
    output: Path = typer.Option("./model_selection_results", "--output", "-o", help="Output directory"),
    max_samples: int | None = typer.Option(100, "--max-samples", help="Max samples per model"),
    target_accuracy: float = typer.Option(0.8, "--target-accuracy", help="Minimum accuracy threshold"),
    judge_model: str = typer.Option("gpt-4", "--judge-model", help="Judge model for evaluation"),
    custom_models: list[str] | None = typer.Option(None, "--models", "-m", help="Custom model IDs to test"),
    size_limit: float | None = typer.Option(None, "--size-limit", help="Max model size in billions"),
    metrics: list[str] | None = typer.Option(None, "--metrics", help="Specific metrics to run")
):
    """Evaluate multiple models to find the smallest accurate one."""

    console.print("[bold cyan] Multi-Model Evaluation[/bold cyan]")
    console.print(f"Finding smallest model with accuracy ≥ {target_accuracy}")

    # Prepare model candidates
    if custom_models:
        # Parse custom models (assume format: "model_id:size_in_B")
        candidates = []
        for model_spec in custom_models:
            if ":" in model_spec:
                model_id, size = model_spec.split(":", 1)
                candidates.append(ModelCandidate(
                    model_id=model_id,
                    parameter_count=float(size),
                    min_gpu_memory=int(float(size) * 6),  # Rough estimate
                ))
            else:
                # Use default size estimates
                candidates.append(ModelCandidate(
                    model_id=model_spec,
                    parameter_count=3.0,  # Default assumption
                    min_gpu_memory=16
                ))
    else:
        # Use predefined candidates
        candidates = MultiModelEvaluator.RECOMMENDED_MODELS

        # Filter by size limit if specified
        if size_limit:
            candidates = [c for c in candidates if c.parameter_count <= size_limit]

    # Create evaluator
    evaluator = MultiModelEvaluator(
        dataset_path=str(dataset),
        output_dir=str(output),
        max_samples=max_samples,
        target_accuracy=target_accuracy,
        judge_model=judge_model
    )

    # Run evaluation
    try:
        results = asyncio.run(evaluator.evaluate_all_models(
            model_candidates=candidates,
            metrics=metrics
        ))

        # Show dashboard location
        dashboard_path = output / "multi_model_dashboard.html"
        if dashboard_path.exists():
            console.print(f"\n[bold cyan] Dashboard:[/bold cyan] {dashboard_path}")
            console.print("Open in browser to view detailed comparison and recommendations.")

    except Exception as e:
        console.print(f"[red] Multi-model evaluation failed: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def universal_evaluate(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Evaluation dataset path"),
    output: Path = typer.Option("./universal_eval_results", "--output", "-o", help="Output directory"),
    max_samples: int | None = typer.Option(100, "--max-samples", help="Max samples per model"),
    target_accuracy: float = typer.Option(0.8, "--target-accuracy", help="Minimum accuracy threshold"),
    models: list[str] | None = typer.Option(None, "--model", "-m", help="Specific models to test (can specify multiple)"),
    include_local: bool = typer.Option(True, "--local/--no-local", help="Include local Ollama models"),
    include_cloud: bool = typer.Option(True, "--cloud/--no-cloud", help="Include cloud models"),
    size_limit: float | None = typer.Option(None, "--size-limit", help="Max model size in billions"),
    judge_model: str = typer.Option("gpt-4", "--judge", help="Judge model for evaluation"),
    judge_criteria: list[str] | None = typer.Option(None, "--criteria", "-c", help="Evaluation criteria"),
    list_models: bool = typer.Option(False, "--list", help="List available models and exit")
):
    """Universal model evaluation using LiteLLM (supports Ollama, OpenAI, Anthropic, OpenRouter)."""

    # Configure judge
    judge_config = JudgeConfig(
        model=judge_model,
        criteria=judge_criteria or ["accuracy", "relevance", "coherence", "completeness", "conciseness"]
    )

    # Create evaluator
    evaluator = LiteLLMEvaluator(
        dataset_path=str(dataset) if not list_models else "",
        output_dir=str(output),
        max_samples=max_samples,
        target_accuracy=target_accuracy,
        judge_config=judge_config
    )

    # Just list models if requested
    if list_models:
        console.print("[bold cyan] Available Models via LiteLLM[/bold cyan]\n")

        all_models = evaluator.list_available_models()

        # Group by provider
        by_provider = {}
        for model in all_models:
            if model.provider not in by_provider:
                by_provider[model.provider] = []
            by_provider[model.provider].append(model)

        # Display by provider
        for provider, models in sorted(by_provider.items()):
            table = Table(title=f"{provider.upper()} Models")
            table.add_column("Model ID", style="cyan")
            table.add_column("Size (B)", style="magenta")
            table.add_column("Type", style="green")

            for model in sorted(models, key=lambda x: x.estimated_params or 999):
                size = f"{model.estimated_params}" if model.estimated_params else "?"
                model_type = "Local" if model.is_local else "Cloud"
                table.add_row(model.model_id, size, model_type)

            console.print(table)
            console.print()

        return

    # Run evaluation
    console.print("[bold cyan] Universal Model Evaluation via LiteLLM[/bold cyan]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Judge: {judge_model} with criteria: {', '.join(judge_config.criteria)}")

    try:
        results = asyncio.run(evaluator.evaluate_models(
            models=models,
            include_local=include_local,
            include_cloud=include_cloud,
            size_limit=size_limit
        ))

        # Show dashboard location
        dashboard_path = output / "multi_model_dashboard.html"
        if dashboard_path.exists():
            console.print(f"\n[bold cyan] Dashboard:[/bold cyan] {dashboard_path}")
            console.print("Open in browser for detailed results with judge evaluations.")

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red] Evaluation failed: {str(e)}[/red]")
        logger.exception("Universal evaluation error")
        raise typer.Exit(1)


# ===== Utility Commands =====

@app.command()
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

[bold yellow]7. Evaluate models (before/after training):[/bold yellow]
   unsloth evaluate --base-model unsloth/Phi-3.5-mini-instruct \\
                    --lora-model ./outputs/adapter \\
                    --dataset qa_test.jsonl

[bold yellow]8. Find smallest accurate model:[/bold yellow]
   unsloth multi-evaluate --dataset qa_test.jsonl \\
                          --target-accuracy 0.85 \\
                          --size-limit 5.0

[bold yellow]9. Universal evaluation (Ollama/OpenAI/Anthropic/OpenRouter):[/bold yellow]
   # List all available models
   unsloth universal-evaluate --list
   
   # Evaluate specific models with judge
   unsloth universal-evaluate --dataset qa_test.jsonl \\
                              --model ollama/phi3.5 \\
                              --model gpt-4o-mini \\
                              --judge claude-3-opus

[bold yellow]10. RunPod management:[/bold yellow]
   unsloth runpod list
   unsloth runpod gpus
   unsloth runpod stop <pod_id>

For more help: unsloth --help
"""
    console.print(guide)


@app.command()
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
    # Check for API keys
    required_keys = ["HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        console.print(f"[yellow]⚠️  Missing environment variables: {', '.join(missing_keys)}[/yellow]")
        console.print("Some features may not work without these keys.\n")

    app()


if __name__ == "__main__":
    main()
