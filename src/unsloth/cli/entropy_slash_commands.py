"""
Module: entropy_slash_commands.py
Description: Entropy-aware slash commands for enhanced training workflows

External Dependencies:
- click: https://click.palletsprojects.com/
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> cli = create_entropy_commands()
>>> cli.main(["entropy-analyze", "--model", "phi-3.5", "--dataset", "qa.jsonl"])

Expected Output:
>>> # Entropy analysis report with high-entropy regions identified

Example Usage:
>>> from unsloth.cli.entropy_slash_commands import create_entropy_commands
>>> entropy_cli = create_entropy_commands()
"""

import click
import json
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

from loguru import logger

from unsloth.training.entropy_utils import calculate_token_entropy, get_entropy_weight
from unsloth.data.entropy_aware_thinking_enhancer import EntropyAwareThinkingEnhancer
from unsloth.evaluation.entropy_evaluator import EntropyAwareEvaluator
from unsloth.utils.tensorboard_verifier import TensorBoardVerifier
from unsloth.core.enhanced_config import EnhancedTrainingConfig


@click.group()
def entropy():
    """Entropy-aware training commands."""
    pass


@entropy.command()
@click.option("--model", required=True, help="Model name or path")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Dataset to analyze")
@click.option("--output", "-o", type=click.Path(), help="Output file for analysis")
@click.option("--sample-size", default=1000, help="Number of examples to analyze")
@click.option("--threshold", default=0.7, type=float, help="Entropy threshold for high-entropy tokens")
def analyze(model: str, dataset: str, output: Optional[str], sample_size: int, threshold: float):
    """Analyze dataset entropy patterns."""
    logger.info(f"Analyzing entropy for {dataset} with {model}")
    
    # Create evaluator
    evaluator = EntropyAwareEvaluator(
        model_name=model,
        entropy_threshold=threshold
    )
    
    # Load dataset
    import datasets
    ds = datasets.load_dataset("json", data_files=dataset)["train"]
    
    # Sample if needed
    if len(ds) > sample_size:
        ds = ds.select(range(sample_size))
        
    # Analyze entropy
    analysis = evaluator.analyze_dataset_entropy(ds)
    
    # Display results
    click.echo("\n Entropy Analysis Results:")
    click.echo(f"Average entropy: {analysis['summary']['avg_entropy']:.3f}")
    click.echo(f"High-entropy ratio: {analysis['summary']['high_entropy_ratio']:.2%}")
    click.echo(f"Entropy by position: {analysis['patterns']['by_position']}")
    
    # Save if requested
    if output:
        Path(output).write_text(json.dumps(analysis, indent=2))
        click.echo(f"\n Analysis saved to: {output}")
        

@entropy.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input dataset")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output enhanced dataset")
@click.option("--model", help="Model for enhancement (auto-selected if not specified)")
@click.option("--max-iterations", default=3, help="Max thinking iterations")
@click.option("--entropy-threshold", default=0.7, type=float, help="Threshold for enhancement")
@click.option("--batch-size", default=10, help="Processing batch size")
def enhance(
    input: str,
    output: str,
    model: Optional[str],
    max_iterations: int,
    entropy_threshold: float,
    batch_size: int
):
    """Enhance dataset with entropy-aware thinking."""
    logger.info(f"Enhancing {input} with entropy-aware thinking")
    
    # Create config
    from unsloth.core.enhanced_config import EntropyAwareTeacherConfig
    config = EntropyAwareTeacherConfig(
        max_iterations=max_iterations,
        convergence_threshold=0.1,
        entropy_threshold=entropy_threshold
    )
    
    # Create enhancer
    enhancer = EntropyAwareThinkingEnhancer(config)
    
    # Load dataset
    import datasets
    ds = datasets.load_dataset("json", data_files=input)["train"]
    
    # Process in batches
    enhanced_examples = []
    total = len(ds)
    
    with click.progressbar(range(0, total, batch_size), label="Enhancing") as bar:
        for i in bar:
            batch = ds[i:i+batch_size]
            
            # Enhance batch
            for example in batch:
                enhanced = asyncio.run(enhancer.enhance_example(example))
                enhanced_examples.append(enhanced)
                
    # Save enhanced dataset
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for example in enhanced_examples:
            f.write(json.dumps(example) + "\n")
            
    click.echo(f"\n Enhanced {len(enhanced_examples)} examples")
    click.echo(f" Saved to: {output}")
    

@entropy.command()
@click.option("--log-dir", required=True, type=click.Path(exists=True), help="TensorBoard log directory")
@click.option("--output", "-o", type=click.Path(), help="Output report file")
@click.option("--screenshot", is_flag=True, help="Capture screenshots (requires Playwright)")
@click.option("--port", default=6006, help="TensorBoard port")
def verify(log_dir: str, output: Optional[str], screenshot: bool, port: int):
    """Verify entropy-aware training progress."""
    logger.info(f"Verifying training logs in {log_dir}")
    
    # Create verifier
    verifier = TensorBoardVerifier(
        log_dir=log_dir,
        port=port,
        screenshot_dir=str(Path(log_dir) / "screenshots") if screenshot else None
    )
    
    # Analyze logs
    analysis = verifier.analyze_training_logs()
    
    # Display results
    click.echo("\n Training Verification:")
    click.echo(f"Status: {analysis['status']}")
    click.echo(f"Loss trend: {analysis.get('loss_trend', 'N/A')}")
    
    if analysis.get("entropy_metrics"):
        click.echo("\n Entropy Metrics:")
        for metric, values in analysis["entropy_metrics"].items():
            click.echo(f"  {metric}: {values.get('trend', 'N/A')} "
                      f"({values.get('initial', 0):.3f} → {values.get('final', 0):.3f})")
            
    # Warnings
    if analysis.get("warnings"):
        click.echo("\n⚠️  Warnings:")
        for warning in analysis["warnings"]:
            click.echo(f"  - {warning}")
            
    # Generate full report if requested
    if output or screenshot:
        async def generate_report():
            report = await verifier.verify_training_progress()
            if output:
                Path(output).write_text(json.dumps(report, indent=2))
                click.echo(f"\n Report saved to: {output}")
            return report
            
        report = asyncio.run(generate_report())
        
        if screenshot and report.get("screenshots"):
            click.echo(f"\n Screenshots saved: {len(report['screenshots'])}")
            

@entropy.command()
@click.option("--model", required=True, help="Model to train")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Training dataset")
@click.option("--output", "-o", default="./outputs", help="Output directory")
@click.option("--weight-function", type=click.Choice(["linear", "exponential", "sigmoid"]), 
              default="exponential", help="Entropy weight function")
@click.option("--entropy-scale", default=1.0, type=float, help="Entropy weight scale")
@click.option("--min-weight", default=1.0, type=float, help="Minimum entropy weight")
@click.option("--max-weight", default=2.0, type=float, help="Maximum entropy weight")
@click.option("--epochs", default=3, type=int, help="Training epochs")
@click.option("--batch-size", default=4, type=int, help="Batch size")
@click.option("--learning-rate", default=2e-4, type=float, help="Learning rate")
@click.option("--hub-upload", is_flag=True, help="Upload to HuggingFace Hub")
@click.option("--hub-id", help="HuggingFace Hub model ID")
def train(
    model: str,
    dataset: str,
    output: str,
    weight_function: str,
    entropy_scale: float,
    min_weight: float,
    max_weight: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hub_upload: bool,
    hub_id: Optional[str]
):
    """Run entropy-aware training."""
    logger.info(f"Starting entropy-aware training for {model}")
    
    # Create config
    config = EnhancedTrainingConfig(
        model_name=model,
        dataset_path=dataset,
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        # Entropy-specific settings
        entropy_aware_training=True,
        entropy_weight_function=weight_function,
        entropy_scale=entropy_scale,
        entropy_min_weight=min_weight,
        entropy_max_weight=max_weight,
        # HuggingFace settings
        push_to_hub=hub_upload,
        hub_model_id=hub_id or f"{model.split('/')[-1]}-entropy"
    )
    
    # Run training
    from unsloth.training.entropy_aware_trainer import create_entropy_aware_trainer
    
    async def run_training():
        trainer = await create_entropy_aware_trainer(config)
        
        # Train
        trainer.train()
        
        # Save
        trainer.save_model(output)
        
        # Upload if requested
        if hub_upload:
            from unsloth.upload.entropy_aware_hub_uploader import EntropyAwareHubUploader
            uploader = EntropyAwareHubUploader()
            
            # Get entropy metrics
            entropy_metrics = trainer.callback_handler.callbacks[0].get_entropy_summary()
            
            result = await uploader.upload_entropy_enhanced_model(
                adapter_path=Path(output),
                model_id=config.hub_model_id,
                base_model=model,
                training_metrics=entropy_metrics
            )
            
            click.echo(f"\n Model uploaded: {result['url']}")
            
    asyncio.run(run_training())
    click.echo(f"\n Training complete! Adapter saved to: {output}")
    

@entropy.command()
@click.option("--adapter", required=True, type=click.Path(exists=True), help="Adapter path")
@click.option("--base-model", required=True, help="Base model name")
@click.option("--test-dataset", type=click.Path(exists=True), help="Test dataset")
@click.option("--compare-baseline", is_flag=True, help="Compare with baseline model")
@click.option("--output", "-o", type=click.Path(), help="Output evaluation report")
def evaluate(
    adapter: str,
    base_model: str,
    test_dataset: Optional[str],
    compare_baseline: bool,
    output: Optional[str]
):
    """Evaluate entropy-aware model."""
    logger.info(f"Evaluating {adapter} on {base_model}")
    
    # Create evaluator
    evaluator = EntropyAwareEvaluator(
        model_name=base_model,
        adapter_path=adapter
    )
    
    # Load test data if provided
    if test_dataset:
        import datasets
        test_ds = datasets.load_dataset("json", data_files=test_dataset)["train"]
    else:
        # Use default test prompts
        test_ds = [
            {"prompt": "What is machine learning?"},
            {"prompt": "Explain quantum computing in simple terms."},
            {"prompt": "How does photosynthesis work?"},
        ]
        
    # Evaluate
    results = evaluator.evaluate_model(test_ds)
    
    # Display results
    click.echo("\n Evaluation Results:")
    click.echo(f"Average entropy: {results['entropy_metrics']['avg_entropy']:.3f}")
    click.echo(f"High-entropy improvement: {results['entropy_metrics']['high_entropy_improvement']:.2%}")
    
    if results.get("accuracy_metrics"):
        click.echo(f"Accuracy: {results['accuracy_metrics']['accuracy']:.2%}")
        
    # Compare with baseline if requested
    if compare_baseline:
        baseline_evaluator = EntropyAwareEvaluator(model_name=base_model)
        baseline_results = baseline_evaluator.evaluate_model(test_ds)
        
        click.echo("\n Comparison with Baseline:")
        click.echo(f"Entropy reduction: {baseline_results['entropy_metrics']['avg_entropy'] - results['entropy_metrics']['avg_entropy']:.3f}")
        
    # Save report if requested
    if output:
        Path(output).write_text(json.dumps(results, indent=2))
        click.echo(f"\n Report saved to: {output}")
        

@entropy.command()
@click.option("--model-size", required=True, help="Model size (e.g., 7B, 13B, 70B)")
@click.option("--dataset-size", required=True, type=int, help="Dataset size in examples")
@click.option("--epochs", default=3, type=int, help="Training epochs")
@click.option("--compare-providers", is_flag=True, help="Compare RunPod vs Vertex costs")
def estimate_cost(model_size: str, dataset_size: int, epochs: int, compare_providers: bool):
    """Estimate training costs with entropy-aware overhead."""
    from runpod_ops import CostCalculator, InstanceOptimizer
    
    calculator = CostCalculator()
    optimizer = InstanceOptimizer()
    
    # Estimate tokens (with entropy overhead)
    avg_tokens_per_example = 512
    entropy_overhead = 1.2  # 20% overhead for entropy calculations
    total_tokens = int(dataset_size * avg_tokens_per_example * epochs * entropy_overhead)
    
    click.echo(f"\n Cost Estimation for Entropy-Aware Training:")
    click.echo(f"Model: {model_size}")
    click.echo(f"Dataset: {dataset_size:,} examples")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Total tokens (with overhead): {total_tokens:,}")
    
    if compare_providers:
        # Compare all providers
        comparison = calculator.compare_providers(model_size, total_tokens)
        
        click.echo("\n Provider Comparison:")
        for i, (provider, info) in enumerate(list(comparison.items())[:5]):
            click.echo(f"\n{i+1}. {provider}")
            click.echo(f"   Cost: ${info['total_cost']:.2f}")
            click.echo(f"   Time: {info['processing_time_hours']:.1f} hours")
            click.echo(f"   $/1K tokens: ${info['cost_per_1k_tokens']:.4f}")
    else:
        # Just RunPod optimization
        config = optimizer.optimize_for_training(
            model_size,
            dataset_size,
            epochs
        )
        
        click.echo(f"\n Recommended Configuration:")
        click.echo(f"GPU: {config['gpu_count']}x {config['gpu_type']}")
        click.echo(f"Batch size: {config['batch_size']}")
        click.echo(f"Estimated time: {config['estimated_hours']:.1f} hours")
        click.echo(f"Estimated cost: ${config['estimated_cost']:.2f}")
        

def create_entropy_commands() -> click.Group:
    """Create entropy command group."""
    return entropy


def register_entropy_slash_commands(cli: click.Group):
    """Register entropy commands with main CLI."""
    cli.add_command(entropy)
    
    # Also add shortcuts at top level
    cli.add_command(analyze, name="entropy-analyze")
    cli.add_command(enhance, name="entropy-enhance")
    cli.add_command(train, name="entropy-train")
    cli.add_command(verify, name="entropy-verify")
    cli.add_command(evaluate, name="entropy-evaluate")
    cli.add_command(estimate_cost, name="entropy-cost")


# Slash command examples that would be generated
ENTROPY_SLASH_COMMANDS = """
# Entropy-Aware Slash Commands:

/unsloth-entropy-analyze      # Analyze dataset entropy patterns
/unsloth-entropy-enhance      # Enhance dataset with thinking
/unsloth-entropy-train        # Run entropy-aware training
/unsloth-entropy-verify       # Verify training with TensorBoard
/unsloth-entropy-evaluate     # Evaluate entropy-aware model
/unsloth-entropy-cost         # Estimate training costs

# Advanced entropy commands:
/unsloth-entropy-analyze --model phi-3.5 --dataset qa.jsonl --threshold 0.8
/unsloth-entropy-enhance --input raw.jsonl --output enhanced.jsonl --max-iterations 5
/unsloth-entropy-train --model llama-7b --dataset train.jsonl --weight-function sigmoid
/unsloth-entropy-verify --log-dir ./logs --screenshot --output report.json
"""


# Validation
if __name__ == "__main__":
    # Test command creation
    entropy_cli = create_entropy_commands()
    
    print("Entropy-Aware Commands")
    print("=" * 50)
    
    # List commands
    for name, cmd in entropy_cli.commands.items():
        print(f"\n{name}: {cmd.help}")
        
        # Show options
        for param in cmd.params:
            if hasattr(param, 'help'):
                print(f"  --{param.name}: {param.help}")
                
    print(ENTROPY_SLASH_COMMANDS)
    print("\n Module validation passed")