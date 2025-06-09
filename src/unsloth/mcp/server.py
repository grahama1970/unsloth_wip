"""
unsloth_wip FastMCP Server
Module: server.py
Description: Functions for server operations

Granger standard MCP server implementation for unsloth_wip.
"""

import os
from pathlib import Path

from fastmcp import FastMCP

from unsloth.mcp.prompts import get_prompt_registry
from unsloth.mcp.unsloth_prompts import register_all_prompts

# Initialize server
mcp = FastMCP("unsloth_wip")
mcp.description = "unsloth_wip - Granger spoke module"

# Register prompts
register_all_prompts()
prompt_registry = get_prompt_registry()


# =============================================================================
# PROMPTS - Required for Granger standard
# =============================================================================

@mcp.prompt()
async def capabilities() -> str:
    """List all MCP server capabilities"""
    return await prompt_registry.execute("unsloth_wip:capabilities")


@mcp.prompt()
async def help(context: str = None) -> str:
    """Get context-aware help"""
    return await prompt_registry.execute("unsloth_wip:help", context=context)


@mcp.prompt()
async def quick_start() -> str:
    """Quick start guide for new users"""
    return await prompt_registry.execute("unsloth_wip:quick-start")


# =============================================================================
# TOOLS - Add your existing tools here
# =============================================================================

@mcp.tool()
async def train_model(
    model_name: str,
    dataset_path: str,
    output_dir: str = "./outputs/pipeline",
    hub_model_id: str = None,
    use_runpod: bool = False,
    skip_enhancement: bool = False,
    max_samples: int = None,
    use_grokking: bool = False,
    use_dapo: bool = False
) -> dict:
    """Run complete training pipeline with optional enhancements.
    
    Args:
        model_name: Model to fine-tune (e.g., unsloth/Phi-3.5-mini-instruct)
        dataset_path: Path to Q&A dataset
        output_dir: Output directory for results
        hub_model_id: Optional HuggingFace Hub ID for upload
        use_runpod: Force training on RunPod cloud
        skip_enhancement: Skip student-teacher enhancement
        max_samples: Limit number of training samples
        use_grokking: Enable grokking for better generalization
        use_dapo: Enable DAPO RL training
    
    Returns:
        Training results dictionary
    """
    from ..pipeline.complete_training_pipeline import CompletePipeline
    
    pipeline = CompletePipeline(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        use_runpod=use_runpod
    )
    
    # Configure additional options
    if use_dapo:
        pipeline.config.use_dapo = True
    if use_grokking:
        pipeline.config.use_grokking = True
    if skip_enhancement:
        pipeline.config.skip_enhancement = True
    if max_samples:
        pipeline.config.max_samples = max_samples
    
    results = await pipeline.run_pipeline()
    return results


@mcp.tool()
async def enhance_dataset(
    dataset_path: str,
    output_path: str,
    model: str = "claude-3-haiku-20240307",
    teacher_model: str = "claude-3-5-sonnet-20241022",
    max_samples: int = None,
    max_iterations: int = 3
) -> dict:
    """Enhance Q&A dataset with student-teacher thinking.
    
    Args:
        dataset_path: Path to input Q&A dataset
        output_path: Path for enhanced dataset
        model: Student model name
        teacher_model: Teacher model name
        max_samples: Limit number of samples
        max_iterations: Maximum enhancement iterations
    
    Returns:
        Enhancement results
    """
    from ..data.thinking_enhancer import StudentTeacherConfig, ThinkingEnhancer
    
    config = StudentTeacherConfig(
        student_model=model,
        teacher_model=teacher_model,
        max_iterations=max_iterations
    )
    
    enhancer = ThinkingEnhancer(config)
    
    # Load dataset
    import json
    with open(dataset_path) as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    # Enhance dataset
    enhanced_data = []
    for item in data:
        enhanced = await enhancer.enhance_qa_pair(item["question"], item["answer"])
        enhanced_data.append(enhanced)
    
    # Save enhanced dataset
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(enhanced_data, f, indent=2)
    
    return {
        "status": "success",
        "input_samples": len(data),
        "enhanced_samples": len(enhanced_data),
        "output_path": output_path
    }


@mcp.tool()
async def evaluate_model(
    model_path: str,
    test_dataset: str = None,
    judge_model: str = "gpt-4o-mini",
    metrics: list[str] = None
) -> dict:
    """Evaluate a fine-tuned model.
    
    Args:
        model_path: Path to model or adapter
        test_dataset: Optional test dataset path
        judge_model: Model to use as judge
        metrics: List of metrics to compute
    
    Returns:
        Evaluation results
    """
    from ..evaluation.litellm_evaluator import JudgeConfig, LiteLLMEvaluator
    from ..evaluation.config import EvaluationConfig
    
    config = EvaluationConfig()
    judge_config = JudgeConfig(judge_model=judge_model)
    
    evaluator = LiteLLMEvaluator(config, judge_config)
    
    # Load test data if provided
    test_data = None
    if test_dataset:
        import json
        with open(test_dataset) as f:
            test_data = json.load(f)
    
    # Run evaluation
    results = await evaluator.evaluate(
        model_name=model_path,
        test_data=test_data,
        metrics=metrics or ["accuracy", "fluency", "relevance"]
    )
    
    return results


@mcp.tool()
async def list_runpod_gpus() -> dict:
    """List available GPUs on RunPod.
    
    Returns:
        Available GPU configurations
    """
    import runpod
    from ..training.runpod_trainer import RunPodUnslothTrainer
    
    trainer = RunPodUnslothTrainer(api_key=os.getenv("RUNPOD_API_KEY"))
    gpus = trainer.list_available_gpus()
    
    return {
        "status": "success",
        "gpus": gpus,
        "count": len(gpus)
    }


@mcp.tool()
async def check_training_status(pod_id: str) -> dict:
    """Check status of a RunPod training job.
    
    Args:
        pod_id: RunPod pod ID
    
    Returns:
        Training status and logs
    """
    from ..training.runpod_training_ops import RunPodTrainingOps
    
    ops = RunPodTrainingOps(api_key=os.getenv("RUNPOD_API_KEY"))
    status = await ops.get_training_status(pod_id)
    
    return status


@mcp.tool()
async def generate_text(
    model_path: str,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.9,
    top_p: float = 0.95
) -> dict:
    """Generate text using a fine-tuned model.
    
    Args:
        model_path: Path to model or adapter
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated text
    """
    from ..inference.generate import GenerationConfig, InferenceEngine
    
    config = GenerationConfig(
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p
    )
    
    engine = InferenceEngine(model_path, config)
    result = engine.generate(prompt)
    
    return {
        "prompt": prompt,
        "generated_text": result,
        "config": config.__dict__
    }


@mcp.tool()
async def validate_model(
    model_path: str,
    test_prompts: list[str] = None
) -> dict:
    """Validate a fine-tuned model.
    
    Args:
        model_path: Path to model or adapter
        test_prompts: Optional test prompts
    
    Returns:
        Validation results
    """
    from ..validation.model_validator import ModelValidator
    
    validator = ModelValidator()
    results = validator.validate(
        model_path=model_path,
        test_prompts=test_prompts
    )
    
    return results


@mcp.tool()
async def calculate_token_entropy(
    text: str,
    model_name: str = "gpt2"
) -> dict:
    """Calculate token-level entropy for text.
    
    Args:
        text: Input text
        model_name: Model to use for tokenization
    
    Returns:
        Entropy statistics
    """
    from ..training.entropy_utils import calculate_token_entropy
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokens = tokenizer(text, return_tensors="pt")
    entropy = calculate_token_entropy(tokens["input_ids"])
    
    return {
        "text": text,
        "mean_entropy": float(entropy.mean()),
        "max_entropy": float(entropy.max()),
        "min_entropy": float(entropy.min()),
        "token_count": len(tokens["input_ids"][0])
    }


# =============================================================================
# SERVER
# =============================================================================

def serve():
    """Start the MCP server"""
    mcp.run(transport="stdio")  # Use stdio for Claude Code


if __name__ == "__main__":
    # Quick validation
    import asyncio

    async def validate():
        result = await capabilities()
        assert "unsloth_wip" in result.lower()
        print(" Server validation passed")

    asyncio.run(validate())

    # Start server
    serve()
