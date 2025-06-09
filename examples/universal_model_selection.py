#!/usr/bin/env python3
"""Example: Find the smallest accurate model using LiteLLM universal evaluation."""

import asyncio
from pathlib import Path
from unsloth.evaluation import LiteLLMEvaluator, JudgeConfig


async def main():
    """Demonstrate universal model evaluation with judge scores."""
    
    # Configure judge model with best practices
    judge_config = JudgeConfig(
        model="gpt-4",  # Or claude-3-opus-20240229 for alternative perspective
        temperature=0.0,  # Deterministic judging
        use_cot=True,  # Chain of thought for better reasoning
        use_few_shot=True,  # Include examples for calibration
        criteria=[
            "accuracy",      # Factual correctness
            "relevance",     # Answers the question directly
            "coherence",     # Logical flow and structure
            "completeness",  # Covers all aspects
            "conciseness"    # No unnecessary verbosity
        ],
        scoring_scale=(1, 5),  # Simpler scale for consistency
        position_swap=True,    # Mitigate position bias
        num_judges=1  # Could use 2-3 for consensus
    )
    
    # Create evaluator
    evaluator = LiteLLMEvaluator(
        dataset_path="./data/sample_eval_data.jsonl",
        output_dir="./universal_eval_results",
        max_samples=50,  # Quick evaluation
        target_accuracy=0.8,  # 80% threshold
        judge_config=judge_config
    )
    
    # First, list available models
    print(" Discovering available models...\n")
    available_models = evaluator.list_available_models()
    
    print(f"Found {len(available_models)} models:")
    for model in available_models[:10]:  # Show first 10
        size = f"{model.estimated_params}B" if model.estimated_params else "?"
        print(f"  - {model.model_id} ({size}) [{model.provider}]")
    
    if len(available_models) > 10:
        print(f"  ... and {len(available_models) - 10} more\n")
    
    # Example 1: Evaluate all small models (under 5B parameters)
    print("\n Evaluating small models (< 5B parameters)...")
    results = await evaluator.evaluate_models(
        include_local=True,
        include_cloud=True,
        size_limit=5.0
    )
    
    # Show recommendations
    analysis = results.get("analysis", {})
    print("\n Recommendations:")
    
    if analysis.get("smallest_accurate_model"):
        print(f"  Smallest accurate model: {analysis['smallest_accurate_model']}")
    
    if analysis.get("best_local_model"):
        print(f"  Best local model: {analysis['best_local_model']}")
    
    if analysis.get("best_cloud_model"):
        print(f"  Best cloud model: {analysis['best_cloud_model']}")
    
    # Example 2: Compare specific models
    print("\n\n Comparing specific models...")
    specific_models = [
        "ollama/phi3.5",           # Local 3.8B model
        "gpt-4o-mini",             # OpenAI's small model'
        "claude-3-haiku-20240307", # Anthropic's small model'
        "ollama/llama3.2:1b",      # Local 1B model
    ]
    
    # Filter to available models
    available_ids = [m.model_id for m in available_models]
    models_to_test = [m for m in specific_models if m in available_ids]
    
    if models_to_test:
        results2 = await evaluator.evaluate_models(
            models=models_to_test
        )
        
        # Show judge evaluation details for top model
        rankings = results2.get("analysis", {}).get("rankings", [])
        if rankings:
            top_model = rankings[0]
            print(f"\n Top model: {top_model['model_id']}")
            print(f"   Composite score: {top_model['composite_score']:.3f}")
            
            # Show judge criteria scores
            if "judge_scores" in top_model:
                print("   Judge scores by criteria:")
                for criterion, score in top_model["judge_scores"]["criteria_scores"].items():
                    print(f"     - {criterion}: {score:.1f}/5")
    
    print("\n Evaluation complete! Check ./universal_eval_results/ for full results.")


if __name__ == "__main__":
    # Example usage patterns:
    print("Universal Model Selection Examples:")
    print("==================================\n")
    
    print("1. Find smallest accurate model from all sources:")
    print("   unsloth universal-evaluate --dataset test.jsonl --target-accuracy 0.85\n")
    
    print("2. Test only local Ollama models:")
    print("   unsloth universal-evaluate --dataset test.jsonl --no-cloud\n")
    
    print("3. Compare specific models with custom judge:")
    print("   unsloth universal-evaluate --dataset test.jsonl \\")
    print("                              --model ollama/phi3.5 \\")
    print("                              --model gpt-4o-mini \\")
    print("                              --model claude-3-haiku \\")
    print("                              --judge claude-3-opus\n")
    
    print("4. Evaluate with custom criteria:")
    print("   unsloth universal-evaluate --dataset test.jsonl \\")
    print("                              --criteria accuracy \\")
    print("                              --criteria technical_correctness \\")
    print("                              --criteria code_quality\n")
    
    print("Running example evaluation...")
    print("=" * 50 + "\n")
    
    asyncio.run(main())