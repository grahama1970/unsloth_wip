#!/usr/bin/env python3
"""Example script for evaluating models with the Unsloth evaluation system."""

import asyncio
from pathlib import Path
from unsloth.evaluation import ModelEvaluator, EvaluationConfig, DashboardGenerator


async def main():
    """Run a simple evaluation example."""
    
    # Configure evaluation
    config = EvaluationConfig(
        # Models
        base_model_path="unsloth/Phi-3.5-mini-instruct",
        lora_model_path="./outputs/adapter",  # Optional - set to None for single model eval
        
        # Dataset
        dataset_path="./data/qa_test.jsonl",
        max_samples=100,  # Limit for quick testing
        
        # Metrics to run
        metrics=[
            {"name": "perplexity", "enabled": True},
            {"name": "answer_relevancy", "enabled": True},
            {"name": "hallucination", "enabled": True},
            {"name": "faithfulness", "enabled": True},
        ],
        
        # Judge model settings
        use_judge_model=True,
        judge_config={
            "model_name": "gpt-4",
            "temperature": 0.0,
            "criteria": ["accuracy", "relevance", "coherence", "fluency"]
        },
        
        # Output settings
        output_dir="./evaluation_results",
        generate_html_report=True,
        generate_mlflow_run=False,  # Set to True if using MLflow
        
        # Comparison
        compare_models=True  # Will compare base vs LoRA if both provided
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Run evaluation
    print(" Starting evaluation...")
    results = await evaluator.evaluate_all()
    
    # Generate dashboard
    if config.generate_html_report:
        print(" Generating HTML dashboard...")
        dashboard = DashboardGenerator(results, config.output_dir)
        dashboard_path = dashboard.generate()
        print(f" Dashboard saved to: {dashboard_path}")
    
    # Print summary
    if results.get("comparison"):
        comparison = results["comparison"]["summary"]
        print(f"\n Evaluation Summary:")
        print(f"   Improvements: {comparison['total_improvements']}")
        print(f"   Regressions: {comparison['total_regressions']}")
        print(f"   Recommendation: {comparison['recommendation']}")
        
        if "judge_score_improvement" in comparison:
            judge = comparison["judge_score_improvement"]
            print(f"\n Judge Score:")
            print(f"   Base: {judge['base']:.2f}/10")
            print(f"   LoRA: {judge['lora']:.2f}/10")
            print(f"   Improvement: {judge['improvement_pct']:+.1f}%")
    
    print("\n Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())