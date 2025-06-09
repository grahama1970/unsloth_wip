"""
Module: config.py
Description: Configuration for model evaluation

External Dependencies:
- pydantic: https://docs.pydantic.dev/

Sample Input:
>>> config = EvaluationConfig(model_name="unsloth/Phi-3.5-mini-instruct")

Expected Output:
>>> config.use_deepeval
True

Example Usage:
>>> from unsloth.evaluation.config import EvaluationConfig
>>> config = EvaluationConfig()
"""

from typing import Any

from pydantic import BaseModel, Field


class MetricConfig(BaseModel):
    """Configuration for individual metrics."""

    name: str = Field(..., description="Metric name")
    enabled: bool = Field(True, description="Whether to run this metric")
    params: dict[str, Any] = Field(default_factory=dict, description="Metric parameters")


class JudgeModelConfig(BaseModel):
    """Configuration for judge model evaluation."""

    model_name: str = Field("gpt-4", description="Judge model to use")
    temperature: float = Field(0.0, description="Temperature for judge model")
    max_tokens: int = Field(500, description="Max tokens for judge response")
    criteria: list[str] = Field(
        default_factory=lambda: ["accuracy", "relevance", "coherence", "fluency"],
        description="Evaluation criteria"
    )


class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""

    # Dataset configuration
    dataset_path: str = Field(..., description="Path to evaluation dataset")
    dataset_name: str | None = Field(None, description="Dataset name for HuggingFace")
    dataset_split: str = Field("test", description="Dataset split to use")
    max_samples: int | None = Field(None, description="Maximum number of samples to evaluate")

    # Model configuration
    base_model_path: str = Field(..., description="Path or HF ID of base model")
    lora_model_path: str | None = Field(None, description="Path to LoRA adapter")
    device: str = Field("cuda", description="Device to run evaluation on")
    load_in_4bit: bool = Field(True, description="Load model in 4-bit quantization")

    # Metrics configuration
    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [
            MetricConfig(name="perplexity"),
            MetricConfig(name="answer_relevancy"),
            MetricConfig(name="faithfulness"),
            MetricConfig(name="contextual_precision"),
            MetricConfig(name="contextual_recall"),
            MetricConfig(name="hallucination"),
            MetricConfig(name="toxicity"),
            MetricConfig(name="bias"),
        ],
        description="Metrics to evaluate"
    )

    # Judge model configuration
    use_judge_model: bool = Field(True, description="Whether to use judge model evaluation")
    judge_config: JudgeModelConfig = Field(
        default_factory=JudgeModelConfig,
        description="Judge model configuration"
    )

    # Output configuration
    output_dir: str = Field("./evaluation_results", description="Directory for results")
    generate_html_report: bool = Field(True, description="Generate HTML dashboard")
    generate_mlflow_run: bool = Field(True, description="Log to MLflow")
    mlflow_tracking_uri: str | None = Field(None, description="MLflow tracking URI")
    mlflow_experiment_name: str = Field("unsloth_evaluation", description="MLflow experiment name")

    # Comparison configuration
    compare_models: bool = Field(True, description="Compare base vs LoRA model")
    comparison_metrics: list[str] = Field(
        default_factory=lambda: ["accuracy_delta", "perplexity_reduction", "judge_score_improvement"],
        description="Metrics to compare between models"
    )
