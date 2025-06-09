"""
Module: __init__.py
Description: Evaluation module for comparing LLM models pre and post LoRA training

External Dependencies:
- deepeval: https://docs.deepeval.com/
- litellm: https://docs.litellm.ai/
- torch: https://pytorch.org/docs/stable/index.html

Sample Input:
>>> from unsloth.evaluation import ModelEvaluator
>>> evaluator = ModelEvaluator(config)

Expected Output:
>>> results = evaluator.evaluate_model(model)
>>> results["metrics"]["accuracy"]
0.85

Example Usage:
>>> from unsloth.evaluation import EntropyAwareEvaluator
>>> evaluator = EntropyAwareEvaluator(config)
>>> evaluator.generate_entropy_report(model, dataset)
"""

from .config import EvaluationConfig
from .dashboard import DashboardGenerator
from .entropy_evaluator import EntropyAwareEvaluator, EntropyEvaluationConfig, EntropyMetric
from .evaluator import ModelEvaluator
from .litellm_evaluator import JudgeConfig, LiteLLMEvaluator, LiteLLMModel
from .multi_model_evaluator import ModelCandidate, MultiModelEvaluator

__all__ = [
    "ModelEvaluator",
    "EvaluationConfig",
    "EntropyAwareEvaluator",
    "EntropyEvaluationConfig",
    "EntropyMetric",
    "DashboardGenerator",
    "MultiModelEvaluator",
    "ModelCandidate",
    "LiteLLMEvaluator",
    "LiteLLMModel",
    "JudgeConfig"
]
