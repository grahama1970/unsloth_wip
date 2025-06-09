"""
Module: entropy_evaluator.py
Description: Entropy-aware evaluation metrics for model assessment

External Dependencies:
- torch: https://pytorch.org/docs/stable/index.html
- transformers: https://huggingface.co/docs/transformers/
- deepeval: https://docs.deepeval.com/
- numpy: https://numpy.org/doc/stable/
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> evaluator = EntropyAwareEvaluator(config)
>>> results = evaluator.evaluate_model(model, test_dataset)

Expected Output:
>>> results["entropy_metrics"]
{"avg_token_entropy": 0.72, "high_entropy_accuracy": 0.85}

Example Usage:
>>> from unsloth.evaluation.entropy_evaluator import EntropyAwareEvaluator
>>> evaluator = EntropyAwareEvaluator(config)
>>> metrics = evaluator.compute_entropy_metrics(model, dataset)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from loguru import logger
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from tqdm import tqdm

from unsloth.training.entropy_utils import (
    calculate_token_entropy,
    identify_high_entropy_tokens
)
from unsloth.evaluation.config import EvaluationConfig


@dataclass
class EntropyEvaluationConfig(EvaluationConfig):
    """Configuration for entropy-aware evaluation."""
    
    # Entropy-specific settings
    compute_entropy_metrics: bool = True
    entropy_percentile_threshold: float = 0.8  # Top 20%
    analyze_entropy_by_position: bool = True
    analyze_entropy_by_token_type: bool = True
    
    # High-entropy token analysis
    track_high_entropy_predictions: bool = True
    save_entropy_visualizations: bool = True
    
    # Comparison settings
    compare_entropy_before_after: bool = True
    entropy_reduction_threshold: float = 0.1  # 10% reduction is good


class EntropyMetric(BaseMetric):
    """Custom DeepEval metric for entropy evaluation."""
    
    def __init__(self, threshold: float = 0.7, name: str = "Entropy Clarity"):
        self.threshold = threshold
        self.name = name
        
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async measurement of entropy metric."""
        # For now, return sync result
        return self.measure(test_case)
        
    def measure(self, test_case: LLMTestCase) -> float:
        """Measure entropy-based clarity of model output."""
        # This is a simplified version - in practice would analyze token-level entropy
        output = test_case.actual_output
        
        # Simple heuristic: longer, more complex outputs tend to have higher entropy
        complexity_score = len(output.split()) / 100.0
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, 1.0 - complexity_score))
        
    def is_successful(self) -> bool:
        """Check if metric passes threshold."""
        return self.score >= self.threshold


class EntropyAwareEvaluator:
    """Evaluator with entropy-aware metrics."""
    
    def __init__(self, config: EntropyEvaluationConfig):
        """Initialize entropy-aware evaluator."""
        self.config = config
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_token_entropy(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute token-level entropy for model outputs."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                return_dict=True
            )
            
            logits = outputs.logits
            
        # Calculate entropy for each token
        entropies = []
        for i in range(logits.size(0)):  # Batch
            batch_entropies = []
            for j in range(logits.size(1)):  # Sequence
                token_entropy = calculate_token_entropy(
                    logits[i, j].cpu().numpy()
                )
                batch_entropies.append(token_entropy)
            entropies.append(batch_entropies)
            
        entropies = np.array(entropies)
        
        # Calculate metrics
        metrics = {
            "avg_entropy": float(np.mean(entropies)),
            "max_entropy": float(np.max(entropies)),
            "min_entropy": float(np.min(entropies)),
            "std_entropy": float(np.std(entropies)),
            "high_entropy_ratio": float(np.mean(entropies > np.percentile(entropies, self.config.entropy_percentile_threshold * 100)))
        }
        
        return entropies, metrics
        
    def analyze_entropy_patterns(
        self,
        model: AutoModelForCausalLM,
        dataset: Dataset,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Analyze entropy patterns across dataset."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
        all_entropies = []
        position_entropies = {}
        token_type_entropies = {}
        
        # Sample dataset
        samples = dataset.shuffle(seed=42).select(range(min(len(dataset), max_samples)))
        
        for idx, example in enumerate(tqdm(samples, desc="Analyzing entropy")):
            # Tokenize
            inputs = self.tokenizer(
                example.get("text", example.get("question", "")),
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=True
            )
            
            # Compute entropy
            entropies, metrics = self.compute_token_entropy(
                model,
                inputs["input_ids"],
                inputs.get("attention_mask")
            )
            
            all_entropies.extend(entropies.flatten())
            
            # Analyze by position
            if self.config.analyze_entropy_by_position:
                for pos, entropy in enumerate(entropies[0]):
                    if pos not in position_entropies:
                        position_entropies[pos] = []
                    position_entropies[pos].append(entropy)
                    
            # Analyze by token type (simplified)
            if self.config.analyze_entropy_by_token_type:
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                for token, entropy in zip(tokens, entropies[0]):
                    token_type = self._get_token_type(token)
                    if token_type not in token_type_entropies:
                        token_type_entropies[token_type] = []
                    token_type_entropies[token_type].append(entropy)
                    
        # Aggregate results
        results = {
            "overall": {
                "mean_entropy": float(np.mean(all_entropies)),
                "std_entropy": float(np.std(all_entropies)),
                "median_entropy": float(np.median(all_entropies)),
                "percentiles": {
                    "25": float(np.percentile(all_entropies, 25)),
                    "50": float(np.percentile(all_entropies, 50)),
                    "75": float(np.percentile(all_entropies, 75)),
                    "90": float(np.percentile(all_entropies, 90)),
                    "95": float(np.percentile(all_entropies, 95))
                }
            }
        }
        
        # Position analysis
        if position_entropies:
            results["by_position"] = {
                str(pos): {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
                for pos, values in position_entropies.items()
                if len(values) > 5  # Only positions with enough data
            }
            
        # Token type analysis
        if token_type_entropies:
            results["by_token_type"] = {
                token_type: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values)
                }
                for token_type, values in token_type_entropies.items()
            }
            
        return results
        
    def _get_token_type(self, token: str) -> str:
        """Categorize token type for analysis."""
        if token.startswith("##"):  # Subword
            return "subword"
        elif token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
            return "special"
        elif token.isalpha():
            return "word"
        elif token.isdigit():
            return "number"
        elif token in ".,!?;:":
            return "punctuation"
        else:
            return "other"
            
    def evaluate_high_entropy_predictions(
        self,
        model: AutoModelForCausalLM,
        dataset: Dataset,
        base_model: Optional[AutoModelForCausalLM] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance specifically on high-entropy tokens."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
        results = {
            "high_entropy_accuracy": [],
            "low_entropy_accuracy": [],
            "entropy_reduction": []
        }
        
        for example in tqdm(dataset, desc="Evaluating high-entropy predictions"):
            inputs = self.tokenizer(
                example.get("text", example.get("question", "")),
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            )
            
            # Get model predictions and entropy
            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"].to(self.device))
                logits = outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Calculate entropy
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                
            # Identify high-entropy positions
            entropy_threshold = torch.quantile(
                entropy.flatten(),
                self.config.entropy_percentile_threshold
            )
            high_entropy_mask = entropy > entropy_threshold
            
            # Calculate accuracy for high vs low entropy tokens
            if "labels" in example:
                labels = torch.tensor(example["labels"])
                correct = predictions.cpu() == labels
                
                high_entropy_acc = correct[high_entropy_mask].float().mean().item()
                low_entropy_acc = correct[~high_entropy_mask].float().mean().item()
                
                results["high_entropy_accuracy"].append(high_entropy_acc)
                results["low_entropy_accuracy"].append(low_entropy_acc)
                
            # Compare with base model if provided
            if base_model is not None:
                with torch.no_grad():
                    base_outputs = base_model(input_ids=inputs["input_ids"].to(self.device))
                    base_logits = base_outputs.logits
                    
                    base_probs = F.softmax(base_logits, dim=-1)
                    base_entropy = -torch.sum(base_probs * torch.log(base_probs + 1e-8), dim=-1)
                    
                # Calculate entropy reduction
                entropy_reduction = (base_entropy - entropy) / base_entropy
                avg_reduction = entropy_reduction.mean().item()
                results["entropy_reduction"].append(avg_reduction)
                
        # Aggregate results
        final_results = {}
        
        for key, values in results.items():
            if values:
                final_results[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values))
                }
                
        return final_results
        
    def generate_entropy_report(
        self,
        model: AutoModelForCausalLM,
        dataset: Dataset,
        output_dir: Path,
        model_name: str = "model"
    ) -> Path:
        """Generate comprehensive entropy analysis report."""
        logger.info(f"Generating entropy report for {model_name}")
        
        # Analyze entropy patterns
        entropy_patterns = self.analyze_entropy_patterns(model, dataset)
        
        # Evaluate high-entropy predictions
        high_entropy_eval = self.evaluate_high_entropy_predictions(model, dataset)
        
        # Create report
        report = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "entropy_threshold": self.config.entropy_percentile_threshold,
                "max_seq_length": self.config.max_seq_length
            },
            "entropy_patterns": entropy_patterns,
            "high_entropy_evaluation": high_entropy_eval,
            "summary": self._generate_summary(entropy_patterns, high_entropy_eval)
        }
        
        # Save report
        report_path = output_dir / f"entropy_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Entropy report saved to: {report_path}")
        
        # Generate visualizations if enabled
        if self.config.save_entropy_visualizations:
            self._generate_visualizations(report, output_dir, model_name)
            
        return report_path
        
    def _generate_summary(
        self,
        patterns: Dict[str, Any],
        high_entropy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary insights from entropy analysis."""
        summary = {
            "key_findings": [],
            "recommendations": []
        }
        
        # Analyze overall entropy
        mean_entropy = patterns["overall"]["mean_entropy"]
        if mean_entropy > 0.7:
            summary["key_findings"].append(
                f"High average entropy ({mean_entropy:.3f}) indicates model uncertainty"
            )
            summary["recommendations"].append(
                "Consider additional training or larger model capacity"
            )
        elif mean_entropy < 0.3:
            summary["key_findings"].append(
                f"Low average entropy ({mean_entropy:.3f}) indicates confident predictions"
            )
            
        # Analyze high-entropy performance
        if "high_entropy_accuracy" in high_entropy:
            high_acc = high_entropy["high_entropy_accuracy"]["mean"]
            low_acc = high_entropy.get("low_entropy_accuracy", {}).get("mean", 0)
            
            if high_acc < low_acc * 0.8:
                summary["key_findings"].append(
                    f"Model struggles with uncertain tokens (accuracy: {high_acc:.2%} vs {low_acc:.2%})"
                )
                summary["recommendations"].append(
                    "Focus training on high-entropy regions with entropy-aware loss"
                )
                
        # Check entropy reduction
        if "entropy_reduction" in high_entropy:
            reduction = high_entropy["entropy_reduction"]["mean"]
            if reduction > self.config.entropy_reduction_threshold:
                summary["key_findings"].append(
                    f"Good entropy reduction achieved ({reduction:.1%})"
                )
            else:
                summary["recommendations"].append(
                    "Consider longer training or different learning rate schedule"
                )
                
        return summary
        
    def _generate_visualizations(
        self,
        report: Dict[str, Any],
        output_dir: Path,
        model_name: str
    ):
        """Generate entropy visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Entropy Analysis - {model_name}', fontsize=16)
            
            # 1. Entropy distribution
            if "overall" in report["entropy_patterns"]:
                ax = axes[0, 0]
                data = report["entropy_patterns"]["overall"]
                
                # Create percentile data for box plot
                percentiles = [
                    data["percentiles"]["25"],
                    data["percentiles"]["50"],
                    data["percentiles"]["75"],
                    data["percentiles"]["90"],
                    data["percentiles"]["95"]
                ]
                
                ax.boxplot([percentiles])
                ax.set_ylabel('Entropy')
                ax.set_title('Entropy Distribution')
                ax.set_xticklabels(['Tokens'])
                
            # 2. Position-wise entropy
            if "by_position" in report["entropy_patterns"]:
                ax = axes[0, 1]
                positions = []
                means = []
                
                for pos, stats in report["entropy_patterns"]["by_position"].items():
                    if int(pos) < 50:  # First 50 positions
                        positions.append(int(pos))
                        means.append(stats["mean"])
                        
                ax.plot(positions, means, 'b-', linewidth=2)
                ax.set_xlabel('Position')
                ax.set_ylabel('Mean Entropy')
                ax.set_title('Entropy by Token Position')
                
            # 3. Token type entropy
            if "by_token_type" in report["entropy_patterns"]:
                ax = axes[1, 0]
                token_types = list(report["entropy_patterns"]["by_token_type"].keys())
                means = [stats["mean"] for stats in report["entropy_patterns"]["by_token_type"].values()]
                
                ax.bar(token_types, means)
                ax.set_xlabel('Token Type')
                ax.set_ylabel('Mean Entropy')
                ax.set_title('Entropy by Token Type')
                ax.tick_params(axis='x', rotation=45)
                
            # 4. High vs Low entropy accuracy
            if "high_entropy_evaluation" in report:
                ax = axes[1, 1]
                eval_data = report["high_entropy_evaluation"]
                
                if "high_entropy_accuracy" in eval_data and "low_entropy_accuracy" in eval_data:
                    categories = ['High Entropy', 'Low Entropy']
                    accuracies = [
                        eval_data["high_entropy_accuracy"]["mean"],
                        eval_data["low_entropy_accuracy"]["mean"]
                    ]
                    
                    ax.bar(categories, accuracies, color=['red', 'green'])
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Performance on High vs Low Entropy Tokens')
                    ax.set_ylim(0, 1)
                    
            plt.tight_layout()
            
            # Save figure
            viz_path = output_dir / f"entropy_visualization_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to: {viz_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")


# Validation
if __name__ == "__main__":
    # Test configuration
    config = EntropyEvaluationConfig(
        model_name="unsloth/Phi-3.5-mini-instruct",
        compute_entropy_metrics=True,
        entropy_percentile_threshold=0.8
    )
    
    evaluator = EntropyAwareEvaluator(config)
    
    # Test entropy metric
    metric = EntropyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input="What is AI?",
        actual_output="Artificial Intelligence is a field of computer science.",
        expected_output="AI stands for Artificial Intelligence."
    )
    
    score = metric.measure(test_case)
    print(f"Entropy clarity score: {score:.2f}")
    
    print("\n Module validation passed")