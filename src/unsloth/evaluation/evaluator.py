"""Model evaluation with DeepEval and MLflow integration."""
Module: evaluator.py

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from datasets import Dataset, load_dataset

# Evaluation frameworks
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase
from litellm import acompletion
from lm_eval import evaluator as lm_evaluator
from loguru import logger
from peft import PeftModel
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EvaluationConfig


class ModelEvaluator:
    """Evaluates models before and after LoRA training."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {
            "base_model": {},
            "lora_model": {},
            "comparison": {},
            "metadata": {
                "evaluation_date": datetime.utcnow().isoformat(),
                "config": config.model_dump()
            }
        }

        # Initialize MLflow if enabled
        if config.generate_mlflow_run:
            if config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment_name)

    async def evaluate_all(self) -> dict[str, Any]:
        """Run full evaluation pipeline."""
        logger.info("Starting model evaluation pipeline")

        # Load dataset
        dataset = self._load_dataset()

        # Evaluate base model
        logger.info("Evaluating base model...")
        base_results = await self._evaluate_model(
            model_path=self.config.base_model_path,
            dataset=dataset,
            model_type="base"
        )
        self.results["base_model"] = base_results

        # Evaluate LoRA model if provided
        if self.config.lora_model_path and self.config.compare_models:
            logger.info("Evaluating LoRA model...")
            lora_results = await self._evaluate_model(
                model_path=self.config.base_model_path,
                lora_path=self.config.lora_model_path,
                dataset=dataset,
                model_type="lora"
            )
            self.results["lora_model"] = lora_results

            # Compare results
            self.results["comparison"] = self._compare_results(base_results, lora_results)

        # Save results
        self._save_results()

        return self.results

    def _load_dataset(self) -> Dataset:
        """Load evaluation dataset."""
        if self.config.dataset_name:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split
            )
        else:
            dataset = load_dataset(
                "json",
                data_files=self.config.dataset_path,
                split="train"
            )

        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))

        return dataset

    async def _evaluate_model(
        self,
        model_path: str,
        dataset: Dataset,
        model_type: str,
        lora_path: str | None = None
    ) -> dict[str, Any]:
        """Evaluate a single model."""
        results = {
            "metrics": {},
            "judge_scores": {},
            "examples": [],
            "model_info": {
                "model_path": model_path,
                "lora_path": lora_path,
                "model_type": model_type
            }
        }

        # Load model and tokenizer
        model, tokenizer = self._load_model(model_path, lora_path)

        # Run DeepEval metrics
        if any(m.name in ["answer_relevancy", "faithfulness", "hallucination"] for m in self.config.metrics):
            deepeval_results = await self._run_deepeval_metrics(model, tokenizer, dataset)
            results["metrics"].update(deepeval_results)

        # Run lm-eval benchmarks
        if any(m.name in ["perplexity", "hellaswag", "mmlu"] for m in self.config.metrics):
            lm_eval_results = self._run_lm_eval(model_path, lora_path, dataset)
            results["metrics"].update(lm_eval_results)

        # Run judge model evaluation
        if self.config.use_judge_model:
            judge_results = await self._run_judge_evaluation(model, tokenizer, dataset)
            results["judge_scores"] = judge_results

        # Collect example outputs
        results["examples"] = await self._collect_examples(model, tokenizer, dataset, num_examples=5)

        # Log to MLflow
        if self.config.generate_mlflow_run:
            self._log_to_mlflow(results, model_type)

        # Clean up model
        del model
        torch.cuda.empty_cache()

        return results

    def _load_model(self, model_path: str, lora_path: str | None = None) -> tuple[Any, Any]:
        """Load model with optional LoRA adapter."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path)

        return model, tokenizer

    async def _run_deepeval_metrics(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Dataset
    ) -> dict[str, float]:
        """Run DeepEval metrics."""
        results = {}
        test_cases = []

        # Convert dataset to test cases
        for item in dataset:
            test_case = LLMTestCase(
                input=item.get("question", item.get("input", "")),
                actual_output=self._generate_response(model, tokenizer, item["question"]),
                expected_output=item.get("answer", item.get("output", "")),
                context=item.get("context", []),
            )
            test_cases.append(test_case)

        # Initialize metrics
        metrics = []
        for metric_config in self.config.metrics:
            if not metric_config.enabled:
                continue

            if metric_config.name == "answer_relevancy":
                metrics.append(AnswerRelevancyMetric(**metric_config.params))
            elif metric_config.name == "faithfulness":
                metrics.append(FaithfulnessMetric(**metric_config.params))
            elif metric_config.name == "hallucination":
                metrics.append(HallucinationMetric(**metric_config.params))
            elif metric_config.name == "toxicity":
                metrics.append(ToxicityMetric(**metric_config.params))
            elif metric_config.name == "bias":
                metrics.append(BiasMetric(**metric_config.params))

        # Run evaluation
        if metrics:
            eval_results = evaluate(test_cases, metrics)
            for metric in metrics:
                results[metric.__class__.__name__] = metric.score

        return results

    def _run_lm_eval(
        self,
        model_path: str,
        lora_path: str | None,
        dataset: Dataset
    ) -> dict[str, float]:
        """Run lm-eval benchmarks."""
        results = {}

        # Prepare model args
        model_args = f"pretrained={model_path}"
        if lora_path:
            model_args += f",peft={lora_path}"
        if self.config.load_in_4bit:
            model_args += ",load_in_4bit=True"

        # Run selected benchmarks
        tasks = []
        for metric in self.config.metrics:
            if metric.name == "perplexity":
                tasks.append("wikitext")
            elif metric.name == "hellaswag":
                tasks.append("hellaswag")
            elif metric.name == "mmlu":
                tasks.append("mmlu")

        if tasks:
            outputs = lm_evaluator.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=tasks,
                device=self.config.device,
                batch_size="auto"
            )

            for task, result in outputs["results"].items():
                for metric_name, value in result.items():
                    if isinstance(value, (int, float)):
                        results[f"{task}_{metric_name}"] = value

        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _run_judge_evaluation(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Dataset
    ) -> dict[str, Any]:
        """Run judge model evaluation."""
        judge_results = {
            "overall_score": 0.0,
            "criteria_scores": {},
            "detailed_feedback": []
        }

        judge_config = self.config.judge_config
        all_scores = {criterion: [] for criterion in judge_config.criteria}

        for idx, item in enumerate(dataset):
            question = item.get("question", item.get("input", ""))
            expected = item.get("answer", item.get("output", ""))
            actual = self._generate_response(model, tokenizer, question)

            # Create judge prompt
            judge_prompt = f"""You are an expert evaluator. Please evaluate the following response based on these criteria: {', '.join(judge_config.criteria)}.

Question: {question}
Expected Answer: {expected}
Model Response: {actual}

For each criterion, provide a score from 0-10 and brief explanation. Format your response as JSON:
{{
    "scores": {{"criterion": score, ...}},
    "feedback": {{"criterion": "explanation", ...}},
    "overall": score
}}"""

            # Get judge evaluation
            response = await acompletion(
                model=judge_config.model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=judge_config.temperature,
                max_tokens=judge_config.max_tokens,
                response_format={"type": "json_object"}
            )

            try:
                eval_result = json.loads(response.choices[0].message.content)

                # Collect scores
                for criterion, score in eval_result["scores"].items():
                    if criterion in all_scores:
                        all_scores[criterion].append(score)

                judge_results["detailed_feedback"].append({
                    "question": question,
                    "evaluation": eval_result
                })

            except Exception as e:
                logger.warning(f"Failed to parse judge response: {e}")

        # Calculate average scores
        for criterion, scores in all_scores.items():
            if scores:
                judge_results["criteria_scores"][criterion] = sum(scores) / len(scores)

        judge_results["overall_score"] = sum(judge_results["criteria_scores"].values()) / len(judge_results["criteria_scores"])

        return judge_results

    async def _collect_examples(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Dataset,
        num_examples: int = 5
    ) -> list[dict[str, str]]:
        """Collect example model outputs."""
        examples = []

        for i, item in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
            question = item.get("question", item.get("input", ""))
            expected = item.get("answer", item.get("output", ""))
            actual = self._generate_response(model, tokenizer, question)

            examples.append({
                "question": question,
                "expected": expected,
                "actual": actual,
                "index": i
            })

        return examples

    def _generate_response(self, model: Any, tokenizer: Any, prompt: str) -> str:
        """Generate response from model."""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()

        return response

    def _compare_results(
        self,
        base_results: dict[str, Any],
        lora_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare base and LoRA model results."""
        comparison = {
            "improvements": {},
            "regressions": {},
            "summary": {}
        }

        # Compare metrics
        base_metrics = base_results.get("metrics", {})
        lora_metrics = lora_results.get("metrics", {})

        for metric in set(base_metrics.keys()) | set(lora_metrics.keys()):
            if metric in base_metrics and metric in lora_metrics:
                base_val = base_metrics[metric]
                lora_val = lora_metrics[metric]

                # Calculate improvement (higher is better for most metrics)
                if "perplexity" in metric.lower():
                    # Lower is better for perplexity
                    improvement = (base_val - lora_val) / base_val * 100
                else:
                    # Higher is better for most metrics
                    improvement = (lora_val - base_val) / base_val * 100

                if improvement > 0:
                    comparison["improvements"][metric] = {
                        "base": base_val,
                        "lora": lora_val,
                        "improvement_pct": improvement
                    }
                else:
                    comparison["regressions"][metric] = {
                        "base": base_val,
                        "lora": lora_val,
                        "regression_pct": abs(improvement)
                    }

        # Compare judge scores
        if "judge_scores" in base_results and "judge_scores" in lora_results:
            base_judge = base_results["judge_scores"]["overall_score"]
            lora_judge = lora_results["judge_scores"]["overall_score"]
            judge_improvement = (lora_judge - base_judge) / base_judge * 100

            comparison["summary"]["judge_score_improvement"] = {
                "base": base_judge,
                "lora": lora_judge,
                "improvement_pct": judge_improvement
            }

        # Overall summary
        comparison["summary"]["total_improvements"] = len(comparison["improvements"])
        comparison["summary"]["total_regressions"] = len(comparison["regressions"])
        comparison["summary"]["recommendation"] = (
            "LoRA adapter shows overall improvement"
            if len(comparison["improvements"]) > len(comparison["regressions"])
            else "LoRA adapter needs further tuning"
        )

        return comparison

    def _save_results(self):
        """Save evaluation results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results as JSON
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Save comparison summary as CSV
        if self.results.get("comparison"):
            comparison_df = pd.DataFrame([
                {
                    "metric": metric,
                    "base_value": data["base"],
                    "lora_value": data["lora"],
                    "improvement_pct": data.get("improvement_pct", -data.get("regression_pct", 0))
                }
                for metric, data in {
                    **self.results["comparison"].get("improvements", {}),
                    **self.results["comparison"].get("regressions", {})
                }.items()
            ])
            comparison_df.to_csv(output_dir / "comparison_summary.csv", index=False)

        logger.info(f"Results saved to {output_dir}")

    def _log_to_mlflow(self, results: dict[str, Any], model_type: str):
        """Log results to MLflow."""
        with mlflow.start_run(run_name=f"evaluation_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            # Log metrics
            for metric_name, value in results.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{model_type}_{metric_name}", value)

            # Log judge scores
            if "judge_scores" in results:
                mlflow.log_metric(f"{model_type}_judge_overall", results["judge_scores"]["overall_score"])
                for criterion, score in results["judge_scores"]["criteria_scores"].items():
                    mlflow.log_metric(f"{model_type}_judge_{criterion}", score)

            # Log parameters
            mlflow.log_params({
                "model_type": model_type,
                "model_path": results["model_info"]["model_path"],
                "lora_path": results["model_info"].get("lora_path", "none"),
                "dataset": self.config.dataset_path or self.config.dataset_name,
                "max_samples": self.config.max_samples or "all"
            })

            # Log artifacts
            mlflow.log_dict(results, f"{model_type}_results.json")
