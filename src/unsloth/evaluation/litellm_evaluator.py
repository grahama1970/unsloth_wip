"""Universal model evaluation using LiteLLM for all model providers."""
Module: litellm_evaluator.py

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# LiteLLM integration
from llm_call import ask
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

# ask is deprecated, using tenacity for retries instead


@dataclass
class LiteLLMModel:
    """Model accessible through LiteLLM."""
    model_id: str
    provider: str = ""
    estimated_params: float | None = None  # billions
    cost_per_1k_tokens: float | None = None
    context_window: int = 4096
    supports_streaming: bool = True
    is_local: bool = False
    notes: str = ""

    def __post_init__(self):
        """Infer provider from model ID."""
        if not self.provider:
            if self.model_id.startswith("ollama/"):
                self.provider = "ollama"
                self.is_local = True
            elif self.model_id.startswith("openai/") or self.model_id.startswith("gpt"):
                self.provider = "openai"
            elif self.model_id.startswith("anthropic/") or self.model_id.startswith("claude"):
                self.provider = "anthropic"
            elif self.model_id.startswith("openrouter/"):
                self.provider = "openrouter"
            elif "/" in self.model_id:
                self.provider = self.model_id.split("/")[0]
            else:
                self.provider = "unknown"


@dataclass
class JudgeConfig:
    """Configuration for judge model evaluation."""
    model: str = "gpt-4"
    temperature: float = 0.0
    use_cot: bool = True  # Chain of thought
    use_few_shot: bool = True
    criteria: list[str] = field(default_factory=lambda: [
        "accuracy",
        "relevance",
        "coherence",
        "completeness",
        "conciseness"
    ])
    scoring_scale: tuple[int, int] = (1, 5)  # Min, max score
    position_swap: bool = True  # Swap positions to avoid bias
    num_judges: int = 1  # Use multiple judges for consensus


class LiteLLMEvaluator:
    """Evaluate any model accessible through LiteLLM."""

    # Common models with estimated sizes
    MODEL_SIZES = {
        # Ollama models
        "ollama/tinyllama": 1.1,
        "ollama/phi": 2.7,
        "ollama/phi3": 3.8,
        "ollama/phi3.5": 3.8,
        "ollama/gemma:2b": 2.0,
        "ollama/gemma2:2b": 2.0,
        "ollama/qwen2.5:1.5b": 1.5,
        "ollama/qwen2.5:3b": 3.0,
        "ollama/llama3.2:1b": 1.0,
        "ollama/llama3.2:3b": 3.0,
        "ollama/mistral": 7.0,
        "ollama/llama3.1:8b": 8.0,
        "ollama/gemma2:9b": 9.0,
        "ollama/llama3.1:70b": 70.0,

        # OpenAI models
        "gpt-3.5-turbo": 20.0,  # Estimated
        "gpt-4": 175.0,  # Estimated
        "gpt-4-turbo": 175.0,
        "gpt-4o": 200.0,  # Estimated
        "gpt-4o-mini": 8.0,  # Estimated

        # Anthropic models
        "claude-3-haiku": 20.0,  # Estimated
        "claude-3-sonnet": 70.0,  # Estimated
        "claude-3-opus": 175.0,  # Estimated
        "claude-3.5-sonnet": 175.0,  # Estimated

        # OpenRouter models (examples)
        "openrouter/meta-llama/llama-3.2-1b-instruct": 1.0,
        "openrouter/meta-llama/llama-3.2-3b-instruct": 3.0,
        "openrouter/mistralai/mistral-7b-instruct": 7.0,
        "openrouter/google/gemma-2-9b-it": 9.0,
    }

    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "./litellm_evaluation_results",
        max_samples: int | None = 100,
        target_accuracy: float = 0.8,
        judge_config: JudgeConfig | None = None
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.target_accuracy = target_accuracy
        self.judge_config = judge_config or JudgeConfig()
        self.console = Console()
        self.results = {}

        # Load dataset once
        self.dataset = self._load_dataset()

    def list_available_models(self, provider: str | None = None) -> list[LiteLLMModel]:
        """List all available models through LiteLLM."""
        models = []

        # Get Ollama models if available
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                    if line:
                        model_name = line.split()[0]
                        full_name = f"ollama/{model_name}"
                        models.append(LiteLLMModel(
                            model_id=full_name,
                            provider="ollama",
                            estimated_params=self.MODEL_SIZES.get(full_name),
                            is_local=True
                        ))
        except:
            logger.debug("Ollama not available")

        # Add known cloud models
        cloud_models = [
            # OpenAI
            LiteLLMModel("gpt-3.5-turbo", estimated_params=20.0),
            LiteLLMModel("gpt-4", estimated_params=175.0),
            LiteLLMModel("gpt-4-turbo", estimated_params=175.0),
            LiteLLMModel("gpt-4o-mini", estimated_params=8.0),

            # Anthropic
            LiteLLMModel("claude-3-haiku-20240307", estimated_params=20.0),
            LiteLLMModel("claude-3-sonnet-20240229", estimated_params=70.0),
            LiteLLMModel("claude-3-opus-20240229", estimated_params=175.0),
            LiteLLMModel("claude-3-5-sonnet-20241022", estimated_params=175.0),
        ]

        # Add OpenRouter models if API key is set
        import os
        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_models = [
                LiteLLMModel("openrouter/meta-llama/llama-3.2-1b-instruct", estimated_params=1.0),
                LiteLLMModel("openrouter/meta-llama/llama-3.2-3b-instruct", estimated_params=3.0),
                LiteLLMModel("openrouter/mistralai/mistral-7b-instruct", estimated_params=7.0),
                LiteLLMModel("openrouter/google/gemma-2-9b-it", estimated_params=9.0),
                LiteLLMModel("openrouter/microsoft/phi-3.5-mini-128k-instruct", estimated_params=3.8),
            ]
            cloud_models.extend(openrouter_models)

        models.extend(cloud_models)

        # Filter by provider if specified
        if provider:
            models = [m for m in models if m.provider == provider]

        return models

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Load evaluation dataset."""
        import json

        dataset = []
        with open(self.dataset_path) as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))

        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset[:self.max_samples]

        return dataset

    async def evaluate_models(
        self,
        models: list[str] | None = None,
        include_local: bool = True,
        include_cloud: bool = True,
        size_limit: float | None = None
    ) -> dict[str, Any]:
        """Evaluate specified models or auto-discover."""

        # Get model list
        if models:
            # Convert strings to LiteLLMModel objects
            model_objects = []
            for model_id in models:
                size = self.MODEL_SIZES.get(model_id)
                model_objects.append(LiteLLMModel(
                    model_id=model_id,
                    estimated_params=size
                ))
        else:
            # Auto-discover models
            available = self.list_available_models()

            # Filter by type
            if not include_local:
                available = [m for m in available if not m.is_local]
            if not include_cloud:
                available = [m for m in available if m.is_local]

            # Filter by size
            if size_limit:
                available = [
                    m for m in available
                    if m.estimated_params is None or m.estimated_params <= size_limit
                ]

            model_objects = available

        if not model_objects:
            self.console.print("[red]No models found matching criteria[/red]")
            return {}

        # Show models to evaluate
        self._print_models_table(model_objects)

        # Evaluate each model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:

            eval_task = progress.add_task(
                f"[cyan]Evaluating {len(model_objects)} models...",
                total=len(model_objects)
            )

            for model in model_objects:
                progress.update(
                    eval_task,
                    description=f"[cyan]Evaluating {model.model_id}..."
                )

                try:
                    # Evaluate model
                    result = await self._evaluate_single_model(model)
                    self.results[model.model_id] = result

                except Exception as e:
                    logger.error(f"Failed to evaluate {model.model_id}: {e}")
                    self.results[model.model_id] = {
                        "error": str(e),
                        "model_info": model.__dict__
                    }

                progress.advance(eval_task)

        # Run judge evaluation on all results
        self.console.print("\n[bold cyan] Running judge evaluation...[/bold cyan]")
        await self._run_judge_evaluation()

        # Analyze results
        self.results["analysis"] = self._analyze_results()

        # Save results
        self._save_results()

        # Generate dashboard
        self._generate_dashboard()

        # Print summary
        self._print_summary()

        return self.results

    async def _evaluate_single_model(self, model: LiteLLMModel) -> dict[str, Any]:
        """Evaluate a single model on the dataset."""
        result = {
            "model_info": model.__dict__,
            "responses": [],
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Test each example
        for idx, example in enumerate(self.dataset):
            question = example.get("question", example.get("input", ""))
            expected = example.get("answer", example.get("output", ""))
            context = example.get("context", [])

            # Prepare prompt
            if context:
                prompt = f"Context: {' '.join(context)}\n\nQuestion: {question}"
            else:
                prompt = question

            try:
                # Get model response
                response = await ask(
                    prompt=prompt,
                    model=model.model_id,
                    temperature=0.7,
                    max_tokens=256
                )

                result["responses"].append({
                    "index": idx,
                    "question": question,
                    "expected": expected,
                    "response": response,
                    "context": context
                })

            except Exception as e:
                logger.warning(f"Failed to get response from {model.model_id}: {e}")
                result["responses"].append({
                    "index": idx,
                    "question": question,
                    "expected": expected,
                    "response": None,
                    "error": str(e)
                })

        # Calculate basic metrics
        valid_responses = [r for r in result["responses"] if r.get("response") and not r.get("error")]
        result["metrics"]["response_rate"] = len(valid_responses) / len(self.dataset)
        result["metrics"]["avg_response_length"] = (
            sum(len(r["response"]) for r in valid_responses) / len(valid_responses)
            if valid_responses else 0
        )

        return result

    async def _run_judge_evaluation(self):
        """Run judge model evaluation on all collected responses."""
        judge_model = self.judge_config.model

        # Prepare for judge evaluation
        all_evaluations = defaultdict(list)

        for model_id, result in self.results.items():
            if model_id == "analysis" or "error" in result:
                continue

            responses = result.get("responses", [])

            for resp in responses:
                if not resp.get("response"):
                    continue

                # Get judge evaluation
                try:
                    scores = await self._get_judge_scores(
                        question=resp["question"],
                        expected=resp["expected"],
                        actual=resp["response"],
                        context=resp.get("context", [])
                    )

                    resp["judge_scores"] = scores

                    # Aggregate scores
                    for criterion, score in scores["criteria_scores"].items():
                        all_evaluations[model_id].append({
                            "criterion": criterion,
                            "score": score
                        })

                except Exception as e:
                    logger.warning(f"Judge evaluation failed: {e}")

        # Calculate average judge scores for each model
        for model_id, result in self.results.items():
            if model_id == "analysis" or "error" in result:
                continue

            evaluations = all_evaluations.get(model_id, [])
            if evaluations:
                # Group by criterion
                criterion_scores = defaultdict(list)
                for eval in evaluations:
                    criterion_scores[eval["criterion"]].append(eval["score"])

                # Calculate averages
                result["judge_scores"] = {
                    "criteria_scores": {
                        criterion: sum(scores) / len(scores)
                        for criterion, scores in criterion_scores.items()
                    }
                }

                # Overall score
                all_scores = [s for scores in criterion_scores.values() for s in scores]
                result["judge_scores"]["overall_score"] = sum(all_scores) / len(all_scores) if all_scores else 0

                # Normalize to 0-1 for consistency
                max_score = self.judge_config.scoring_scale[1]
                result["judge_scores"]["overall_score_normalized"] = result["judge_scores"]["overall_score"] / max_score

    async def _get_judge_scores(
        self,
        question: str,
        expected: str,
        actual: str,
        context: list[str]
    ) -> dict[str, Any]:
        """Get judge evaluation scores with bias mitigation."""

        # Build judge prompt
        criteria_str = "\n".join([f"- {c}" for c in self.judge_config.criteria])
        min_score, max_score = self.judge_config.scoring_scale

        prompt = f"""You are an expert evaluator assessing the quality of AI model responses.

Evaluation Criteria:
{criteria_str}

Scoring Scale: {min_score} (worst) to {max_score} (best)

"""

        # Add few-shot examples if enabled
        if self.judge_config.use_few_shot:
            prompt += """Example Evaluations:

Question: What is the capital of France?
Expected: The capital of France is Paris.
Response: Paris is the capital city of France.
Evaluation:
- accuracy: 5 (Completely correct)
- relevance: 5 (Directly answers the question)
- coherence: 5 (Clear and well-structured)
- completeness: 4 (Correct but minimal detail)
- conciseness: 5 (Appropriately brief)

Question: Explain photosynthesis
Expected: Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.
Response: Plants make food from sun.
Evaluation:
- accuracy: 2 (Oversimplified and incomplete)
- relevance: 3 (Related but insufficient)
- coherence: 3 (Too brief to assess structure)
- completeness: 1 (Missing key components)
- conciseness: 5 (Very concise, perhaps too much)

"""

        # Add context if available
        if context:
            prompt += f"Context: {' '.join(context)}\n\n"

        # Add the evaluation task
        prompt += f"""Now evaluate this response:

Question: {question}
Expected Answer: {expected}
Model Response: {actual}

"""

        if self.judge_config.use_cot:
            prompt += "First, provide step-by-step reasoning for each criterion, then assign scores.\n"

        prompt += f"""Provide your evaluation in this JSON format:
{{
    "reasoning": {{
        "accuracy": "Reasoning for accuracy score",
        "relevance": "Reasoning for relevance score",
        "coherence": "Reasoning for coherence score",
        "completeness": "Reasoning for completeness score",
        "conciseness": "Reasoning for conciseness score"
    }},
    "scores": {{
        "accuracy": {min_score}-{max_score},
        "relevance": {min_score}-{max_score},
        "coherence": {min_score}-{max_score},
        "completeness": {min_score}-{max_score},
        "conciseness": {min_score}-{max_score}
    }}
}}"""

        # Get evaluation
        response = await ask(
            prompt=prompt,
            model=self.judge_config.model,
            temperature=self.judge_config.temperature,
            response_format={"type": "json_object"}
        )

        try:
            evaluation = json.loads(response)

            # Handle position bias if enabled
            if self.judge_config.position_swap:
                # Swap expected and actual, re-evaluate
                swap_prompt = prompt.replace(
                    f"Expected Answer: {expected}\nModel Response: {actual}",
                    f"Expected Answer: {actual}\nModel Response: {expected}"
                )

                swap_response = await ask(
                    prompt=swap_prompt,
                    model=self.judge_config.model,
                    temperature=self.judge_config.temperature,
                    response_format={"type": "json_object"}
                )

                swap_eval = json.loads(swap_response)

                # Average the scores (inverting the swapped scores appropriately)
                final_scores = {}
                for criterion in self.judge_config.criteria:
                    orig_score = evaluation["scores"][criterion]
                    # For swapped eval, we need to consider what a high score means
                    # If model was rated high when in expected position, that's actually bad
                    swap_score = max_score + min_score - swap_eval["scores"][criterion]
                    final_scores[criterion] = (orig_score + swap_score) / 2

                evaluation["scores"] = final_scores
                evaluation["position_swap_applied"] = True

            return {
                "criteria_scores": evaluation["scores"],
                "reasoning": evaluation.get("reasoning", {}),
                "overall_score": sum(evaluation["scores"].values()) / len(evaluation["scores"])
            }

        except Exception as e:
            logger.error(f"Failed to parse judge response: {e}")
            # Return neutral scores
            return {
                "criteria_scores": dict.fromkeys(self.judge_config.criteria, (min_score + max_score) / 2),
                "reasoning": {},
                "overall_score": (min_score + max_score) / 2
            }

    def _analyze_results(self) -> dict[str, Any]:
        """Analyze results to find best models."""
        analysis = {
            "models_evaluated": 0,
            "models_passed_threshold": 0,
            "best_model": None,
            "smallest_accurate_model": None,
            "best_local_model": None,
            "best_cloud_model": None,
            "rankings": []
        }

        # Collect valid results
        valid_results = []
        for model_id, result in self.results.items():
            if model_id == "analysis" or "error" in result:
                continue

            judge_scores = result.get("judge_scores", {})
            metrics = result.get("metrics", {})
            model_info = result.get("model_info", {})

            if judge_scores and "overall_score_normalized" in judge_scores:
                composite_score = judge_scores["overall_score_normalized"]

                # Factor in response rate
                if metrics.get("response_rate", 0) < 0.8:
                    composite_score *= metrics["response_rate"]

                valid_results.append({
                    "model_id": model_id,
                    "parameter_count": model_info.get("estimated_params", 0),
                    "is_local": model_info.get("is_local", False),
                    "provider": model_info.get("provider", "unknown"),
                    "composite_score": composite_score,
                    "judge_scores": judge_scores,
                    "metrics": metrics,
                    "passed_threshold": composite_score >= self.target_accuracy
                })

        analysis["models_evaluated"] = len(valid_results)

        # Sort by composite score
        valid_results.sort(key=lambda x: x["composite_score"], reverse=True)
        analysis["rankings"] = valid_results

        if valid_results:
            analysis["best_model"] = valid_results[0]["model_id"]

            # Find best local and cloud models
            local_models = [r for r in valid_results if r["is_local"]]
            cloud_models = [r for r in valid_results if not r["is_local"]]

            if local_models:
                analysis["best_local_model"] = local_models[0]["model_id"]
            if cloud_models:
                analysis["best_cloud_model"] = cloud_models[0]["model_id"]

        # Find smallest model that passes threshold
        passed_models = [r for r in valid_results if r["passed_threshold"]]
        analysis["models_passed_threshold"] = len(passed_models)

        if passed_models:
            # Sort by size (smallest first)
            passed_with_size = [m for m in passed_models if m["parameter_count"] and m["parameter_count"] > 0]
            if passed_with_size:
                passed_with_size.sort(key=lambda x: x["parameter_count"])
                analysis["smallest_accurate_model"] = passed_with_size[0]["model_id"]

        return analysis

    def _print_models_table(self, models: list[LiteLLMModel]):
        """Print table of models to be evaluated."""
        table = Table(title="Models to Evaluate")
        table.add_column("Model ID", style="cyan")
        table.add_column("Provider", style="yellow")
        table.add_column("Size (B)", style="magenta")
        table.add_column("Type", style="green")

        for model in models:
            size = f"{model.estimated_params}" if model.estimated_params else "Unknown"
            model_type = "Local" if model.is_local else "Cloud"
            table.add_row(
                model.model_id,
                model.provider,
                size,
                model_type
            )

        self.console.print(table)
        self.console.print(f"\nDataset: {self.dataset_path}")
        self.console.print(f"Samples: {len(self.dataset)}")
        self.console.print(f"Target accuracy: {self.target_accuracy}\n")

    def _print_summary(self):
        """Print evaluation summary with judge scores."""
        analysis = self.results.get("analysis", {})

        # Create detailed results table
        table = Table(title="Model Evaluation Results (with Judge Scores)")
        table.add_column("Rank", style="white")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="yellow")
        table.add_column("Size (B)", style="magenta")
        table.add_column("Score", style="blue")

        # Add columns for each criterion
        for criterion in self.judge_config.criteria:
            table.add_column(criterion.capitalize()[:3], style="green")

        table.add_column("Status", style="bold")

        for idx, ranking in enumerate(analysis.get("rankings", [])[:15], 1):
            status = "" if ranking["passed_threshold"] else ""
            size = f"{ranking['parameter_count']}" if ranking['parameter_count'] else "?"

            # Get criterion scores
            criterion_scores = []
            judge_scores = ranking.get("judge_scores", {}).get("criteria_scores", {})
            for criterion in self.judge_config.criteria:
                score = judge_scores.get(criterion, 0)
                criterion_scores.append(f"{score:.1f}")

            table.add_row(
                str(idx),
                ranking["model_id"],
                ranking["provider"],
                size,
                f"{ranking['composite_score']:.3f}",
                *criterion_scores,
                status
            )

        self.console.print("\n")
        self.console.print(table)

        # Print analysis summary
        self.console.print("\n[bold] Analysis Summary:[/bold]")
        self.console.print(f"Models evaluated: {analysis['models_evaluated']}")
        self.console.print(f"Models passing threshold: {analysis['models_passed_threshold']}")

        # Recommendations panel
        recommendations = []

        if analysis.get("best_model"):
            recommendations.append(f"[bold green] Best overall:[/bold green] {analysis['best_model']}")

        if analysis.get("smallest_accurate_model"):
            recommendations.append(f"[bold cyan] Smallest accurate:[/bold cyan] {analysis['smallest_accurate_model']}")

        if analysis.get("best_local_model"):
            recommendations.append(f"[bold yellow] Best local:[/bold yellow] {analysis['best_local_model']}")

        if analysis.get("best_cloud_model"):
            recommendations.append(f"[bold blue]☁️  Best cloud:[/bold blue] {analysis['best_cloud_model']}")

        if recommendations:
            self.console.print(Panel(
                "\n".join(recommendations),
                title="Recommendations",
                border_style="green"
            ))

        self.console.print(f"\n[bold]Results saved to:[/bold] {self.output_dir}")

    def _save_results(self):
        """Save evaluation results."""
        # Save full results
        results_path = self.output_dir / "litellm_evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save rankings as CSV
        analysis = self.results.get("analysis", {})
        rankings = analysis.get("rankings", [])

        if rankings:
            df_data = []
            for r in rankings:
                judge_scores = r.get("judge_scores", {}).get("criteria_scores", {})
                row = {
                    "model_id": r["model_id"],
                    "provider": r["provider"],
                    "size_billions": r["parameter_count"],
                    "composite_score": r["composite_score"],
                    "passed_threshold": r["passed_threshold"],
                    **{f"judge_{k}": v for k, v in judge_scores.items()}
                }
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(self.output_dir / "model_rankings_with_judges.csv", index=False)

    def _generate_dashboard(self):
        """Generate evaluation dashboard."""
        # Reuse the multi-model dashboard with judge scores
        from .multi_model_evaluator import MultiModelDashboard

        dashboard_data = {
            "title": "LiteLLM Universal Model Evaluation",
            "generation_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "analysis": self.results.get("analysis", {}),
            "models": self.results,
            "target_accuracy": self.target_accuracy,
            "judge_config": self.judge_config.__dict__,
            "metadata": {
                "dataset": self.dataset_path,
                "samples_evaluated": len(self.dataset),
                "judge_model": self.judge_config.model,
                "criteria": self.judge_config.criteria
            }
        }

        dashboard = MultiModelDashboard(dashboard_data, str(self.output_dir))
        dashboard.generate()
