"""Multi-model evaluation for finding the smallest accurate model."""
Module: multi_model_evaluator.py
Description: Data models and schemas for multi model evaluator

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import EvaluationConfig
from .evaluator import ModelEvaluator


@dataclass
class ModelCandidate:
    """Represents a model candidate for evaluation."""
    model_id: str
    parameter_count: float  # in billions
    min_gpu_memory: int  # in GB
    recommended_batch_size: int = 8
    load_in_4bit: bool = True
    notes: str = ""


class MultiModelEvaluator:
    """Evaluate multiple models to find the smallest accurate one."""

    # Pre-defined model candidates with sizes
    RECOMMENDED_MODELS = [
        ModelCandidate("unsloth/tinyllama-1.1b", 1.1, 6),
        ModelCandidate("unsloth/Llama-3.2-1B-Instruct", 1.0, 8),
        ModelCandidate("unsloth/Phi-3.5-mini-instruct", 3.8, 12),
        ModelCandidate("unsloth/Llama-3.2-3B-Instruct", 3.0, 16),
        ModelCandidate("unsloth/gemma-2b-it", 2.0, 10),
        ModelCandidate("unsloth/Qwen2.5-1.5B-Instruct", 1.5, 8),
        ModelCandidate("unsloth/mistral-7b-instruct-v0.3", 7.0, 24, load_in_4bit=True),
        ModelCandidate("unsloth/Meta-Llama-3.1-8B-Instruct", 8.0, 24, load_in_4bit=True),
        ModelCandidate("unsloth/gemma-2-9b-it", 9.0, 32, load_in_4bit=True),
    ]

    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "./model_selection_results",
        max_samples: int | None = 100,
        target_accuracy: float = 0.8,
        judge_model: str = "gpt-4"
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.target_accuracy = target_accuracy
        self.judge_model = judge_model
        self.console = Console()
        self.results = {}

    async def evaluate_all_models(
        self,
        model_candidates: list[ModelCandidate] | None = None,
        metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Evaluate all model candidates."""

        if model_candidates is None:
            model_candidates = self.RECOMMENDED_MODELS

        if metrics is None:
            metrics = ["answer_relevancy", "faithfulness", "hallucination"]

        self.console.print(f"[bold cyan]🔍 Evaluating {len(model_candidates)} models[/bold cyan]")
        self.console.print(f"Target accuracy: {self.target_accuracy}")
        self.console.print(f"Dataset: {self.dataset_path}")
        self.console.print(f"Samples per model: {self.max_samples or 'all'}\n")

        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.console.print(f"[yellow]GPU Memory: {gpu_memory:.1f} GB[/yellow]\n")
        else:
            gpu_memory = 0
            self.console.print("[yellow]No GPU detected, evaluation will be slow[/yellow]\n")

        # Filter models by GPU memory
        viable_models = [
            m for m in model_candidates
            if gpu_memory == 0 or m.min_gpu_memory <= gpu_memory * 0.9  # 90% safety margin
        ]

        if len(viable_models) < len(model_candidates):
            skipped = len(model_candidates) - len(viable_models)
            self.console.print(f"[yellow]Skipping {skipped} models due to GPU memory constraints[/yellow]\n")

        # Evaluate each model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:

            eval_task = progress.add_task(
                "[cyan]Evaluating models...",
                total=len(viable_models)
            )

            for model in viable_models:
                progress.update(
                    eval_task,
                    description=f"[cyan]Evaluating {model.model_id} ({model.parameter_count}B)..."
                )

                try:
                    # Configure evaluation for this model
                    config = EvaluationConfig(
                        base_model_path=model.model_id,
                        dataset_path=self.dataset_path,
                        max_samples=self.max_samples,
                        load_in_4bit=model.load_in_4bit,
                        output_dir=str(self.output_dir / model.model_id.replace("/", "_")),
                        use_judge_model=True,
                        judge_config={
                            "model_name": self.judge_model,
                            "temperature": 0.0,
                            "criteria": ["accuracy", "relevance", "coherence"]
                        },
                        generate_html_report=False,  # We'll create a combined report
                        generate_mlflow_run=False,
                        compare_models=False  # Single model eval
                    )

                    # Only enable requested metrics
                    config.metrics = [
                        m for m in config.metrics
                        if m.name in metrics
                    ]

                    # Run evaluation
                    evaluator = ModelEvaluator(config)
                    result = await evaluator.evaluate_all()

                    # Store results
                    self.results[model.model_id] = {
                        "model_info": {
                            "parameter_count": model.parameter_count,
                            "min_gpu_memory": model.min_gpu_memory,
                        },
                        "evaluation": result["base_model"],
                        "timestamp": datetime.utcnow().isoformat()
                    }

                except Exception as e:
                    logger.error(f"Failed to evaluate {model.model_id}: {e}")
                    self.results[model.model_id] = {
                        "model_info": {
                            "parameter_count": model.parameter_count,
                            "min_gpu_memory": model.min_gpu_memory,
                        },
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }

                progress.advance(eval_task)

                # Clean GPU memory
                torch.cuda.empty_cache()

        # Analyze and rank models
        self.results["analysis"] = self._analyze_results()

        # Save results
        self._save_results()

        # Generate comparison dashboard
        self._generate_comparison_dashboard()

        # Print summary
        self._print_summary()

        return self.results

    def _analyze_results(self) -> dict[str, Any]:
        """Analyze results to find best model."""
        analysis = {
            "models_evaluated": 0,
            "models_passed_threshold": 0,
            "best_model": None,
            "smallest_accurate_model": None,
            "rankings": []
        }

        # Collect valid results
        valid_results = []
        for model_id, result in self.results.items():
            if model_id == "analysis" or "error" in result:
                continue

            metrics = result["evaluation"].get("metrics", {})
            judge_scores = result["evaluation"].get("judge_scores", {})

            if metrics or judge_scores:
                # Calculate composite score
                scores = []

                # Metric scores (normalize to 0-1)
                if "answer_relevancy" in metrics:
                    scores.append(metrics["answer_relevancy"])
                if "faithfulness" in metrics:
                    scores.append(metrics["faithfulness"])
                if "hallucination" in metrics:
                    scores.append(1 - metrics["hallucination"])  # Invert hallucination

                # Judge score (normalize from 0-10 to 0-1)
                if judge_scores.get("overall_score"):
                    scores.append(judge_scores["overall_score"] / 10)

                if scores:
                    composite_score = sum(scores) / len(scores)

                    valid_results.append({
                        "model_id": model_id,
                        "parameter_count": result["model_info"]["parameter_count"],
                        "composite_score": composite_score,
                        "metrics": metrics,
                        "judge_scores": judge_scores,
                        "passed_threshold": composite_score >= self.target_accuracy
                    })

        analysis["models_evaluated"] = len(valid_results)

        # Sort by composite score (best first)
        valid_results.sort(key=lambda x: x["composite_score"], reverse=True)
        analysis["rankings"] = valid_results

        if valid_results:
            analysis["best_model"] = valid_results[0]["model_id"]

        # Find smallest model that passes threshold
        passed_models = [r for r in valid_results if r["passed_threshold"]]
        analysis["models_passed_threshold"] = len(passed_models)

        if passed_models:
            # Sort by size (smallest first)
            passed_models.sort(key=lambda x: x["parameter_count"])
            analysis["smallest_accurate_model"] = passed_models[0]["model_id"]

        return analysis

    def _save_results(self):
        """Save all results to JSON."""
        results_path = self.output_dir / "multi_model_evaluation.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Also save rankings as CSV
        analysis = self.results.get("analysis", {})
        rankings = analysis.get("rankings", [])

        if rankings:
            df = pd.DataFrame(rankings)
            df.to_csv(self.output_dir / "model_rankings.csv", index=False)

    def _generate_comparison_dashboard(self):
        """Generate unified comparison dashboard."""
        # Prepare data for dashboard
        dashboard_data = {
            "title": "Multi-Model Evaluation Results",
            "generation_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "analysis": self.results.get("analysis", {}),
            "models": self.results,
            "target_accuracy": self.target_accuracy,
            "metadata": {
                "dataset": self.dataset_path,
                "samples_per_model": self.max_samples
            }
        }

        # Create custom dashboard
        dashboard = MultiModelDashboard(dashboard_data, str(self.output_dir))
        dashboard.generate()

    def _print_summary(self):
        """Print evaluation summary."""
        analysis = self.results.get("analysis", {})

        # Create results table
        table = Table(title="Model Evaluation Results")
        table.add_column("Model", style="cyan")
        table.add_column("Size (B)", style="yellow")
        table.add_column("Score", style="magenta")
        table.add_column("Judge", style="blue")
        table.add_column("Status", style="green")

        for ranking in analysis.get("rankings", []):
            status = "✅ Pass" if ranking["passed_threshold"] else "❌ Fail"
            judge_score = ranking.get("judge_scores", {}).get("overall_score", 0)

            table.add_row(
                ranking["model_id"],
                f"{ranking['parameter_count']}",
                f"{ranking['composite_score']:.3f}",
                f"{judge_score:.1f}/10",
                status
            )

        self.console.print("\n")
        self.console.print(table)

        # Print recommendations
        self.console.print("\n[bold]📊 Analysis Summary:[/bold]")
        self.console.print(f"Models evaluated: {analysis['models_evaluated']}")
        self.console.print(f"Models passing threshold: {analysis['models_passed_threshold']}")

        if analysis.get("best_model"):
            self.console.print(f"\n[bold green]🏆 Best model:[/bold green] {analysis['best_model']}")

        if analysis.get("smallest_accurate_model"):
            self.console.print(f"[bold cyan]📦 Smallest accurate model:[/bold cyan] {analysis['smallest_accurate_model']}")
        else:
            self.console.print(f"[bold red]⚠️  No models passed the accuracy threshold of {self.target_accuracy}[/bold red]")

        self.console.print(f"\n[bold]Results saved to:[/bold] {self.output_dir}")


class MultiModelDashboard:
    """Generate dashboard for multi-model comparison."""

    def __init__(self, data: dict[str, Any], output_dir: str):
        self.data = data
        self.output_dir = Path(output_dir)

    def generate(self) -> str:
        """Generate the HTML dashboard."""

        # Create visualizations
        charts = {}

        # 1. Model size vs accuracy scatter plot
        charts["size_vs_accuracy"] = self._create_size_accuracy_chart()

        # 2. Comprehensive metrics comparison
        charts["metrics_comparison"] = self._create_metrics_chart()

        # 3. Model rankings bar chart
        charts["rankings"] = self._create_rankings_chart()

        # Render template
        template_data = {
            **self.data,
            "charts": charts
        }

        html_content = self._render_template(template_data)

        # Save dashboard
        dashboard_path = self.output_dir / "multi_model_dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(html_content)

        return str(dashboard_path)

    def _create_size_accuracy_chart(self) -> str:
        """Create scatter plot of model size vs accuracy."""
        import plotly.graph_objects as go

        rankings = self.data["analysis"].get("rankings", [])

        if not rankings:
            return ""

        # Prepare data
        sizes = [r["parameter_count"] for r in rankings]
        scores = [r["composite_score"] for r in rankings]
        labels = [r["model_id"].split("/")[-1] for r in rankings]
        colors = ["#10B981" if r["passed_threshold"] else "#EF4444" for r in rankings]

        # Create figure
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(go.Scatter(
            x=sizes,
            y=scores,
            mode='markers+text',
            text=labels,
            textposition="top center",
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        ))

        # Add threshold line
        fig.add_hline(
            y=self.data["target_accuracy"],
            line_dash="dash",
            line_color="#6B7280",
            annotation_text=f"Target Accuracy ({self.data['target_accuracy']})",
            annotation_position="right"
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Model Size vs Accuracy',
                'font': {'size': 24, 'family': 'Inter, system-ui, sans-serif', 'weight': 600}
            },
            xaxis_title="Model Size (Billion Parameters)",
            yaxis_title="Composite Score",
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='white',
            font={'family': 'Inter, system-ui, sans-serif', 'size': 14},
            margin=dict(t=80, b=80, l=80, r=40),
            hovermode='closest'
        )

        return fig.to_html(div_id="size_accuracy", include_plotlyjs=False)

    def _create_metrics_chart(self) -> str:
        """Create comprehensive metrics comparison."""
        import plotly.graph_objects as go

        # Collect all metrics across models
        metrics_data = {}

        for model_id, result in self.data["models"].items():
            if model_id == "analysis" or "error" in result:
                continue

            metrics = result["evaluation"].get("metrics", {})
            judge_scores = result["evaluation"].get("judge_scores", {})

            model_name = model_id.split("/")[-1]

            for metric, value in metrics.items():
                if metric not in metrics_data:
                    metrics_data[metric] = {"models": [], "values": []}
                metrics_data[metric]["models"].append(model_name)
                metrics_data[metric]["values"].append(value)

            # Add judge score
            if judge_scores.get("overall_score"):
                if "judge_score" not in metrics_data:
                    metrics_data["judge_score"] = {"models": [], "values": []}
                metrics_data["judge_score"]["models"].append(model_name)
                metrics_data["judge_score"]["values"].append(judge_scores["overall_score"] / 10)

        # Create subplots
        from plotly.subplots import make_subplots

        num_metrics = len(metrics_data)
        if num_metrics == 0:
            return ""

        fig = make_subplots(
            rows=(num_metrics + 1) // 2,
            cols=2,
            subplot_titles=list(metrics_data.keys()),
            vertical_spacing=0.15
        )

        # Add traces
        for idx, (metric, data) in enumerate(metrics_data.items()):
            row = (idx // 2) + 1
            col = (idx % 2) + 1

            fig.add_trace(
                go.Bar(
                    x=data["models"],
                    y=data["values"],
                    name=metric,
                    marker_color='#4F46E5',
                    showlegend=False
                ),
                row=row,
                col=col
            )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Detailed Metrics Comparison',
                'font': {'size': 24, 'family': 'Inter, system-ui, sans-serif', 'weight': 600}
            },
            height=300 * ((num_metrics + 1) // 2),
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='white',
            font={'family': 'Inter, system-ui, sans-serif', 'size': 12},
            margin=dict(t=100, b=40, l=40, r=40)
        )

        fig.update_xaxes(tickangle=-45)

        return fig.to_html(div_id="metrics_comparison", include_plotlyjs=False)

    def _create_rankings_chart(self) -> str:
        """Create model rankings bar chart."""
        import plotly.graph_objects as go

        rankings = self.data["analysis"].get("rankings", [])[:10]  # Top 10

        if not rankings:
            return ""

        # Prepare data
        models = [r["model_id"].split("/")[-1] for r in rankings]
        scores = [r["composite_score"] for r in rankings]
        colors = ["#10B981" if r["passed_threshold"] else "#EF4444" for r in rankings]

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=models,
            x=scores,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{s:.3f}" for s in scores],
            textposition='outside'
        ))

        # Add threshold line
        fig.add_vline(
            x=self.data["target_accuracy"],
            line_dash="dash",
            line_color="#6B7280",
            annotation_text="Target",
            annotation_position="top"
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Model Rankings by Composite Score',
                'font': {'size': 24, 'family': 'Inter, system-ui, sans-serif', 'weight': 600}
            },
            xaxis_title="Composite Score",
            yaxis_title="",
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='white',
            font={'family': 'Inter, system-ui, sans-serif', 'size': 14},
            margin=dict(t=80, b=60, l=200, r=60),
            height=max(400, len(models) * 40)
        )

        fig.update_yaxes(tickmode='linear')

        return fig.to_html(div_id="rankings", include_plotlyjs=False)

    def _render_template(self, data: dict[str, Any]) -> str:
        """Render the HTML template."""
        from jinja2 import Template

        template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    
    <!-- 2025 Style Guide Compliant CSS -->
    <style>
        :root {
            --color-primary-start: #4F46E5;
            --color-primary-end: #6366F1;
            --color-secondary: #6B7280;
            --color-background: #F9FAFB;
            --color-accent: #10B981;
            --color-danger: #EF4444;
            --font-family-base: 'Inter', system-ui, -apple-system, sans-serif;
            --border-radius-base: 8px;
            --spacing-base: 8px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: var(--font-family-base);
            font-size: 16px;
            line-height: 1.5;
            color: #111827;
            background-color: var(--color-background);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: calc(var(--spacing-base) * 4);
        }
        
        h1 {
            font-size: 3rem;
            margin-bottom: calc(var(--spacing-base) * 2);
            background: linear-gradient(135deg, var(--color-primary-start), var(--color-primary-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2 {
            font-size: 2rem;
            margin-bottom: calc(var(--spacing-base) * 3);
            margin-top: calc(var(--spacing-base) * 6);
            color: #1F2937;
        }
        
        /* Hero Card */
        .hero-card {
            background: linear-gradient(135deg, var(--color-primary-start), var(--color-primary-end));
            color: white;
            padding: calc(var(--spacing-base) * 5);
            border-radius: var(--border-radius-base);
            margin-bottom: calc(var(--spacing-base) * 6);
            box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
        }
        
        .hero-card h2 {
            color: white;
            margin-top: 0;
        }
        
        .hero-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: calc(var(--spacing-base) * 4);
            margin-top: calc(var(--spacing-base) * 4);
        }
        
        .hero-stat {
            text-align: center;
        }
        
        .hero-stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            display: block;
        }
        
        .hero-stat-label {
            font-size: 0.875rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Recommendation Cards */
        .recommendation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: calc(var(--spacing-base) * 3);
            margin-bottom: calc(var(--spacing-base) * 6);
        }
        
        .recommendation-card {
            background: white;
            border-radius: var(--border-radius-base);
            padding: calc(var(--spacing-base) * 3);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 2px solid transparent;
            transition: all 250ms ease;
        }
        
        .recommendation-card.best {
            border-color: var(--color-accent);
        }
        
        .recommendation-card.smallest {
            border-color: var(--color-primary-start);
        }
        
        .recommendation-card h3 {
            font-size: 1.25rem;
            margin-bottom: calc(var(--spacing-base) * 2);
        }
        
        .model-name {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--color-primary-start);
            margin: calc(var(--spacing-base) * 2) 0;
        }
        
        .model-stats {
            display: flex;
            justify-content: space-between;
            margin-top: calc(var(--spacing-base) * 2);
            padding-top: calc(var(--spacing-base) * 2);
            border-top: 1px solid #E5E7EB;
        }
        
        .model-stat {
            text-align: center;
        }
        
        .model-stat-value {
            font-size: 1.25rem;
            font-weight: 600;
            display: block;
        }
        
        .model-stat-label {
            font-size: 0.75rem;
            color: var(--color-secondary);
            text-transform: uppercase;
        }
        
        /* Charts */
        .chart-container {
            background: white;
            border-radius: var(--border-radius-base);
            padding: calc(var(--spacing-base) * 3);
            margin-bottom: calc(var(--spacing-base) * 4);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Rankings Table */
        .rankings-table {
            background: white;
            border-radius: var(--border-radius-base);
            padding: calc(var(--spacing-base) * 3);
            margin-bottom: calc(var(--spacing-base) * 4);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            background-color: #F3F4F6;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            color: #6B7280;
            padding: 16px 12px;
            text-align: left;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #E5E7EB;
        }
        
        tr:hover {
            background-color: #F9FAFB;
        }
        
        .status-pass {
            color: var(--color-accent);
            font-weight: 600;
        }
        
        .status-fail {
            color: var(--color-danger);
            font-weight: 600;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: calc(var(--spacing-base) * 4);
            color: var(--color-secondary);
            font-size: 0.875rem;
        }
    </style>
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- Inter Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p style="color: var(--color-secondary); margin-bottom: calc(var(--spacing-base) * 4);">
                Generated on {{ generation_date }} | Target Accuracy: {{ target_accuracy }}
            </p>
        </header>
        
        <!-- Hero Summary -->
        <div class="hero-card">
            <h2>Evaluation Summary</h2>
            <div class="hero-stats">
                <div class="hero-stat">
                    <span class="hero-stat-value">{{ analysis.models_evaluated }}</span>
                    <span class="hero-stat-label">Models Evaluated</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-value">{{ analysis.models_passed_threshold }}</span>
                    <span class="hero-stat-label">Passed Threshold</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-value">{{ metadata.samples_per_model or "All" }}</span>
                    <span class="hero-stat-label">Samples per Model</span>
                </div>
            </div>
        </div>
        
        <!-- Recommendations -->
        <h2>Recommendations</h2>
        <div class="recommendation-grid">
            {% if analysis.best_model %}
            <div class="recommendation-card best">
                <h3>🏆 Best Overall Model</h3>
                <div class="model-name">{{ analysis.best_model.split('/')[-1] }}</div>
                {% for ranking in analysis.rankings %}
                    {% if ranking.model_id == analysis.best_model %}
                    <div class="model-stats">
                        <div class="model-stat">
                            <span class="model-stat-value">{{ "%.3f"|format(ranking.composite_score) }}</span>
                            <span class="model-stat-label">Score</span>
                        </div>
                        <div class="model-stat">
                            <span class="model-stat-value">{{ ranking.parameter_count }}B</span>
                            <span class="model-stat-label">Size</span>
                        </div>
                        <div class="model-stat">
                            <span class="model-stat-value">{{ "%.1f"|format(ranking.judge_scores.overall_score) }}/10</span>
                            <span class="model-stat-label">Judge</span>
                        </div>
                    </div>
                    {% endif %}
                {% endfor %}
            </div>
            {% endif %}
            
            {% if analysis.smallest_accurate_model %}
            <div class="recommendation-card smallest">
                <h3>📦 Smallest Accurate Model</h3>
                <div class="model-name">{{ analysis.smallest_accurate_model.split('/')[-1] }}</div>
                {% for ranking in analysis.rankings %}
                    {% if ranking.model_id == analysis.smallest_accurate_model %}
                    <div class="model-stats">
                        <div class="model-stat">
                            <span class="model-stat-value">{{ "%.3f"|format(ranking.composite_score) }}</span>
                            <span class="model-stat-label">Score</span>
                        </div>
                        <div class="model-stat">
                            <span class="model-stat-value">{{ ranking.parameter_count }}B</span>
                            <span class="model-stat-label">Size</span>
                        </div>
                        <div class="model-stat">
                            <span class="model-stat-value">{{ "%.1f"|format(ranking.judge_scores.overall_score) }}/10</span>
                            <span class="model-stat-label">Judge</span>
                        </div>
                    </div>
                    {% endif %}
                {% endfor %}
            </div>
            {% endif %}
        </div>
        
        <!-- Charts -->
        {% if charts.size_vs_accuracy %}
        <div class="chart-container">
            <h3>Model Size vs Accuracy</h3>
            {{ charts.size_vs_accuracy|safe }}
        </div>
        {% endif %}
        
        {% if charts.rankings %}
        <div class="chart-container">
            <h3>Model Rankings</h3>
            {{ charts.rankings|safe }}
        </div>
        {% endif %}
        
        {% if charts.metrics_comparison %}
        <div class="chart-container">
            <h3>Detailed Metrics</h3>
            {{ charts.metrics_comparison|safe }}
        </div>
        {% endif %}
        
        <!-- Rankings Table -->
        <div class="rankings-table">
            <h3>Complete Rankings</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Size (B)</th>
                        <th>Composite Score</th>
                        <th>Judge Score</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for ranking in analysis.rankings %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ ranking.model_id }}</td>
                        <td>{{ ranking.parameter_count }}</td>
                        <td>{{ "%.3f"|format(ranking.composite_score) }}</td>
                        <td>{{ "%.1f"|format(ranking.judge_scores.overall_score) }}/10</td>
                        <td class="{{ 'status-pass' if ranking.passed_threshold else 'status-fail' }}">
                            {{ "✅ Pass" if ranking.passed_threshold else "❌ Fail" }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <footer class="footer">
            <p>Multi-Model Evaluation Dashboard • Powered by DeepEval & Unsloth</p>
            <p style="margin-top: var(--spacing-base);">
                <small>2025 Style Guide Compliant</small>
            </p>
        </footer>
    </div>
    
    <!-- Plotly responsive config -->
    <script>
        window.addEventListener('resize', function() {
            const plots = document.querySelectorAll('[id^="size_"], [id^="rankings"], [id^="metrics_"]');
            plots.forEach(plot => {
                Plotly.Plots.resize(plot);
            });
        });
    </script>
</body>
</html>""")

        return template.render(**data)
