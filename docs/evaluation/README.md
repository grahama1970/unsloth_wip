# Unsloth Model Evaluation System

A comprehensive evaluation framework for comparing LLM models before and after LoRA fine-tuning, featuring rich visualizations and judge model assessments.

## Features

- **Multi-metric evaluation**: Perplexity, answer relevancy, hallucination detection, toxicity, bias, and more
- **Judge model integration**: Use GPT-4 or other LLMs to evaluate response quality
- **Beautiful dashboards**: HTML reports following 2025 Style Guide with interactive Plotly charts
- **Model comparison**: Side-by-side comparison of base vs LoRA-adapted models
- **MLflow integration**: Track experiments and compare runs
- **Academic benchmarks**: Integration with lm-evaluation-harness for standard benchmarks

## Quick Start

### CLI Usage

```bash
# Evaluate a single model
unsloth evaluate \
  --base-model unsloth/Phi-3.5-mini-instruct \
  --dataset data/eval_qa.jsonl \
  --output ./evaluation_results

# Compare base model with LoRA adapter
unsloth evaluate \
  --base-model unsloth/Phi-3.5-mini-instruct \
  --lora-model ./outputs/adapter \
  --dataset data/eval_qa.jsonl \
  --judge-model gpt-4 \
  --output ./evaluation_results

# Run specific metrics only
unsloth evaluate \
  --base-model unsloth/Phi-3.5-mini-instruct \
  --dataset data/eval_qa.jsonl \
  --metrics perplexity answer_relevancy \
  --no-html
```

### Python API

```python
import asyncio
from unsloth.evaluation import ModelEvaluator, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    base_model_path="unsloth/Phi-3.5-mini-instruct",
    lora_model_path="./outputs/adapter",
    dataset_path="./data/qa_test.jsonl",
    max_samples=100,
    use_judge_model=True,
    judge_config={
        "model_name": "gpt-4",
        "criteria": ["accuracy", "relevance", "coherence"]
    },
    generate_html_report=True
)

# Run evaluation
evaluator = ModelEvaluator(config)
results = asyncio.run(evaluator.evaluate_all())
```

## Dataset Format

The evaluation dataset should be in JSONL format with the following structure:

```json
{"question": "What is machine learning?", "answer": "Machine learning is...", "context": ["Optional context"]}
{"question": "How do vaccines work?", "answer": "Vaccines work by...", "context": ["Optional context"]}
```

## Available Metrics

### DeepEval Metrics
- **answer_relevancy**: How relevant the answer is to the question
- **faithfulness**: Whether the answer is faithful to the provided context
- **contextual_precision**: Precision of using the provided context
- **contextual_recall**: Recall of important context information
- **hallucination**: Detection of fabricated information
- **toxicity**: Detection of harmful content
- **bias**: Detection of biased responses

### LM-Eval Benchmarks
- **perplexity**: Language modeling performance (via WikiText)
- **hellaswag**: Common sense reasoning
- **mmlu**: Massive Multitask Language Understanding

### Judge Model Evaluation
Configurable criteria evaluated by a judge model (e.g., GPT-4):
- Accuracy
- Relevance
- Coherence
- Fluency
- Custom criteria

## Output

### JSON Results
Detailed results saved to `evaluation_results.json`:
```json
{
  "base_model": {
    "metrics": {...},
    "judge_scores": {...},
    "examples": [...]
  },
  "lora_model": {...},
  "comparison": {
    "improvements": {...},
    "regressions": {...},
    "summary": {...}
  }
}
```

### HTML Dashboard
Interactive dashboard with:
- Metrics comparison bar charts
- Judge evaluation radar charts
- Performance improvement waterfall chart
- Example outputs comparison table
- Summary statistics and recommendations

### CSV Summary
Model comparison summary exported to `comparison_summary.csv`

## Integration with Training Pipeline

The evaluation system integrates seamlessly with the Unsloth training pipeline:

```bash
# 1. Train your model
unsloth train --model unsloth/Phi-3.5-mini-instruct --dataset train.jsonl

# 2. Evaluate the results
unsloth evaluate \
  --base-model unsloth/Phi-3.5-mini-instruct \
  --lora-model ./outputs/adapter \
  --dataset test.jsonl

# 3. View the dashboard
open evaluation_results/evaluation_dashboard.html
```

## Advanced Configuration

### Custom Metrics

```python
config = EvaluationConfig(
    metrics=[
        MetricConfig(name="perplexity", enabled=True),
        MetricConfig(
            name="custom_metric",
            enabled=True,
            params={"threshold": 0.8}
        )
    ]
)
```

### MLflow Tracking

```python
config = EvaluationConfig(
    generate_mlflow_run=True,
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="model_comparison"
)
```

### Judge Model Configuration

```python
config = EvaluationConfig(
    judge_config=JudgeModelConfig(
        model_name="claude-3-opus",
        temperature=0.0,
        criteria=["technical_accuracy", "explanation_quality", "code_correctness"]
    )
)
```

## Tips for Best Results

1. **Dataset Quality**: Use high-quality evaluation datasets that represent your use case
2. **Sample Size**: Use at least 100-500 examples for reliable results
3. **Judge Model**: GPT-4 or Claude-3 work best as judge models
4. **Metrics Selection**: Choose metrics relevant to your application
5. **Baseline Comparison**: Always compare against the base model to measure improvement

## Troubleshooting

- **GPU Memory**: Use `--max-samples` to limit evaluation size
- **API Rate Limits**: The system includes retry logic for judge model calls
- **Missing Dependencies**: Run `pip install deepeval mlflow lm-eval`

## Contributing

To add new metrics or visualizations, see the developer guide in `docs/`.