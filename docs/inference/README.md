# Unsloth Inference Module

Comprehensive tools for running inference and testing fine-tuned models with LoRA adapters.

## Features

- **Fast Inference**: Optimized inference engine with 4-bit quantization support
- **Interactive Chat**: Real-time chat interface for testing models
- **Comprehensive Testing**: Automated test suite with multiple categories
- **Judge Evaluation**: GPT-4 powered quality assessment
- **Batch Processing**: Efficient batch inference for multiple prompts
- **LoRA Merging**: Tools to merge adapters with base models
- **Beautiful Reports**: HTML test reports with performance metrics

## Quick Start

### 1. Basic Inference

```bash
# Single prompt
unsloth infer --model ./outputs/adapter \
              --prompt "What is machine learning?"

# With custom parameters
unsloth infer --model ./outputs/adapter \
              --prompt "Explain quantum computing" \
              --temperature 0.3 \
              --max-tokens 500
```

### 2. Interactive Chat

```bash
# Start interactive chat
unsloth infer --model ./outputs/adapter --interactive

# With system prompt
unsloth infer --model ./outputs/adapter \
              --interactive \
              --system "You are a helpful AI assistant specialized in Python programming."
```

### 3. Comprehensive Testing

```bash
# Run full test suite
unsloth test-inference --model ./outputs/adapter

# Test specific categories
unsloth test-inference --model ./outputs/adapter \
                       --category "Math" \
                       --category "Reasoning"

# Use custom judge model
unsloth test-inference --model ./outputs/adapter \
                       --judge claude-3-opus-20240229

# Skip judge evaluation for speed
unsloth test-inference --model ./outputs/adapter --no-judge
```

### 4. Custom Test Cases

Create a `custom_tests.json` file:

```json
[
  {
    "category": "Domain Knowledge",
    "question": "What are the advantages of LoRA fine-tuning?",
    "expected_keywords": ["parameter", "efficient", "memory", "weights"],
    "expected_format": "technical_explanation"
  },
  {
    "category": "Code Generation",
    "question": "Write a Python function to calculate Fibonacci numbers",
    "expected_keywords": ["def", "fibonacci", "return", "if"],
    "expected_format": "python_code"
  }
]
```

Then run:
```bash
unsloth test-inference --model ./outputs/adapter \
                       --custom-tests custom_tests.json
```

## Python API

### Basic Usage

```python
from unsloth.inference import InferenceEngine, GenerationConfig

# Initialize engine
engine = InferenceEngine("./outputs/adapter", load_in_4bit=True)
engine.load_model()

# Generate text
config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

response = engine.generate("What is AI?", config)
print(response)
```

### Batch Processing

```python
questions = [
    "What is machine learning?",
    "Explain neural networks",
    "How does backpropagation work?"
]

responses = engine.generate_batch(questions, config)
for q, a in zip(questions, responses):
    print(f"Q: {q}\nA: {a}\n")
```

### Test Suite

```python
import asyncio
from unsloth.inference import InferenceTestSuite

async def test_model():
    test_suite = InferenceTestSuite(
        model_path="./outputs/adapter",
        use_judge=True,
        judge_model="gpt-4"
    )
    
    results = await test_suite.run_tests(
        categories=["Reasoning", "Math", "Code Generation"]
    )
    
    # Results include performance metrics, judge scores, etc.
    print(f"Overall accuracy: {results['overall_metrics']['success_rate']*100:.1f}%")

asyncio.run(test_model())
```

## Merging LoRA Adapters

For deployment, you may want to merge the LoRA adapter with the base model:

```python
from unsloth.inference import merge_adapter_for_unsloth

# Merge using Unsloth's optimized method
merged_path = merge_adapter_for_unsloth(
    base_model_path="unsloth/Phi-3.5-mini-instruct",
    adapter_path="./outputs/adapter",
    output_path="./merged_model"
)

# Or use standard PEFT merge
from unsloth.inference import merge_lora_adapter

merged_path = merge_lora_adapter(
    base_model_path="unsloth/Phi-3.5-mini-instruct",
    adapter_path="./outputs/adapter",
    output_path="./merged_model",
    push_to_hub="username/my-merged-model"  # Optional
)
```

## Test Categories

The default test suite includes:

1. **Factual Knowledge**: Basic facts and information retrieval
2. **Reasoning**: Logical puzzles and deduction
3. **Math**: Arithmetic and word problems
4. **Code Generation**: Programming tasks
5. **Creative Writing**: Storytelling and creative tasks
6. **Instruction Following**: Format compliance and instructions
7. **Domain Knowledge**: Technical and specialized topics

## Performance Metrics

The test suite tracks:

- **Response Time**: Average inference speed in ms
- **Tokens/Second**: Generation throughput
- **Keyword Accuracy**: Expected terms in responses
- **Format Compliance**: Adherence to requested formats
- **Judge Scores**: Quality ratings from judge model (1-10)

## Output Files

After running tests:

```
inference_test_results/
├── test_results.json      # Detailed results with all responses
├── test_results.csv       # Tabular summary for analysis
└── test_report.html       # Beautiful HTML report
```

## Tips for Best Results

1. **Temperature Tuning**: Lower temperatures (0.3-0.5) for factual tasks, higher (0.7-0.9) for creative
2. **Custom Tests**: Create domain-specific tests for your use case
3. **Judge Selection**: GPT-4 or Claude-3-Opus provide best evaluation quality
4. **Batch Size**: Process multiple prompts together for efficiency
5. **System Prompts**: Use system prompts to guide model behavior

## Troubleshooting

- **Out of Memory**: Use `--load-in-4bit` or reduce batch size
- **Slow Inference**: Ensure GPU is available with `nvidia-smi`
- **Import Errors**: Install with `pip install -e .` from project root