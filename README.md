# Unsloth Enhanced Training Pipeline

A comprehensive pipeline for training LoRA adapters with student-teacher thinking enhancement, integrating ArangoDB Q&A generation, Claude-powered hints, and automatic deployment to Hugging Face.

## 🌟 Key Features

- **Student-Teacher Enhancement**: Uses the target model as student with Claude (anthropic/max) as teacher
- **Automatic Scaling**: Detects model size and uses RunPod for large models (30B+)
- **Memory Optimized**: 4-bit quantization, gradient checkpointing, Flash Attention 2
- **Complete Pipeline**: From Q&A data to deployed model on Hugging Face
- **Grokking Support**: Optional extended training for superior generalization

## Pipeline Overview

This is part of a larger document processing pipeline:

1. **sparta** - Data ingestion from STIX and other sources
2. **marker** - Document extraction and structuring
3. **arangodb** - Graph database storage and Q&A generation
4. **unsloth** (this module) - Fine-tuning LoRA adapters with student-teacher enhancement

## 🧠 How Student-Teacher Enhancement Works

The key innovation: using the **same model we're training** as the student to capture model-specific reasoning patterns.

```python
# Example iteration
Student (Phi-3.5): "15 * 13 = 180"  # Wrong attempt
Teacher (Claude): "Aha! Try breaking it down: 15 * (10 + 3)"
Student: "Oh wait, 15 * 10 = 150, 15 * 3 = 45, so 195!"  # Corrected
```

## Project Structure

```
unsloth_wip/
├── src/unsloth/
│   ├── core/                    # Core configurations
│   │   ├── enhanced_config.py   # Training configuration
│   │   └── grokking_config.py   # Grokking settings
│   ├── data/                    # Data processing
│   │   ├── loader.py           # Dataset loading
│   │   └── thinking_enhancer.py # Student-teacher enhancement
│   ├── training/               # Training implementations
│   │   ├── enhanced_trainer.py # Local training
│   │   ├── runpod_trainer.py   # RunPod integration
│   │   └── runpod_serverless.py # Serverless training
│   ├── validation/             # Model validation
│   │   └── model_validator.py  # Comprehensive testing
│   ├── upload/                 # Model deployment
│   │   └── hub_uploader.py     # Hugging Face upload
│   ├── pipeline/               # Complete pipeline
│   │   └── complete_training_pipeline.py
│   └── examples/               # Example scripts
├── docs/                       # Documentation
├── archive/                    # Old code for reference
├── .env                        # Environment variables
└── pyproject.toml              # Project configuration
```

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository>
cd unsloth_wip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export ANTHROPIC_API_KEY="your_claude_key"
export OPENAI_API_KEY="your_openai_key"

# Run complete pipeline
python -m src.unsloth.pipeline.complete_training_pipeline \
    --model unsloth/Phi-3.5-mini-instruct \
    --dataset /path/to/qa_data.jsonl \
    --output ./outputs/my_training \
    --hub-id myusername/phi-3.5-enhanced
```

## 📊 Supported Models

| Model | Local Training | RunPod Required | Recommended GPU |
|-------|---------------|-----------------|------------------|
| Phi-3.5 (3.8B) | ✅ | ❌ | RTX 3090 |
| Llama-3.2 (3B) | ✅ | ❌ | RTX 3090 |
| Llama-3.2 (7B) | ✅ | ❌ | RTX 4090 |
| Mistral (7B) | ✅ | ❌ | RTX 4090 |
| Llama-2 (13B) | ✅* | ⭕ | A100 40GB |
| Llama-2 (30B) | ❌ | ✅ | A100 80GB |
| Llama-2 (70B) | ❌ | ✅ | H100 80GB |

*With careful memory management

## 🔧 Configuration Options

### Basic Training
```python
config = EnhancedTrainingConfig(
    model_name="unsloth/Phi-3.5-mini-instruct",
    r=16,                    # LoRA rank
    learning_rate=2e-4,      # Learning rate
    num_train_epochs=3,      # Training epochs
    per_device_train_batch_size=4,
    gradient_checkpointing=True
)
```

### Student-Teacher Settings
```python
student_teacher_config = StudentTeacherConfig(
    teacher_model="anthropic/max",  # Claude for hints
    max_iterations=3,               # Attempts per question
    thinking_format="iterative"     # Show clear iterations
)
```

## 🚀 Advanced Features

### Grokking for Better Generalization
```python
grokking = GrokkingConfig(
    enable_grokking=True,
    grokking_multiplier=30.0,  # 30x epochs
    grokking_weight_decay=0.1
)
```

### RunPod Serverless Deployment
For production training of large models:
```python
# Deploy as serverless endpoint
python src/unsloth/training/runpod_serverless.py
```

### Comprehensive Validation
Automatic validation includes:
- Basic inference testing
- Comparison with base model
- Performance benchmarking
- File integrity checks

## 📈 Monitoring

- **TensorBoard**: `tensorboard --logdir outputs/tensorboard`
- **Progress Tracking**: Check `pipeline_results.json`
- **Real-time Logs**: Uses loguru for detailed logging

## 🌐 Requirements

- Python 3.8+
- CUDA-capable GPU (24GB+ VRAM recommended)
- 100GB+ disk space for models and datasets

### API Keys Required
- `HF_TOKEN`: Hugging Face (for model upload)
- `ANTHROPIC_API_KEY`: Claude API (teacher model)
- `OPENAI_API_KEY`: OpenAI API (judge model)
- `RUNPOD_API_KEY`: RunPod (optional, for large models)

## 📚 Documentation

- [Complete Pipeline Guide](docs/complete_pipeline_guide.md)
- [Student-Teacher Approach](docs/student_teacher_approach.md)
- [Quick Reference](docs/quick_reference.md)
- [Task Lists](docs/tasks/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast and memory-efficient fine-tuning
- [Claude](https://anthropic.com) - Teacher model for hints
- [RunPod](https://runpod.io) - GPU infrastructure for large models

## ⚡ Performance Tips

1. **Start Small**: Test with 100 examples first
2. **Monitor GPU**: Use `nvidia-smi -l 1` during training
3. **Batch Size**: Find the largest stable batch size
4. **Checkpointing**: Save every 50-100 steps
5. **Early Stopping**: Use validation loss for early stopping

## 🐛 Troubleshooting

See [Quick Reference](docs/quick_reference.md#-troubleshooting) for common issues and solutions.

---

Built with ❤️ for efficient LLM fine-tuning