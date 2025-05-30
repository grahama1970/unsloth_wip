# Unsloth Pipeline Quick Reference

## 🚀 Quick Start

```bash
# Basic training
python -m src.unsloth.pipeline.complete_training_pipeline \
    --model unsloth/Phi-3.5-mini-instruct \
    --dataset qa_data.jsonl \
    --output ./outputs/training

# With Hugging Face upload
python -m src.unsloth.pipeline.complete_training_pipeline \
    --model unsloth/Phi-3.5-mini-instruct \
    --dataset qa_data.jsonl \
    --output ./outputs/training \
    --hub-id username/model-name

# Force RunPod (for testing)
python -m src.unsloth.pipeline.complete_training_pipeline \
    --model unsloth/llama-3.2-3b-instruct \
    --dataset qa_data.jsonl \
    --output ./outputs/training \
    --force-runpod
```

## 📊 Model Size Guidelines

| Model Size | Local GPU Required | RunPod GPU | Training Time |
|------------|-------------------|------------|---------------|
| 3B-7B      | RTX 3090 (24GB)   | RTX 4090   | 1-2 hours     |
| 13B        | RTX A6000 (48GB)  | A100 40GB  | 2-4 hours     |
| 30B        | A100 80GB         | A100 80GB  | 4-8 hours     |
| 70B        | Not feasible      | H100 80GB  | 8-16 hours    |

## 🧠 Student-Teacher Settings

```python
StudentTeacherConfig(
    teacher_model="anthropic/max",  # Claude for hints
    max_iterations=3,               # Attempts per question
    student_temperature=0.7,        # Exploration
    teacher_temperature=0.8,        # Hint creativity
    use_local_student=False,        # Use API for consistency
)
```

## ⚙️ Key Training Parameters

```python
# LoRA Configuration
r=16                    # Rank (8-32 typical)
lora_alpha=16          # Alpha (usually same as r)
lora_dropout=0.05      # Dropout (0.0-0.1)

# Training
learning_rate=2e-4     # LR (1e-4 to 5e-4)
batch_size=4           # Per GPU (2-8)
gradient_accumulation=4 # Effective batch = 16
num_epochs=3           # Epochs (1-5)

# Memory
load_in_4bit=True      # Always for efficiency
gradient_checkpointing=True  # For larger models
```

## 📁 Output Structure

```
outputs/training_run/
├── enhanced_dataset.jsonl    # Student-teacher enhanced
├── checkpoints/             # Training checkpoints
├── adapter/                 # Final LoRA files
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── tensorboard/            # Training metrics
└── pipeline_results.json   # Complete summary
```

## 🔍 Validation Checks

1. **Basic Inference** - Can generate text
2. **Test Prompts** - Quality responses
3. **Base Comparison** - Improvement over base
4. **Performance** - Speed and memory
5. **File Integrity** - All files present

## 🌐 Environment Variables

```bash
export HF_TOKEN="hf_..."              # Hugging Face
export ANTHROPIC_API_KEY="sk-ant..." # Claude (teacher)
export OPENAI_API_KEY="sk-..."       # GPT-4 (judge)
export RUNPOD_API_KEY="..."          # RunPod (optional)
```

## 🛠️ Common Commands

```bash
# Monitor training
tensorboard --logdir outputs/training/tensorboard

# Test model locally
python -m src.unsloth.validation.model_validator \
    --adapter outputs/training/adapter \
    --base-model unsloth/Phi-3.5-mini-instruct

# Upload existing adapter
python -m src.unsloth.upload.hub_uploader \
    --adapter outputs/training/adapter \
    --model-id username/model-name \
    --base-model unsloth/Phi-3.5-mini-instruct
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch_size, enable gradient_checkpointing |
| Slow Training | Check 4-bit is enabled, reduce max_seq_length |
| Poor Quality | More epochs, check enhancement worked |
| RunPod Issues | Use serverless endpoint instead |

## 📈 Performance Tips

1. **Start Small**: Test with 100 examples first
2. **Monitor Memory**: Use `nvidia-smi` during training
3. **Batch Optimization**: Find largest stable batch size
4. **Early Stopping**: Use validation for early stopping
5. **Checkpoint Often**: Save every 50-100 steps

## 🔗 Key Modules

- `thinking_enhancer.py` - Student-teacher enhancement
- `enhanced_trainer.py` - Local training orchestration
- `runpod_serverless.py` - RunPod deployment
- `model_validator.py` - Validation testing
- `hub_uploader.py` - Hugging Face upload
- `complete_training_pipeline.py` - Full pipeline

## 📚 Further Reading

- [Student-Teacher Approach](./student_teacher_approach.md)
- [Complete Pipeline Guide](./complete_pipeline_guide.md)
- [RunPod Notes](../src/unsloth/training/runpod_notes.md)
- [Task Lists](./tasks/)