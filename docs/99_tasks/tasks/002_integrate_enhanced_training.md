# Task 002: Train Model with Enhanced Thinking Data

**Test ID**: enhanced_training_002
**Module**: unsloth.training.enhanced_trainer
**Goal**: Train LoRA adapter using enhanced thinking data

## Working Code Example

```python
# COPY THIS WORKING PATTERN:
from pathlib import Path
from unsloth.core.enhanced_config import EnhancedTrainingConfig
from unsloth.training.enhanced_trainer import EnhancedUnslothTrainer

def train_with_enhanced_thinking():
    # Configure training for enhanced data
    config = EnhancedTrainingConfig(
        # Model
        model_name="unsloth/Phi-3.5-mini-instruct",
        
        # Use enhanced dataset
        dataset_source="arangodb",
        dataset_path="./data/qa_enhanced.jsonl",
        
        # Optimized for iterative thinking patterns
        lora_r=32,  # Higher rank for complex reasoning
        lora_alpha=64,
        lora_dropout=0.1,  # More dropout for generalization
        
        # Training settings
        num_train_epochs=5,  # More epochs for complex patterns
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        
        # Enhanced features
        neftune_noise_alpha=5,  # Help with reasoning diversity
        group_by_length=True,
        
        # Output
        output_dir="./outputs/enhanced_thinking_model",
        tensorboard_log_dir="./outputs/tensorboard/enhanced_run"
    )
    
    # Train
    trainer = EnhancedUnslothTrainer(config)
    results = trainer.train()
    
    return results

# Run it:
result = train_with_enhanced_thinking()
print(f"Training time: {result['training_time']:.2f}s")
print(f"Final loss: {result['train_result'].get('train_loss', 'N/A')}")
```

## Test Details

**Enhanced Dataset Characteristics**:
```json
{
  "thinking_field_stats": {
    "average_length": 450,  // vs 150 original
    "contains_iterations": true,
    "contains_hints": true,
    "self_correction_examples": 0.8  // 80% show correction
  }
}
```

**Training Configuration Differences**:
```python
# Standard thinking:
lora_r=16, epochs=3, lr=5e-5

# Enhanced thinking (complex patterns):
lora_r=32, epochs=5, lr=2e-4
```

**Run Command**:
```bash
# Train on enhanced data
unsloth-cli train \
  --enhanced \
  --dataset ./data/qa_enhanced.jsonl \
  --model unsloth/Phi-3.5-mini-instruct \
  --epochs 5 \
  --lora-r 32 \
  --lr 2e-4
```

**Monitor Training**:
```bash
# TensorBoard
tensorboard --logdir ./outputs/tensorboard/enhanced_run
```

## Common Issues & Solutions

### Issue 1: OOM with larger LoRA rank
```python
# Solution: Reduce batch size or use gradient checkpointing
config = EnhancedTrainingConfig(
    per_device_train_batch_size=1,  # Reduce from 2
    gradient_accumulation_steps=8,   # Increase to maintain effective batch
    use_gradient_checkpointing="unsloth"
)
```

### Issue 2: Overfitting on iteration patterns
```python
# Solution: Increase regularization
config = EnhancedTrainingConfig(
    lora_dropout=0.15,  # Increase from 0.1
    weight_decay=0.05,  # Increase from 0.01
    neftune_noise_alpha=10  # More noise
)
```

### Issue 3: Slow convergence
```python
# Solution: Adjust learning rate schedule
config = EnhancedTrainingConfig(
    learning_rate=5e-4,  # Higher initial LR
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.1  # Longer warmup
)
```

## Validation Requirements

```python
# Training succeeds when:
assert Path("./outputs/enhanced_thinking_model/final_adapter").exists(), "Adapter saved"
assert result['train_result']['train_loss'] < 2.0, "Loss converged"
assert result['training_time'] < 7200, "Training under 2 hours"

# Check TensorBoard logs
tb_log_dir = Path("./outputs/tensorboard/enhanced_run")
assert tb_log_dir.exists(), "TensorBoard logs created"
assert len(list(tb_log_dir.glob("events.out.tfevents.*"))) > 0, "Events logged"
```