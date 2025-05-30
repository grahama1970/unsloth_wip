# Task 005: Integrate RunPod for Large Model Training

**Test ID**: runpod_training_005
**Module**: unsloth.training.runpod_trainer
**Goal**: Enable training on RunPod for models beyond local GPU capacity

## Working Code Example

```python
# COPY THIS WORKING PATTERN:
import os
import asyncio
from pathlib import Path
import runpod
from unsloth.training.runpod_trainer import RunPodTrainingConfig, RunPodTrainer

async def train_large_model_on_runpod():
    # Configure RunPod training
    config = RunPodTrainingConfig(
        # RunPod settings
        runpod_api_key=os.getenv("RUNPOD_API_KEY"),
        gpu_type="H100 PCIe",  # For 70B models
        pod_name="unsloth-training-70b",
        
        # Model settings
        model_name="meta-llama/Llama-2-70b-hf",
        max_seq_length=4096,
        
        # Dataset
        dataset_path="./data/qa_enhanced_large.jsonl",
        dataset_source="arangodb",
        
        # LoRA for 70B model
        lora_r=128,  # Higher rank for larger model
        lora_alpha=256,
        lora_dropout=0.1,
        
        # Training settings
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Small batch for 70B
        gradient_accumulation_steps=16,  # Effective batch = 16
        learning_rate=1e-4,
        
        # RunPod storage
        volume_size_gb=500,  # For model + checkpoints
        output_mount_path="/workspace/outputs",
        
        # Advanced features
        use_flash_attention_2=True,
        gradient_checkpointing=True,
        cpu_offload=True  # Offload optimizer states
    )
    
    # Initialize RunPod trainer
    trainer = RunPodTrainer(config)
    
    # Start training pod
    pod_info = await trainer.start_training_pod()
    print(f"Training pod started: {pod_info['id']}")
    print(f"API endpoint: {pod_info['api_base']}")
    
    # Upload dataset to pod
    await trainer.upload_dataset()
    
    # Start training job
    job_id = await trainer.start_training()
    print(f"Training job started: {job_id}")
    
    # Monitor progress
    async for progress in trainer.monitor_training(job_id):
        print(f"Epoch: {progress['epoch']}, Loss: {progress['loss']:.4f}")
        
    # Download results
    adapter_path = await trainer.download_adapter()
    print(f"Adapter downloaded to: {adapter_path}")
    
    # Cleanup
    await trainer.stop_pod()
    
    return adapter_path

# Run it:
adapter_path = asyncio.run(train_large_model_on_runpod())
print(f"Training complete! Adapter at: {adapter_path}")
```

## Test Details

**RunPod GPU Requirements**:
```json
{
  "model_size": {
    "7B": ["RTX 4090", "RTX A6000", "A100 PCIe"],
    "13B": ["A100 PCIe", "A100 SXM"],
    "30B": ["A100 SXM", "H100 PCIe"],
    "70B": ["H100 PCIe", "H100 SXM", "H100 NVL"]
  },
  "memory_requirements": {
    "7B_4bit": "16GB VRAM",
    "13B_4bit": "24GB VRAM", 
    "30B_4bit": "48GB VRAM",
    "70B_4bit": "80GB VRAM"
  }
}
```

**RunPod Pod Configuration**:
```python
pod_config = {
    "name": "unsloth-training-70b",
    "image_name": "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel",
    "gpu_type_id": "NVIDIA H100 PCIe",
    "cloud_type": "SECURE",
    "volume_in_gb": 500,
    "container_disk_in_gb": 100,
    "docker_args": "python train_unsloth.py",
    "ports": "8888/http,6006/http",  # Jupyter + TensorBoard
    "env": {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY")
    }
}
```

**Run Command**:
```bash
# Configure RunPod
export RUNPOD_API_KEY="your_key_here"

# Run training
unsloth-cli train-runpod \
  --model meta-llama/Llama-2-70b-hf \
  --dataset ./data/qa_enhanced_large.jsonl \
  --gpu-type "H100 PCIe" \
  --epochs 3
```

## Common Issues & Solutions

### Issue 1: GPU not available
```python
# Solution: Fallback to alternative GPUs
config = RunPodTrainingConfig(
    preferred_gpus=["H100 SXM", "H100 PCIe", "A100 SXM"],
    fallback_gpus=["A100 PCIe", "RTX A6000"]  # For smaller configs
)
```

### Issue 2: Dataset upload timeout
```python
# Solution: Use chunked upload
async def upload_large_dataset(trainer, dataset_path):
    chunk_size = 100_000  # 100MB chunks
    
    with open(dataset_path, 'rb') as f:
        chunk_num = 0
        while chunk := f.read(chunk_size):
            await trainer.upload_chunk(chunk, chunk_num)
            chunk_num += 1
```

### Issue 3: OOM during training
```python
# Solution: Aggressive memory optimization
config = RunPodTrainingConfig(
    # Reduce batch size
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    
    # Enable all optimizations
    gradient_checkpointing=True,
    cpu_offload=True,
    optimizer_offload=True,
    
    # Use FSDP for model parallelism
    use_fsdp=True,
    fsdp_config={
        "sharding_strategy": "FULL_SHARD",
        "cpu_offload": True
    }
)
```

## Validation Requirements

```python
# RunPod training succeeds when:

# 1. Pod starts successfully
assert pod_info['status'] == 'RUNNING', "Pod is running"
assert 'api_base' in pod_info, "API endpoint available"

# 2. Training completes
assert job_status['status'] == 'completed', "Training finished"
assert job_status['final_loss'] < 2.0, "Loss converged"

# 3. Adapter downloads
assert adapter_path.exists(), "Adapter downloaded"
adapter_size_mb = adapter_path.stat().st_size / 1024 / 1024
assert adapter_size_mb > 100, "Adapter has reasonable size"

# 4. Costs are reasonable
assert pod_info['total_cost'] < 100, "Training cost under $100"
```