# RunPod Integration Notes

## Current API Limitations (as of 2024)

The RunPod Python SDK has limited functionality compared to what's needed for a complete training pipeline:

### What's Available:
- `runpod.create_pod(name, image, gpu_type)` - Basic pod creation
- `runpod.get_pods()` - List pods
- `runpod.get_pod(pod_id)` - Get pod details
- `runpod.stop_pod(pod_id)` - Stop a pod
- `runpod.resume_pod(pod_id)` - Resume a pod
- `runpod.terminate_pod(pod_id)` - Terminate a pod
- `runpod.get_gpus()` - List available GPUs

### What's Missing:
1. **File Upload/Download**: No direct API methods for file transfer
2. **Command Execution**: No exec API to run commands on pods
3. **Advanced Pod Configuration**: Limited options for volumes, ports, env vars
4. **Progress Monitoring**: No built-in way to monitor training progress

## Alternative Approaches

### 1. RunPod Serverless (Recommended)
Instead of managing pods directly, use RunPod's serverless endpoints:

```python
import runpod

# Deploy as serverless function
def train_model(job):
    config = job["input"]
    # Training logic here
    return {"status": "completed", "model_url": "..."}

runpod.serverless.start({"handler": train_model})
```

### 2. Use runpodctl CLI
Wrap the CLI commands for more functionality:

```python
import subprocess

def create_pod_with_volume():
    cmd = [
        "runpodctl", "create", "pods",
        "--name", "training-pod",
        "--gpuType", "NVIDIA A100",
        "--imageName", "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel",
        "--volumeSize", "100",
        "--ports", "8888/http,6006/http"
    ]
    subprocess.run(cmd)
```

### 3. Pre-built Training Image
Create a Docker image with everything pre-installed:

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel

# Install Unsloth and dependencies
RUN pip install unsloth[colab-new] loguru tensorboard

# Copy training script
COPY train.py /workspace/train.py

# Auto-start training
CMD ["python", "/workspace/train.py"]
```

### 4. Network Volumes for Data Transfer
Use RunPod's network volumes for persistent storage:

1. Create a network volume via web UI
2. Mount it when creating the pod
3. Upload data to the volume before training
4. Download results after training

## Recommended Workflow

Given the API limitations, here's a practical approach:

1. **Prepare**: Package training code and data
2. **Deploy**: Use serverless endpoints for training jobs
3. **Monitor**: Use logging services or custom endpoints
4. **Retrieve**: Download results via network volumes or cloud storage

## Example Serverless Training

```python
# serverless_train.py
import runpod
import torch
from unsloth import FastLanguageModel
from huggingface_hub import upload_folder

def train_handler(job):
    """Serverless training handler."""
    config = job["input"]
    
    # Download dataset from URL
    dataset_url = config["dataset_url"]
    
    # Train model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True
    )
    
    # ... training logic ...
    
    # Upload to HuggingFace
    upload_folder(
        folder_path="./outputs",
        repo_id=config["hub_repo_id"],
        token=config["hf_token"]
    )
    
    return {
        "status": "success",
        "hub_url": f"https://huggingface.co/{config['hub_repo_id']}"
    }

# Deploy this as a serverless endpoint
runpod.serverless.start({"handler": train_handler})
```

This approach is more reliable and scalable than managing pods directly.