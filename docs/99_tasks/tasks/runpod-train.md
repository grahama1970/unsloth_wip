# /runpod-train

Launches RunPod training for models that exceed local GPU capacity.

## Usage
```bash
/runpod-train --model MODEL_NAME --dataset DATASET --gpu GPU_TYPE [OPTIONS]
```

## Examples
```bash
# Train a 7B model with default settings
/runpod-train --model meta-llama/Llama-2-7b-hf --dataset yahma/alpaca-cleaned

# Train with specific GPU and monitoring
/runpod-train \
    --model Qwen/Qwen3-Reranker-4B \
    --dataset yahma/alpaca-cleaned \
    --gpu "RTX A6000" \
    --epochs 3 \
    --monitor \
    --tensorboard

# Resume from checkpoint
/runpod-train --resume POD_ID --checkpoint step_1000
```

## Options
- `--model`: HuggingFace model ID or path
- `--dataset`: HuggingFace dataset ID  
- `--gpu`: GPU type (default: auto-select based on model size)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: auto)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--lora-r`: LoRA rank (default: 16)
- `--monitor`: Enable real-time monitoring
- `--tensorboard`: Launch TensorBoard server
- `--checkpoint-every`: Save checkpoint every N steps
- `--resume`: Resume from existing pod
- `--output-dir`: S3 or local output directory

## GPU Selection Logic
```
Model Size -> Recommended GPU
< 3B      -> T4 (budget)
3-7B      -> RTX 4090 / A10
7-13B     -> RTX A6000 / A40  
13-30B    -> A100 40GB
30B+      -> A100 80GB / H100
```

## Integration with Unsloth
This command automatically:
1. Uploads training data to RunPod volume
2. Configures Unsloth with optimal settings
3. Monitors training progress
4. Downloads completed adapter/model
5. Uploads to HuggingFace if configured

## Environment Variables
- `RUNPOD_API_KEY`: RunPod API authentication
- `HF_TOKEN`: HuggingFace token for model upload
- `RUNPOD_VOLUME_ID`: Persistent volume for datasets