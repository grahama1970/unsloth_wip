# /runpod-infer

Launches RunPod inference server for models that exceed local GPU capacity.

## Usage
```bash
/runpod-infer --model MODEL_NAME --gpu GPU_TYPE [OPTIONS]
```

## Examples
```bash
# Launch inference server for a fine-tuned model
/runpod-infer --model grahamco/qwen3-reranker-entropy-enhanced --gpu "RTX 4090"

# Launch with specific configuration
/runpod-infer \
    --model meta-llama/Llama-2-70b-chat-hf \
    --gpu "A100 80GB" \
    --replicas 2 \
    --max-batch-size 32 \
    --port 8080

# Launch with vLLM optimization
/runpod-infer \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --gpu "A100 40GB" \
    --engine vllm \
    --tensor-parallel 2
```

## Options
- `--model`: HuggingFace model ID or S3 path
- `--gpu`: GPU type (auto-selects based on model)
- `--engine`: Inference engine (transformers/vllm/tgi)
- `--replicas`: Number of replicas for scaling
- `--max-batch-size`: Maximum batch size
- `--port`: Server port (default: 8000)
- `--tensor-parallel`: Tensor parallelism degree
- `--quantization`: Enable quantization (4bit/8bit)
- `--timeout`: Request timeout in seconds

## Response Format
```json
{
    "pod_id": "abc123",
    "endpoint": "https://abc123-8000.proxy.runpod.net",
    "status": "running",
    "gpu": "RTX A6000",
    "model": "grahamco/qwen3-reranker-entropy-enhanced",
    "metrics": {
        "requests_per_second": 45.2,
        "avg_latency_ms": 124,
        "gpu_utilization": 78
    }
}
```

## Cost Optimization
- Pods auto-stop after 10 minutes of inactivity
- Use `--persistent` flag to keep running
- Shared endpoints for multiple projects
- Automatic GPU downgrade if primary unavailable

## Integration Points
- Direct integration with `llm_call` project
- Supports OpenAI-compatible API
- Works with LangChain/LlamaIndex
- MCP server endpoint support