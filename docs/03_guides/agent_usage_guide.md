# Agent Usage Guide for Unsloth CLI

This guide shows how agents (like Claude) can use the Unsloth CLI to train models, including RunPod integration.

## 🚀 Available Methods

### 1. Direct CLI Commands

```bash
# Train a model (auto-detects local vs RunPod)
unsloth train --model unsloth/Phi-3.5-mini-instruct \
              --dataset qa_data.jsonl \
              --hub-id username/my-model

# Force RunPod for any model
unsloth train --model unsloth/Phi-3.5-mini-instruct \
              --dataset qa_data.jsonl \
              --force-runpod

# RunPod-specific commands
unsloth runpod list         # List all pods
unsloth runpod gpus         # Show available GPUs
unsloth runpod stop <id>    # Stop a pod
unsloth runpod train ...    # Direct RunPod training
```

### 2. Slash Commands (in Claude Code)

After generating slash commands:
```bash
# Generate slash commands
unsloth generate-slash --output .claude/commands
```

Available slash commands:
- `/unsloth-train` - Complete training pipeline
- `/unsloth-enhance` - Enhance dataset with student-teacher
- `/unsloth-validate` - Validate trained adapter
- `/unsloth-runpod-list` - List RunPod pods
- `/unsloth-runpod-gpus` - Show available GPUs
- `/unsloth-runpod-stop` - Stop a RunPod pod
- `/unsloth-runpod-train` - Train on RunPod

### 3. MCP Tools

Start the MCP server:
```bash
# After installation, use the entry point:
unsloth-mcp --port 5555

# Or if running from source:
python -m unsloth.cli.mcp_server --port 5555
```

Available MCP tools:
- `unsloth_train()` - Complete training pipeline
- `unsloth_enhance()` - Dataset enhancement
- `unsloth_validate()` - Model validation
- `unsloth_runpod_list()` - List pods
- `unsloth_runpod_gpus()` - Show GPUs
- `unsloth_runpod_stop()` - Stop pod
- `unsloth_runpod_train()` - RunPod training

## 📋 Complete Workflow Examples

### Example 1: Train Small Model (Local)

```bash
# 1. Enhance dataset (optional)
unsloth enhance --input raw_qa.jsonl \
                --output enhanced_qa.jsonl \
                --model unsloth/Phi-3.5-mini-instruct

# 2. Train model (will run locally)
unsloth train --model unsloth/Phi-3.5-mini-instruct \
              --dataset enhanced_qa.jsonl \
              --hub-id myusername/phi-3.5-enhanced

# 3. Validate results
unsloth validate --adapter ./outputs/pipeline/adapter \
                 --base-model unsloth/Phi-3.5-mini-instruct
```

### Example 2: Train Large Model (Auto RunPod)

```bash
# 1. Check available GPUs
unsloth runpod gpus

# 2. Train 70B model (automatically uses RunPod)
unsloth train --model meta-llama/Llama-2-70b-hf \
              --dataset enhanced_qa.jsonl \
              --hub-id myusername/llama-70b-lora

# 3. Monitor pods
unsloth runpod list

# 4. Results are automatically uploaded to HuggingFace
```

### Example 3: Manual RunPod Control

```bash
# 1. Start RunPod training
unsloth runpod train --model unsloth/Llama-3.2-13B \
                     --dataset qa_data.jsonl \
                     --hub-id myusername/llama-13b

# 2. Check pod status
unsloth runpod list

# 3. Stop pod if needed
unsloth runpod stop <pod-id>
```

## 🔧 RunPod Integration Details

### Automatic GPU Selection

The system automatically selects appropriate GPUs:

| Model Size | Preferred GPUs | Fallback GPUs |
|------------|---------------|---------------|
| ≤7B | RTX 4090, RTX A6000 | A100 PCIe |
| 13B | A100 PCIe, A100 SXM | RTX A6000 |
| 30B | A100 SXM 80GB | H100 PCIe |
| 70B | H100 PCIe, H100 SXM | H100 NVL |

### Pod Lifecycle

1. **Creation**: Pods are created with PyTorch Docker image
2. **Training**: Script is auto-generated and executed
3. **Monitoring**: Progress tracked via status files
4. **Cleanup**: Pods auto-terminate after training

### Cost Optimization

- Pods are reused if already running
- Automatic termination after training
- GPU selection prioritizes cost-effective options

## 🌐 Environment Setup

Required environment variables:
```bash
export HF_TOKEN="hf_..."              # For model upload
export ANTHROPIC_API_KEY="sk-ant..." # For student-teacher
export OPENAI_API_KEY="sk-..."       # For validation
export RUNPOD_API_KEY="..."          # For RunPod training
```

## 💡 Tips for Agents

1. **Model Size Detection**: The CLI automatically detects when RunPod is needed
2. **Force RunPod**: Use `--force-runpod` to test RunPod with smaller models
3. **Progress Monitoring**: Use `runpod list` to check training status
4. **Error Recovery**: Pods are automatically cleaned up on failure

## 🛠️ Troubleshooting

### RunPod Issues

```bash
# Check if pods are running
unsloth runpod list

# Stop stuck pods
unsloth runpod stop <pod-id> --terminate

# Verify GPU availability
unsloth runpod gpus
```

### Training Issues

```bash
# Check logs in output directory
cat outputs/pipeline/pipeline_results.json

# Validate adapter after training
unsloth validate --adapter outputs/pipeline/adapter \
                 --base-model <model-name>
```

## 📊 Example Output

Successful training output:
```
🚀 Starting Unsloth Training Pipeline
Model: unsloth/Llama-3.2-70B
Dataset: enhanced_qa.jsonl

☁️ Using RunPod (model size requires H100)
Pod started: pod-abc123
Training in progress...

✅ Pipeline completed successfully!

📊 Enhancement Statistics:
  Examples: 1000
  Avg Iterations: 2.3
  Convergence: 87.0%

🚀 Training Results:
  Location: runpod
  Adapter: outputs/pipeline/adapter

🌐 Model uploaded to: https://huggingface.co/username/model
```

This comprehensive integration ensures agents can easily train models of any size using the appropriate infrastructure!