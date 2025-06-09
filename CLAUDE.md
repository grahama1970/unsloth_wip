# UNSLOTH CONTEXT — CLAUDE.md

> **Inherits standards from global and workspace CLAUDE.md files with overrides below.**

## Project Context
**Purpose:** LLM fine-tuning with student-teacher thinking enhancement  
**Type:** Processing Spoke  
**Status:** Work in Progress  
**Pipeline Position:** Final step in SPARTA → Marker → ArangoDB → Unsloth

## Project-Specific Overrides

### Special Dependencies
```toml
# Unsloth requires ML training libraries
unsloth = "^2024.0.0"
torch = "^2.0.0"
transformers = "^4.35.0"
datasets = "^2.14.0"
peft = "^0.6.0"
bitsandbytes = "^0.41.0"
runpod = "^1.0.0"
```

### Environment Variables
```bash
# .env additions for Unsloth
RUNPOD_API_KEY=your_runpod_api_key
HUGGINGFACE_TOKEN=your_hf_token
TRAINING_DATA_PATH=/home/graham/workspace/data/training
MODEL_OUTPUT_PATH=/home/graham/workspace/data/models
ENABLE_4BIT_QUANTIZATION=true
LORA_RANK=16
LEARNING_RATE=2e-4
```

### Special Considerations
- **GPU Requirements:** CUDA-capable GPU for training (or RunPod cloud)
- **Memory Management:** 4-bit quantization for memory efficiency
- **Cloud Integration:** RunPod for scalable training infrastructure
- **Model Deployment:** Automatic HuggingFace deployment pipeline

---

## License

MIT License — see [LICENSE](LICENSE) for details.