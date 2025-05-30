---
base_model: {base_model}
tags:
- lora
- adapter
- llama
- touch-rugby
- unsloth
language:
- en
license: llama2
---

# LoRA Adapter for {base_model} - Touch Rugby Rules

This is a LoRA adapter trained on the {base_model} model to specialize in touch rugby rules and gameplay. The adapter was trained using Unsloth's optimization techniques for efficient fine-tuning.

## Model Details

- **Base Model:** {base_model}
- **Training Type:** LoRA (Low-Rank Adaptation)
- **Specialization:** Touch Rugby Rules and Gameplay
- **Repository:** {hub_model_id}
- **Training Framework:** Unsloth

## Training Details

This adapter was trained using:

### Hardware
- GPU: NVIDIA A100
- Precision: 4-bit quantization (QLoRA)

### Training Parameters
- **LoRA Configuration:**
  - Rank (r): 8
  - Alpha: 16
  - Target Modules: 
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

### Dataset
- Custom dataset focused on touch rugby rules and gameplay
- Format: Instruction-following conversations
- Size: ~1000 examples

## Usage

To use this adapter with the base model: