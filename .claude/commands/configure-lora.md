# Configure LoRA Adapter

Configure Low-Rank Adaptation (LoRA) parameters for efficient fine-tuning.

## Usage



## Arguments

- : LoRA rank (r value, default: 16)
- : LoRA alpha scaling (default: 32)
- : Dropout probability (default: 0.05)
- : Target modules to apply LoRA (default: q_proj,v_proj)

## Examples



## Configuration Options

- **rank**: Higher values = more parameters but better quality
- **alpha**: Scaling factor for LoRA weights
- **dropout**: Regularization to prevent overfitting
- **target_modules**: Which transformer modules to adapt

---
*Unsloth Fine-tuning Module*
