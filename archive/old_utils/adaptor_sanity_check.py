from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
adapter_path = "/home/grahama/dev/vllm_lora/training_output/Phi-3.5-mini-instruct_touch-rugby-rules_adapter/final_model"
save_path = "./merged_model_debug"

# Load adapter configuration
config = PeftConfig.from_pretrained(adapter_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge adapter into base model
model = model.merge_and_unload()

# Save merged model
print(f"Saving merged model to: {save_path}")
model.save_pretrained(save_path)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.save_pretrained(save_path)

print("Merged model and tokenizer saved successfully.")
