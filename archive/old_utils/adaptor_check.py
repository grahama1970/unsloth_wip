import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Paths
adapter_path = "/home/grahama/dev/vllm_lora/training_output/Phi-3.5-mini-instruct_touch-rugby-rules_adapter/final_model"

# Load adapter configuration
print("Loading adapter configuration...")
config = PeftConfig.from_pretrained(adapter_path)
print(f"Adapter configuration loaded: {config}")

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto"
)
print("Base model loaded.")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Ensure tokenizer special tokens are correctly set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded and configured.")

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("Adapter successfully integrated into base model.")

# Test inference
def generate_response(input_text, model, tokenizer, max_length=512):
    """Generate response using the adapter model."""
    print("Generating response...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Format the prompt with the Phi-3 chat template
    formatted_prompt = f"<|system|>\nYou are a helpful Touch Rugby expert.\n<|end|>\n<|user|>\n{input_text}\n<|end|>\n<|assistant|>\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant's reply
    response_start = response.find("<|assistant|>") + len("<|assistant|>")
    response_end = response.find("<|end|>", response_start)
    return response[response_start:response_end].strip()

# Test questions
questions = [
    "What is a touchdown in Touch Rugby?",
    # "How many players are on a Touch Rugby team?",
    # "What happens after a touchdown in Touch Rugby?"
]

print("=== Testing Model Inference ===")
for question in questions:
    print(f"Q: {question}")
    try:
        response = generate_response(question, model, tokenizer)
        print(f"A: {response}")
    except Exception as e:
        print(f"Error generating response: {e}")
    print("-" * 80)

# Free up memory
torch.cuda.empty_cache()
