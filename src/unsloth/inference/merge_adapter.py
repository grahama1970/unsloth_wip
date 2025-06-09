"""Merge LoRA adapter with base model for deployment."""
Module: merge_adapter.py
Description: Functions for merge adapter operations

from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: str | None = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16
) -> str:
    """
    Merge LoRA adapter with base model.
    
    Args:
        base_model_path: HuggingFace model ID or path to base model
        adapter_path: Path to LoRA adapter
        output_path: Where to save merged model
        push_to_hub: Optional HuggingFace Hub model ID
        device_map: Device mapping for model
        torch_dtype: Data type for model weights
        
    Returns:
        Path to merged model
    """
    logger.info("Merging LoRA adapter with base model...")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter: {adapter_path}")

    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge adapter with base model
    logger.info("Merging adapter weights...")
    model = model.merge_and_unload()

    # Save merged model
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, private=True)
        tokenizer.push_to_hub(push_to_hub, private=True)

    logger.info("✅ Merge complete!")
    return str(output_path)


def merge_adapter_for_unsloth(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    max_seq_length: int = 2048
) -> str:
    """
    Merge adapter using Unsloth's optimized method.
    
    Args:
        base_model_path: Base model name
        adapter_path: Path to adapter
        output_path: Output path
        max_seq_length: Maximum sequence length
        
    Returns:
        Path to merged model
    """
    from unsloth import FastLanguageModel

    logger.info("Using Unsloth optimized merge...")

    # Load model and adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,  # Load in full precision for merging
    )

    # Get PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        adapter_name=str(adapter_path),
    )

    # Merge and save
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")

    logger.info(f"✅ Merged model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--push-to-hub", help="Push to HuggingFace Hub with this model ID")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth merge method")

    args = parser.parse_args()

    if args.use_unsloth:
        merge_adapter_for_unsloth(
            args.base_model,
            args.adapter,
            args.output
        )
    else:
        merge_lora_adapter(
            args.base_model,
            args.adapter,
            args.output,
            args.push_to_hub
        )
