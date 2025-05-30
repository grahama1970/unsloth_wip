"""Inference module for generating text with fine-tuned models."""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel
from loguru import logger
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_new_tokens: int = Field(default=512, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    use_cache: bool = Field(default=True, description="Whether to use KV cache")
    stream: bool = Field(default=False, description="Whether to stream output")


class InferenceEngine:
    """Engine for running inference with fine-tuned models."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_seq_length: int = 2048
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model or adapter
            device: Device to run inference on
            load_in_4bit: Whether to load in 4-bit precision
            max_seq_length: Maximum sequence length
        """
        self.model_path = Path(model_path)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Check if this is an adapter or full model
        adapter_config = self.model_path / "adapter_config.json"
        is_adapter = adapter_config.exists()
        
        if is_adapter:
            # Load base model and adapter
            import json
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path", "unsloth/Phi-3.5-mini-instruct")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )
            
            # Load adapter
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                adapter_name=str(self.model_path),
            )
        else:
            # Load full model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )
            
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        logger.info("Model loaded successfully")
        
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()
            
        config = config or GenerationConfig()
        
        # Format prompt with chat template
        messages = self._format_messages(prompt, system_prompt)
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)
        
        # Setup streamer if needed
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if config.stream else None
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                use_cache=config.use_cache,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode output
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
        
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            system_prompt: Optional system prompt
            
        Returns:
            List of generated texts
        """
        if self.model is None:
            self.load_model()
            
        config = config or GenerationConfig()
        
        # Format all prompts
        all_formatted = []
        for prompt in prompts:
            messages = self._format_messages(prompt, system_prompt)
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            all_formatted.append(formatted)
            
        # Tokenize batch
        inputs = self.tokenizer(
            all_formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                use_cache=config.use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            generated_ids = output[inputs.input_ids[i].shape[-1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(generated_text)
            
        return results
        
    def _format_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Format messages for chat template."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        return messages
        
    def chat(
        self,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Interactive chat mode.
        
        Args:
            config: Generation configuration
            system_prompt: Optional system prompt
        """
        if self.model is None:
            self.load_model()
            
        config = config or GenerationConfig(stream=True)
        
        print("Chat mode started. Type 'exit' to quit.")
        if system_prompt:
            print(f"System: {system_prompt}")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                    
                print("\nAssistant: ", end="", flush=True)
                response = self.generate(user_input, config, system_prompt)
                
                if not config.stream:
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                print(f"\nError: {e}")