import os
import torch
import gc
import subprocess
from pathlib import Path
import webbrowser
import time
from dotenv import load_dotenv
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from dataclasses import dataclass, field
from loguru import logger
from typing import Any, Tuple, Optional, List
from transformers import TrainerCallback
import sys

@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # Changed to a supported Llama model
    dataset_name: str = "Trelis/touch-rugby-rules"
    base_output_dir: str = "./training_output"
    
    # Model config
    max_seq_length: int = 2048
    load_in_4bit: bool = True  # Using 4-bit quantization for memory efficiency
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+# Changed to float16
    
    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: int = 0
    bias: str = "none"
    use_rslora = False
    loftq_config = None
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training config
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    
    # TensorBoard config
    tensorboard_port: int = 6006
    auto_launch_browser: bool = True
    
    # Model handling
    force_retrain: bool = True  # Whether to force retraining even if model exists
    model_exists_ok: bool = True  # Whether to use existing model if found
    
    # Model saving options
    save_adapter: bool = True  # Whether to save the LoRA adapter separately
    save_merged: bool = False  # Whether to save the merged model
    merged_output_dir: str = field(default="", init=True)  # Custom path for merged model
    
    # Training Arguments
    warmup_ratio: float = 0.1
    learning_rate: float = 1e-4
    logging_steps: int = 1
    save_steps: int = 20
    save_total_limit: int = 3
    weight_decay: float = 0.05
    lr_scheduler_type: str = "cosine"
    eval_strategy: str = "steps"
    eval_steps: int = 20
    logging_strategy: str = "steps"
    logging_first_step: bool = True
    optim: str = "adamw_8bit"
    
    # HuggingFace upload options
    push_to_hub: bool = False
    hub_model_id: str = field(default="grahamaco/")  # e.g. "username/model-name"
    hub_private: bool = True
    hub_token: str = field(default="")  # Will use env var HF_TOKEN if not set
    
    @property
    def adapter_name(self) -> str:
        """Generate descriptive adapter name from model and dataset."""
        model_short = self.model_name.split('/')[-1]
        dataset_short = self.dataset_name.split('/')[-1]
        return f"{model_short}_{dataset_short}_adapter"
    
    @property
    def output_dir(self) -> str:
        """Generate output directory path including adapter name."""
        return f"{self.base_output_dir}/{self.adapter_name}"
    
    @property
    def merged_model_path(self) -> str:
        """Get path for merged model, defaulting to output_dir/adapter-merged if not specified."""
        if self.merged_output_dir:
            return self.merged_output_dir
        return f"{self.output_dir}/adapter-merged"

def clear_memory():
    """Clear CUDA cache and garbage collect."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def setup_environment(config: TrainingConfig) -> torch.device:
    """Set up the training environment."""
    clear_memory()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def format_to_openai_messages(prompt: str, completion: str = None) -> list:
    """Convert prompt/completion to OpenAI message format."""
    messages = [{"role": "user", "content": prompt}]
    if completion:
        messages.append({"role": "assistant", "content": completion})
    return messages

def setup_model_and_tokenizer(config: TrainingConfig) -> Tuple[Any, Any]:
    """Load and configure the model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit
    )
    
    # Apply Unsloth's chat template correctly
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    )
    
    logger.info("Model and tokenizer successfully configured.")
    return model, tokenizer

def check_model_exists(config: TrainingConfig) -> bool:
    """Check if a trained model already exists."""
    model_path = Path(config.output_dir) / "final_model"
    adapter_path = model_path / "adapter_config.json"
    
    if adapter_path.exists():
        logger.info(f"Found existing model at {model_path}")
        return True
    return False

def configure_lora_adapter(model, config):
    """Load existing LoRA adapter or create new one if not found."""
    model_path = Path(config.output_dir) / "final_model"
    
    if not config.force_retrain and check_model_exists(config):
        if config.model_exists_ok:
            logger.info("Loading existing LoRA adapter...")
            try:
                model.load_adapter(str(model_path))
                logger.success("Existing adapter loaded successfully")
                return model
            except Exception as e:
                logger.warning(f"Failed to load existing adapter: {e}")
                if not config.force_retrain:
                    raise
        else:
            raise ValueError(
                f"Model already exists at {model_path}. "
                "Set force_retrain=True to retrain or model_exists_ok=True to use existing model."
            )
    
    logger.info("Creating new LoRA adapter with Unsloth's recommended settings...")
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,                    # Unsloth recommended: 8
        target_modules=config.target_modules,# Unsloth recommended: all attention & mlp
        lora_alpha=config.lora_alpha,       # Unsloth recommended: 16
        lora_dropout=config.lora_dropout,   # Unsloth recommended: 0
        bias=config.bias,                   # Unsloth recommended: "none"
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=config.use_rslora,       # Unsloth recommended: False
        loftq_config=config.loftq_config    # Unsloth recommended: None
    )

def format_dataset(examples: dict, tokenizer: Any, config: TrainingConfig) -> dict:
    """Format examples using the chat template."""
    try:
        input_texts, label_texts = [], []

        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            prompt = str(prompt) if prompt else ""
            completion = str(completion) if completion else ""

            # Use format_to_openai_messages to create conversations
            input_messages = format_to_openai_messages(prompt)
            full_messages = format_to_openai_messages(prompt, completion)

            # Apply chat templates - explicitly using llama-3 template
            # logger.debug(f"Input messages: {input_messages}")
            input_text = tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True,
                # chat_template="llama-3" Not needed as default is llama-3
            )
            #logger.debug("Template output:", input_text)  # Add this line here
            input_texts.append(input_text)

            label_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
                # chat_template="llama-3" Not needed as default is llama-3
            )
            label_texts.append(label_text)
        

        # Tokenize inputs with padding and truncation
        model_inputs = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt"
        )

        # Tokenize labels with padding and truncation
        labels = tokenizer(
            label_texts,
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt"
        ).input_ids

     
        # Convert tensors to lists for dataset compatibility
        return {
            "input_ids": model_inputs.input_ids.tolist(),
            "attention_mask": model_inputs.attention_mask.tolist(),
            "labels": labels.tolist()
        }

    except Exception as e:
        logger.error(f"Error formatting batch: {e}")
        logger.error(f"First example - Prompt: {examples['prompt'][0]}")
        logger.error(f"First example - Completion: {examples['completion'][0]}")
        raise

def prepare_dataset(config: TrainingConfig, tokenizer: Any) -> Tuple[Any, Any]:
    """Load and format the dataset."""
    try:
        dataset_info = load_dataset(config.dataset_name, split=None)
        available_splits = dataset_info.keys()
        logger.info(f"Available dataset splits: {available_splits}")

        if "train" not in available_splits:
            raise ValueError(f"Dataset {config.dataset_name} must have a 'train' split")
        val_split = "validation" if "validation" in available_splits else "test"
        if val_split not in available_splits:
            raise ValueError(f"Dataset {config.dataset_name} must have either 'validation' or 'test' split")

        train_dataset = load_dataset(config.dataset_name, split="train")
        val_dataset = load_dataset(config.dataset_name, split=val_split)

        required_columns = {"prompt", "completion"}
        if not required_columns.issubset(train_dataset.column_names):
            raise ValueError(f"Training dataset missing required columns: {required_columns - set(train_dataset.column_names)}")
        if not required_columns.issubset(val_dataset.column_names):
            raise ValueError(f"Validation dataset missing required columns: {required_columns - set(val_dataset.column_names)}")

        train_formatted = train_dataset.map(
            lambda x: format_dataset(x, tokenizer, config),
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Formatting training dataset"
        )

        val_formatted = val_dataset.map(
            lambda x: format_dataset(x, tokenizer, config),
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Formatting validation dataset"
        )

        return train_formatted, val_formatted

    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise

def create_training_args(config: TrainingConfig) -> TrainingArguments:
    """Create training arguments from config."""
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        learning_rate=config.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        report_to="none",  # or ["tensorboard"] if needed
        logging_dir=f"{config.output_dir}/logs",
        logging_strategy=config.logging_strategy,
        logging_first_step=config.logging_first_step,
        seed=3407
    )

def log_memory_stats():
    """Log current memory usage stats."""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Current GPU memory: {current_memory:.2f} GB")
        logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")

def test_model_inference(model: Any, tokenizer: Any, device: torch.device, questions: Optional[List[str]] = None) -> None:
    """Test model with sample questions for debugging."""
    try:
        model.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            FastLanguageModel.for_inference(model)
        clear_memory()
        
        if questions is None:
            questions = [
                "What is a touchdown in Touch Rugby?",
            ]
        
        logger.info("=== Testing Model Inference ===")
        for question in questions:
            try:
                logger.info(f"\nQ: {question}")
                
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                logger.debug(f"Formatted Prompt:\n{formatted_prompt}")
                
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,
                    repetition_penalty=1.2,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"A: {response}\n{'-' * 80}")
                
            except Exception as e:
                logger.error(f"Error generating response for question '{question}': {e}")
                
    except Exception as e:
        logger.error(f"Error in test_model_inference: {e}")

def train_model(
        model: Any, 
        train_dataset: Any, 
        val_dataset: Any, 
        training_args: TrainingArguments, 
        tokenizer: Any, 
        config: TrainingConfig
) -> Tuple[Any, Any, float]:
    """Initialize trainer and start training with validation."""
    
    # Capture initial GPU memory
    start_gpu_memory = 0
    if torch.cuda.is_available():
        start_gpu_memory = round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 3)  # GB

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args
    )
    
    trainer_stats = trainer.train()
    return trainer, trainer_stats, start_gpu_memory


def save_model(model: Any, config: TrainingConfig) -> None:
    """Save model according to configuration options."""
    try:
        if config.save_adapter:
            adapter_path = f"{config.output_dir}/final_model"
            logger.info(f"Saving LoRA adapter to {adapter_path}")
            model.save_pretrained(adapter_path, "lora_adapter")
            logger.success("LoRA adapter saved successfully")
            
            # Create model card from template
            template_path = Path('templates/model_card_template.md')
            if template_path.exists():
                try:
                    with open(template_path, 'r') as f:
                        template = f.read()
                    
                    # Format template with model details
                    model_card = template.format(
                        base_model=config.model_name,
                        hub_model_id=config.hub_model_id if hasattr(config, 'hub_model_id') else "Not specified",
                        adapter_path=adapter_path
                    )
                    
                    # Save model card in adapter directory
                    readme_path = Path(adapter_path) / "README.md"
                    with open(readme_path, "w") as f:
                        f.write(model_card)
                    logger.info("Created README.md from template")
                except Exception as e:
                    logger.warning(f"Failed to create model card from template: {e}")

        if config.save_merged:
            merged_path = config.merged_model_path
            logger.info(f"Saving merged model to {merged_path}")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_path)
            logger.success("Merged model saved successfully")
            
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def setup_tensorboard(config: TrainingConfig) -> Optional[subprocess.Popen]:
    """Setup TensorBoard server and launch browser."""
    try:
        # Start TensorBoard server
        tensorboard_command = [
            "tensorboard",
            "--logdir", config.output_dir,
            "--port", str(config.tensorboard_port),
            "--bind_all"  # Allow remote connections
        ]
        
        process = subprocess.Popen(
            tensorboard_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give TensorBoard time to start
        time.sleep(3)
        
        # Open browser if configured
        if config.auto_launch_browser:
            webbrowser.open(f"http://localhost:{config.tensorboard_port}")
        
        return process
        
    except Exception as e:
        logger.error(f"Failed to start TensorBoard: {e}")
        return None

def cleanup_tensorboard(process: Optional[subprocess.Popen]) -> None:
    """Cleanup TensorBoard process."""
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception as e:
            logger.error(f"Error cleaning up TensorBoard: {e}")

def display_training_metrics(trainer_stats: Any, start_gpu_memory: float) -> None:
    """Display training metrics including time and memory usage."""
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping memory metrics")
            return

        # Get memory stats
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # Convert to GB
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        # Get training time stats
        runtime_seconds = trainer_stats.metrics.get('train_runtime', 0)
        runtime_minutes = round(runtime_seconds / 60, 2)

        # Display metrics
        logger.info("\n=== Training Metrics ===")
        logger.info(f"Training Time: {runtime_seconds:.2f} seconds ({runtime_minutes} minutes)")
        logger.info(f"Peak Reserved Memory: {used_memory} GB")
        logger.info(f"Peak Training Memory: {used_memory_for_lora} GB")
        logger.info(f"Memory Usage: {used_percentage}% of total GPU memory")
        logger.info(f"Training Memory Usage: {lora_percentage}% of total GPU memory")

        # Display additional training metrics if available
        if 'train_loss' in trainer_stats.metrics:
            logger.info(f"Final Training Loss: {trainer_stats.metrics['train_loss']:.4f}")
        if 'eval_loss' in trainer_stats.metrics:
            logger.info(f"Final Evaluation Loss: {trainer_stats.metrics['eval_loss']:.4f}")

    except Exception as e:
        logger.error(f"Error displaying training metrics: {e}")

def main():
    """Main training pipeline."""
    config = TrainingConfig()
    tensorboard_process = None
    
    try:
        model_exists = check_model_exists(config)
        if model_exists:
            if config.force_retrain:
                logger.warning("Existing model found but force_retrain=True, will retrain model")
            elif not config.model_exists_ok:
                raise ValueError("Model already exists. Use force_retrain=True to retrain")
            else:
                logger.info("Using existing model, loading directly for inference...")
                device = setup_environment(config)
                model, tokenizer = setup_model_and_tokenizer(config)
                model = configure_lora_adapter(model, config)
                logger.info("Testing existing model inference:")
                test_model_inference(model, tokenizer, device)
                return
        
        logger.info("Step 1: Setting up environment...")
        device = setup_environment(config)
        
        # logger.info("Step 2: Setting up TensorBoard...")
        # tensorboard_process = setup_tensorboard(config)
        
        logger.info("Step 3: Setting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(config)
        
        logger.info("Step 4: Setting up LoRA...")
        model = configure_lora_adapter(model, config)
        
        if config.force_retrain or not check_model_exists(config):
            logger.info("Step 5: Preparing datasets...")
            train_dataset, val_dataset = prepare_dataset(config, tokenizer)
            
            logger.info("Step 6: Creating training arguments...")
            training_args = create_training_args(config)
            
            logger.info("Step 7: Training model...")
            trainer, trainer_stats, start_gpu_memory = train_model(
                model, train_dataset, val_dataset, training_args, tokenizer, config
            )
            
            # Display training metrics
            display_training_metrics(trainer_stats, start_gpu_memory)
            
            logger.info("Step 8: Saving model...")
            save_model(model, config)
            
            if config.save_adapter:
                logger.success(f"LoRA adapter saved to {config.output_dir}/final_model")
            if config.save_merged:
                logger.success(f"Merged model saved to {config.merged_model_path}")
        else:
            logger.info("Using existing model, skipping training steps")
            
        logger.info("Testing final model inference:")
        test_model_inference(model, tokenizer, device)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        if tensorboard_process:
            logger.info("Cleaning up TensorBoard...")
            # cleanup_tensorboard(tensorboard_process)
        
        logger.info("Clearing memory...")
        clear_memory()

if __name__ == "__main__":
    main()

