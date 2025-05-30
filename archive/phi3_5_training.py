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

# from app.backend.unsloth_wip.utils.tensorboard_utils import cleanup_tensorboard

@dataclass
class TrainingConfig:
    """Configuration for Phi-3.5 training pipeline."""
    model_name: str = "unsloth/Phi-3.5-mini-instruct"
    dataset_name: str = "Trelis/touch-rugby-rules"
    base_output_dir: str = "./training_output"
    
    # Model config
    max_seq_length: int = 2048  # Supports RoPE scaling internally
    load_in_4bit: bool = True
    dtype = None 
    
    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: int = 0
    bias: str = "none"
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
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

def setup_model_and_tokenizer(config: TrainingConfig) -> Tuple[Any, Any]:
    """Load and configure the model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit
    )
    
    # Configure tokenizer with Phi-3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        mapping={
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt"
        }
    )
    
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
    
    logger.info("Creating new LoRA adapter...")
    return FastLanguageModel.get_peft_model(
        model,
        r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None  # And LoftQ
    )

def save_lora_model(model, output_dir):
    """Save the LoRA adapter weights."""
    model.save_pretrained(output_dir, "lora_adapter")  # Save only the adapter
    logger.success(f"LoRA adapter saved to {output_dir}")

def format_dataset(examples: dict, tokenizer: Any) -> dict:
    """Format examples using the chat template."""
    try:
        input_texts, label_texts = [], []
        
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            prompt = str(prompt) if prompt else ""
            completion = str(completion) if completion else ""
            
            input_conversation = [{"from": "human", "value": prompt}]
            input_text = tokenizer.apply_chat_template(
                input_conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            input_texts.append(input_text)
            
            full_conversation = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": completion}
            ]
            label_text = tokenizer.apply_chat_template(
                full_conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            label_texts.append(label_text)
        
        # Tokenize inputs with padding and truncation
        model_inputs = tokenizer(
            input_texts,
            padding="max_length",  # Changed to max_length
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Tokenize labels with padding and truncation
        labels = tokenizer(
            label_texts,
            padding="max_length",  # Changed to max_length
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids
        
        # Convert to lists for dataset compatibility
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
        train_columns = set(train_dataset.column_names)
        val_columns = set(val_dataset.column_names)
        
        if not required_columns.issubset(train_columns):
            raise ValueError(f"Training dataset missing required columns: {required_columns - train_columns}")
        if not required_columns.issubset(val_columns):
            raise ValueError(f"Validation dataset missing required columns: {required_columns - val_columns}")
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        def safe_format(dataset_to_format, desc=""):
            try:
                return dataset_to_format.map(
                    lambda x: format_dataset(x, tokenizer),
                    batched=True,
                    batch_size=32,
                    remove_columns=dataset_to_format.column_names,
                    desc=f"Formatting {desc} dataset"
                )
            except Exception as e:
                logger.error(f"Error formatting {desc} dataset")
                logger.error(f"First example: {dataset_to_format[0]}")
                raise
        
        train_formatted = safe_format(train_dataset, "training")
        val_formatted = safe_format(val_dataset, "validation")
        
        return train_formatted, val_formatted
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise

def create_training_args(config: TrainingConfig) -> TrainingArguments:
    """Create training arguments optimized for small dataset."""
    return TrainingArguments(
        output_dir=config.output_dir,
        # Small batch size to ensure better gradient updates with limited data
        per_device_train_batch_size=1,
        
        # Accumulate gradients from 2 steps for slightly larger effective batch size
        gradient_accumulation_steps=2,
        
        # With small dataset, we want more epochs rather than steps
        num_train_epochs=5,  # Replace max_steps with num_train_epochs
        
        # Warm up over 10% of total training steps
        warmup_ratio=0.1,
        
        # Slightly lower learning rate for more stable training on small dataset
        learning_rate=1e-4,
        
        # Use mixed precision based on hardware support
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        
        # Log more frequently to monitor training closely
        logging_steps=1,
        
        # Save checkpoints more frequently with small dataset
        save_steps=20,
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        
        # 8-bit Adam optimizer for memory efficiency
        optim="adamw_8bit",
        
        # Slightly higher weight decay to prevent overfitting on small dataset
        weight_decay=0.05,
        
        # Cosine schedule works well for small datasets
        lr_scheduler_type="cosine",
        
        # Enable evaluation during training
        evaluation_strategy="steps",
        eval_steps=20,
        
        # Change this line from "none" to ["tensorboard"]
        report_to=["tensorboard"],
        
        # Add these lines for more detailed logging
        logging_dir=f"{config.output_dir}/logs",
        logging_strategy="steps",
        logging_first_step=True,
        
        seed=3407
    )

def log_memory_stats():
    """Log current memory usage stats."""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Current GPU memory: {current_memory:.2f} GB")
        logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")

def test_model_inference(model: Any, tokenizer: Any, questions: Optional[List[str]] = None) -> None:
    """Test model with sample questions for debugging."""
    try:
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        model.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            FastLanguageModel.for_inference(model)
        clear_memory()
        
        if questions is None:
            questions = [
                "What is a touchdown in Touch Rugby?",
                # "How many players are on a Touch Rugby team?",
                #"What happens after a touchdown in Touch Rugby?"
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
) -> Any:
    """Initialize trainer and start training with validation."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add this line to pass the validation dataset
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args
    )
    
    
    trainer.train()
    return trainer

class LogMemoryCallback(TrainerCallback):
    """Callback to log memory usage during training."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            log_memory_stats()


def get_model_device(model):
    """Get the device where the model is located."""
    return model.device if hasattr(model, 'device') else next(model.parameters()).device

def save_model(model: Any, config: TrainingConfig) -> None:
    """Save model according to configuration options."""
    try:
        if config.save_adapter:
            adapter_path = f"{config.output_dir}/final_model"
            logger.info(f"Saving LoRA adapter to {adapter_path}")
            model.save_pretrained(adapter_path, "lora_adapter")
            logger.success("LoRA adapter saved successfully")

        if config.save_merged:
            merged_path = config.merged_model_path
            logger.info(f"Saving merged model to {merged_path}")
            # Merge adapter weights with base model
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
                test_model_inference(model, tokenizer)
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
            trainer = train_model(model, train_dataset, val_dataset, training_args, tokenizer, config)
            
            logger.info("Step 8: Saving model...")
            save_model(model, config)
            
            if config.save_adapter:
                logger.success(f"LoRA adapter saved to {config.output_dir}/final_model")
            if config.save_merged:
                logger.success(f"Merged model saved to {config.merged_model_path}")
        else:
            logger.info("Using existing model, skipping training steps")
            
        logger.info("Testing final model inference:")
        test_model_inference(model, tokenizer)
        
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

