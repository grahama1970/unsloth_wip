"""Enhanced configuration classes for Unsloth training pipeline with advanced features."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
import torch

from .grokking_config import GrokkingConfig


@dataclass
class EnhancedTrainingConfig:
    """Enhanced configuration for training with Unsloth including all best practices."""
    
    # Model settings
    model_name: str = "unsloth/Phi-3.5-mini-instruct"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True
    
    # Dataset settings
    dataset_source: str = "arangodb"  # "arangodb" or "huggingface"
    dataset_path: str = "/home/graham/workspace/experiments/arangodb/qa_output"
    dataset_split: str = "train"
    validation_split: float = 0.1
    max_samples: Optional[int] = None
    metadata_filters: Dict[str, Any] = field(default_factory=lambda: {})
    
    # LoRA settings - Enhanced
    lora_r: int = 16  # Increased from 8 based on best practices
    lora_alpha: int = 32  # 2x rank as recommended
    lora_dropout: float = 0.05  # Small dropout as recommended
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head"  # Added lm_head for better performance
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict[str, Any]] = None
    
    # Advanced LoRA settings
    use_dora: bool = False  # Weight-Decomposed LoRA
    modules_to_save: Optional[List[str]] = None  # Additional modules to train
    rank_pattern: Optional[Dict[int, int]] = None  # Layer-wise rank
    alpha_pattern: Optional[Dict[int, int]] = None  # Layer-wise alpha
    
    # Training arguments
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    num_train_epochs: int = 3
    learning_rate: float = 2e-4  # Higher LR as recommended
    fp16: bool = not torch.cuda.is_bf16_supported()
    bf16: bool = torch.cuda.is_bf16_supported()
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 3407
    
    # Training optimizations
    max_grad_norm: float = 1.0
    neftune_noise_alpha: Optional[float] = 5  # NEFTune for better convergence
    torch_compile: bool = False
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
    dataloader_num_workers: int = 2
    
    # Output settings
    output_dir: str = "./outputs"
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only 3 checkpoints
    
    # Evaluation settings - Enhanced
    evaluation_strategy: str = "steps"
    eval_steps: int = 100  # More frequent evaluation
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    eval_accumulation_steps: Optional[int] = None
    include_inputs_for_metrics: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Logging and monitoring - Enhanced
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    tensorboard_log_dir: Optional[str] = None
    log_gradient_norm: bool = True
    log_weight_histogram: bool = False
    log_learning_rate: bool = True
    logging_first_step: bool = True
    logging_nan_inf_filter: bool = True
    
    # Memory optimizations
    cpu_offload: bool = False
    zero_stage: int = 0
    activation_checkpointing: bool = False
    optim_args: Optional[str] = None  # For 8bit optimizer args
    
    # Processing settings
    dataset_num_proc: int = 2
    packing: bool = False
    group_by_length: bool = True  # Group similar length sequences
    length_column_name: str = "length"
    
    # Adapter settings
    adapter_name: Optional[str] = None
    force_retrain: bool = False
    
    # Custom functions
    compute_metrics: Optional[Callable] = None
    preprocess_logits_for_metrics: Optional[Callable] = None
    
    # Grokking configuration
    grokking: Optional[GrokkingConfig] = None
    use_grokking: bool = False
    
    def __post_init__(self):
        """Post initialization processing."""
        # Generate adapter name if not provided
        if not self.adapter_name:
            model_short = self.model_name.split("/")[-1].lower()
            dataset_short = Path(self.dataset_path).stem if "/" in self.dataset_path else self.dataset_path
            self.adapter_name = f"{model_short}_{dataset_short}_r{self.lora_r}_a{self.lora_alpha}"
            
        # Set run name if not provided
        if not self.run_name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.adapter_name}_{timestamp}"
            
        # Set tensorboard log dir if not provided
        if not self.tensorboard_log_dir:
            self.tensorboard_log_dir = f"{self.output_dir}/tensorboard/{self.run_name}"
            
        # Validate LoRA settings
        if self.lora_alpha < self.lora_r:
            print(f"⚠️ Warning: alpha ({self.lora_alpha}) < rank ({self.lora_r}). "
                  f"Recommended: alpha = 2 * rank = {2 * self.lora_r}")
                  
        # Calculate approximate trainable parameters
        # This is a rough estimate - actual will depend on model architecture
        approx_params = self.lora_r * len(self.target_modules) * 2 * 4096  # Assuming ~4096 hidden size
        print(f"📊 Approximate trainable parameters: {approx_params:,}")
        
        # Initialize grokking config if enabled
        if self.use_grokking and not self.grokking:
            self.grokking = GrokkingConfig(enable_grokking=True)
            
        # Adjust epochs and settings for grokking
        if self.grokking and self.grokking.enable_grokking:
            original_epochs = self.num_train_epochs
            self.num_train_epochs = self.grokking.calculate_grokking_epochs(original_epochs)
            self.weight_decay = self.grokking.grokking_weight_decay
            
            # Disable early stopping for grokking
            if self.grokking.disable_early_stopping:
                self.early_stopping_patience = -1
                
            print(f"🧠 Grokking enabled: {original_epochs} → {self.num_train_epochs} epochs")
            print(f"   Weight decay: {self.weight_decay}")
            print(f"   Early stopping: {'Disabled' if self.early_stopping_patience < 0 else 'Enabled'}")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation set generation."""
    
    # Evaluation set creation
    eval_set_size: float = 0.05  # 5% of total dataset
    eval_set_seed: int = 42
    min_eval_samples: int = 100
    max_eval_samples: int = 5000
    
    # Question rephrasing
    rephrase_model: str = "gpt-4o-mini"  # Model for rephrasing questions
    rephrase_temperature: float = 0.7
    rephrase_max_retries: int = 3
    
    # Evaluation metrics
    compute_perplexity: bool = True
    compute_rouge: bool = True
    compute_bleu: bool = False
    generate_samples: bool = True
    num_generation_samples: int = 10
    
    # Validation
    validate_unchanged_answers: bool = True
    similarity_threshold: float = 0.95  # For answer validation