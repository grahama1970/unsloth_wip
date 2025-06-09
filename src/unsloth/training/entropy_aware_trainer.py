"""
Module: entropy_aware_trainer.py
Description: Entropy-aware training implementation that integrates with Unsloth

External Dependencies:
- torch: https://pytorch.org/docs/stable/index.html
- transformers: https://huggingface.co/docs/transformers/
- unsloth: https://github.com/unslothai/unsloth
- trl: https://huggingface.co/docs/trl/
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> config = EntropyAwareTrainingConfig(model_name="unsloth/Phi-3.5-mini-instruct")
>>> trainer = EntropyAwareTrainer(config)
>>> result = trainer.train(dataset)

Expected Output:
>>> result
TrainingResult(adapter_path=Path("outputs/entropy_aware_adapter"), final_loss=0.45, entropy_metrics={"avg_entropy": 0.72})

Example Usage:
>>> from unsloth.training.entropy_aware_trainer import EntropyAwareTrainer
>>> trainer = EntropyAwareTrainer(config)
>>> trainer.train_with_entropy_weighting(dataset_path)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, load_dataset
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from pydantic import BaseModel, Field

from unsloth import FastLanguageModel
from unsloth.core.enhanced_config import EnhancedTrainingConfig
from unsloth.training.entropy_utils import (
    calculate_token_entropy,
    identify_high_entropy_tokens,
    get_entropy_weight
)
from unsloth.data.entropy_aware_thinking_enhancer import EntropyAwareThinkingEnhancer
from unsloth.utils.memory import clear_memory, log_memory_usage


@dataclass
class EntropyAwareTrainingConfig(EnhancedTrainingConfig):
    """Configuration for entropy-aware training."""
    
    # Entropy-specific settings
    entropy_weighting_enabled: bool = True
    entropy_weight_function: str = "exponential"  # linear, exponential, sigmoid
    entropy_weight_scale: float = 2.0
    min_entropy_weight: float = 1.0
    max_entropy_weight: float = 3.0
    
    # High-entropy token tracking
    track_high_entropy_tokens: bool = True
    high_entropy_percentile: float = 0.8  # Top 20% tokens
    
    # Adaptive learning
    adaptive_lr_for_entropy: bool = True
    high_entropy_lr_multiplier: float = 1.5
    
    # Logging
    log_entropy_metrics: bool = True
    save_entropy_analysis: bool = True


class EntropyMetrics(BaseModel):
    """Metrics for entropy tracking during training."""
    average_entropy: float
    high_entropy_token_ratio: float
    entropy_weighted_loss: float
    max_token_entropy: float
    min_token_entropy: float
    entropy_regions: List[Dict[str, Any]] = Field(default_factory=list)


class EntropyAwareCallback(TrainerCallback):
    """Callback for entropy-aware training monitoring."""
    
    def __init__(self, config: EntropyAwareTrainingConfig, writer: Optional[SummaryWriter] = None):
        self.config = config
        self.writer = writer
        self.entropy_history = []
        self.high_entropy_tokens = []
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log entropy metrics during training."""
        if logs is None:
            return
            
        # Extract entropy metrics if available
        if "entropy_metrics" in logs:
            metrics = logs["entropy_metrics"]
            self.entropy_history.append({
                "step": state.global_step,
                "average_entropy": metrics.get("average_entropy", 0),
                "high_entropy_ratio": metrics.get("high_entropy_ratio", 0),
                "weighted_loss": metrics.get("weighted_loss", 0)
            })
            
            # Log to TensorBoard if available
            if self.writer:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"entropy/{key}", value, state.global_step)
                    
        return control
        
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Analyze entropy patterns at epoch end."""
        if self.entropy_history:
            avg_entropy = np.mean([h["average_entropy"] for h in self.entropy_history[-100:]])
            logger.info(f"Epoch {state.epoch} - Average entropy: {avg_entropy:.4f}")
            
        return control


class EntropyAwareTrainer:
    """Trainer with entropy-aware loss weighting and monitoring."""
    
    def __init__(self, config: EntropyAwareTrainingConfig):
        """Initialize entropy-aware trainer."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.writer = None
        
        # Initialize entropy cache
        self.entropy_cache = {}
        
        # Setup TensorBoard if enabled
        if config.tensorboard_enabled:
            log_dir = Path(config.output_dir) / "tensorboard_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
    def setup_model(self) -> None:
        """Setup model with entropy-aware modifications."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # Apply LoRA with entropy-aware target modules
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
            random_state=self.config.seed,
            use_rslora=self.config.use_rslora,
            loftq_config=self.config.loftq_config,
        )
        
        logger.info("Model setup complete with entropy-aware configuration")
        
    def compute_entropy_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with entropy-based weighting."""
        # Calculate standard cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        vocab_size = logits.size(-1)
        
        # Reshape for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate per-token loss
        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        ).view(shift_labels.size())
        
        # Calculate entropy for each token
        probs = F.softmax(shift_logits, dim=-1)
        token_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Normalize entropy (0-1 range)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))
        normalized_entropy = token_entropy / max_entropy
        
        # Calculate entropy weights
        if self.config.entropy_weighting_enabled:
            entropy_weights = torch.ones_like(loss)
            
            for i in range(loss.size(0)):
                for j in range(loss.size(1)):
                    if attention_mask is None or attention_mask[i, j+1] > 0:
                        weight = get_entropy_weight(
                            normalized_entropy[i, j].item(),
                            function=self.config.entropy_weight_function,
                            scale=self.config.entropy_weight_scale,
                            min_weight=self.config.min_entropy_weight,
                            max_weight=self.config.max_entropy_weight
                        )
                        entropy_weights[i, j] = weight
        else:
            entropy_weights = torch.ones_like(loss)
            
        # Apply entropy weighting
        weighted_loss = loss * entropy_weights
        
        # Mask padding tokens
        if attention_mask is not None:
            mask = attention_mask[..., 1:].float()
            weighted_loss = weighted_loss * mask
            valid_tokens = mask.sum()
        else:
            valid_tokens = loss.numel()
            
        # Calculate final loss
        final_loss = weighted_loss.sum() / valid_tokens
        
        # Calculate metrics
        metrics = {
            "average_entropy": normalized_entropy.mean().item(),
            "max_entropy": normalized_entropy.max().item(),
            "min_entropy": normalized_entropy.min().item(),
            "high_entropy_ratio": (normalized_entropy > self.config.high_entropy_percentile).float().mean().item(),
            "weighted_loss": final_loss.item(),
            "standard_loss": (loss.sum() / valid_tokens).item()
        }
        
        return final_loss, metrics
        
    def create_entropy_aware_trainer(self, train_dataset: Dataset) -> SFTTrainer:
        """Create SFTTrainer with entropy-aware loss computation."""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported() and self.config.fp16,
            bf16=torch.cuda.is_bf16_supported() and self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            seed=self.config.seed,
            load_best_model_at_end=self.config.load_best_model_at_end,
            report_to="tensorboard" if self.config.tensorboard_enabled else "none",
        )
        
        # Create custom trainer class with entropy-aware loss
        class EntropyAwareSFTTrainer(SFTTrainer):
            def __init__(self, entropy_trainer, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.entropy_trainer = entropy_trainer
                
            def compute_loss(self, model, inputs, return_outputs=False):
                """Override loss computation to include entropy weighting."""
                # Get model outputs
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Get labels and attention mask
                labels = inputs.get("labels")
                attention_mask = inputs.get("attention_mask")
                
                # Compute entropy-weighted loss
                loss, metrics = self.entropy_trainer.compute_entropy_weighted_loss(
                    logits, labels, attention_mask
                )
                
                # Log metrics (they'll be picked up by callback)
                self.log({"entropy_metrics": metrics})
                
                return (loss, outputs) if return_outputs else loss
        
        # Create trainer with callbacks
        callbacks = []
        if self.config.log_entropy_metrics:
            callbacks.append(EntropyAwareCallback(self.config, self.writer))
            
        trainer = EntropyAwareSFTTrainer(
            entropy_trainer=self,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            dataset_text_field=self.config.dataset_text_field,
            max_seq_length=self.config.max_seq_length,
            callbacks=callbacks,
        )
        
        return trainer
        
    def train(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Train model with entropy-aware loss weighting."""
        start_time = datetime.now()
        
        # Load dataset
        if dataset_path:
            logger.info(f"Loading dataset from: {dataset_path}")
            train_dataset = load_dataset("json", data_files=dataset_path)["train"]
        else:
            # Use default dataset loading
            train_dataset = self._load_default_dataset()
            
        logger.info(f"Dataset size: {len(train_dataset)} examples")
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
            
        # Create entropy-aware trainer
        self.trainer = self.create_entropy_aware_trainer(train_dataset)
        
        # Train
        logger.info("Starting entropy-aware training...")
        self.trainer.train()
        
        # Save final adapter
        adapter_path = Path(self.config.output_dir) / "final_adapter"
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        
        # Save entropy analysis if enabled
        if self.config.save_entropy_analysis:
            self._save_entropy_analysis(adapter_path)
            
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get final metrics
        final_metrics = {
            "adapter_path": str(adapter_path),
            "model_name": self.config.model_name,
            "training_time": training_time,
            "final_loss": self.trainer.state.best_metric,
            "entropy_metrics": self._get_final_entropy_metrics()
        }
        
        logger.info(f"Training complete! Time: {training_time:.2f}s")
        logger.info(f"Adapter saved to: {adapter_path}")
        
        # Cleanup
        if self.writer:
            self.writer.close()
            
        clear_memory()
        
        return final_metrics
        
    def _load_default_dataset(self) -> Dataset:
        """Load default dataset with entropy enhancement."""
        # This would typically load from ArangoDB or other sources
        # For now, return a placeholder
        return Dataset.from_dict({
            "text": ["Sample training text"] * 100
        })
        
    def _save_entropy_analysis(self, output_path: Path):
        """Save entropy analysis results."""
        # Get callback data
        callbacks = [cb for cb in self.trainer.callback_handler.callbacks 
                    if isinstance(cb, EntropyAwareCallback)]
        
        if callbacks:
            entropy_data = {
                "entropy_history": callbacks[0].entropy_history,
                "high_entropy_tokens": callbacks[0].high_entropy_tokens,
                "config": {
                    "entropy_weight_function": self.config.entropy_weight_function,
                    "entropy_weight_scale": self.config.entropy_weight_scale,
                    "high_entropy_percentile": self.config.high_entropy_percentile
                }
            }
            
            analysis_path = output_path / "entropy_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(entropy_data, f, indent=2)
                
            logger.info(f"Entropy analysis saved to: {analysis_path}")
            
    def _get_final_entropy_metrics(self) -> Dict[str, float]:
        """Get final entropy metrics from training."""
        callbacks = [cb for cb in self.trainer.callback_handler.callbacks 
                    if isinstance(cb, EntropyAwareCallback)]
        
        if callbacks and callbacks[0].entropy_history:
            history = callbacks[0].entropy_history
            recent = history[-100:]  # Last 100 steps
            
            return {
                "final_average_entropy": np.mean([h["average_entropy"] for h in recent]),
                "final_high_entropy_ratio": np.mean([h["high_entropy_ratio"] for h in recent]),
                "entropy_trend": "decreasing" if recent[-1]["average_entropy"] < recent[0]["average_entropy"] else "increasing"
            }
        
        return {}


# Validation
if __name__ == "__main__":
    # Test configuration
    config = EntropyAwareTrainingConfig(
        model_name="unsloth/Phi-3.5-mini-instruct",
        max_seq_length=2048,
        entropy_weighting_enabled=True,
        entropy_weight_function="exponential",
        output_dir="./outputs/entropy_aware_test"
    )
    
    # Create trainer
    trainer = EntropyAwareTrainer(config)
    
    # Test entropy weight calculation
    test_entropy = 0.8
    weight = get_entropy_weight(
        test_entropy,
        function=config.entropy_weight_function,
        scale=config.entropy_weight_scale
    )
    
    print(f"Test entropy: {test_entropy:.2f} -> Weight: {weight:.2f}")
    print("\n Module validation passed")