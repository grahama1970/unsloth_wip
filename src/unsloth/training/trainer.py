"""Main training module for Unsloth fine-tuning with LoRA adapters."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import TrainingConfig
from ..data.loader import ArangoDBDataLoader
from ..utils.memory import clear_memory


class TrainingResult(BaseModel):
    """Training result information."""
    adapter_path: Path
    model_name: str
    dataset_name: str
    training_time: float
    final_loss: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class UnslothTrainer:
    """Unified trainer for Unsloth fine-tuning with support for various datasets."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_model(self) -> None:
        """Setup the model and tokenizer with LoRA configuration."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # Setup LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.random_state,
            use_rslora=self.config.use_rslora,
            loftq_config=self.config.loftq_config,
        )
        
        logger.info("Model and LoRA adapters configured successfully")
        
    def load_dataset(self) -> Dataset:
        """Load and prepare the dataset based on configuration."""
        if self.config.dataset_source == "arangodb":
            # Load from ArangoDB QA output
            loader = ArangoDBDataLoader(
                qa_path=self.config.dataset_path,
                validation_split=self.config.validation_split,
                max_samples=self.config.max_samples,
                metadata_filters=self.config.metadata_filters
            )
            dataset = loader.load_dataset()
            
        elif self.config.dataset_source == "huggingface":
            # Load from HuggingFace
            dataset = load_dataset(
                self.config.dataset_path,
                split=self.config.dataset_split
            )
            
        else:
            raise ValueError(f"Unknown dataset source: {self.config.dataset_source}")
            
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        return dataset
        
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Apply chat template and tokenization to the dataset."""
        # Get appropriate chat template
        chat_template = self._get_chat_template()
        
        # Apply template to dataset
        if "messages" in dataset.column_names:
            # Already in chat format (e.g., from ArangoDB)
            dataset = dataset.map(
                lambda x: {
                    "text": self.tokenizer.apply_chat_template(
                        x["messages"],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                }
            )
        else:
            # Convert to chat format
            dataset = dataset.map(
                lambda x: {
                    "text": chat_template.format(
                        question=x.get("question", x.get("prompt", "")),
                        answer=x.get("answer", x.get("response", ""))
                    )
                }
            )
            
        return dataset
        
    def _get_chat_template(self) -> str:
        """Get the appropriate chat template for the model."""
        model_lower = self.config.model_name.lower()
        
        if "phi" in model_lower:
            return "<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
        elif "llama" in model_lower:
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        else:
            # Default ShareGPT style
            return "### Human: {question}\n\n### Assistant: {answer}"
            
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments."""
        return TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported() and self.config.fp16,
            bf16=torch.cuda.is_bf16_supported() and self.config.bf16,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
        )
        
    def train(self) -> TrainingResult:
        """Execute the training process."""
        start_time = datetime.now()
        
        # Setup model
        self.setup_model()
        
        # Load and prepare dataset
        dataset = self.load_dataset()
        dataset = self.prepare_dataset(dataset)
        
        # Split dataset if needed
        if self.config.validation_split > 0 and "train" not in dataset:
            split = dataset.train_test_split(test_size=self.config.validation_split)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
            
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=self.config.dataset_num_proc,
            packing=self.config.packing,
            args=training_args,
        )
        
        # Train
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save adapter
        adapter_path = Path(self.config.output_dir) / "final_adapter"
        logger.info(f"Saving adapter to {adapter_path}")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get final metrics
        metrics = {}
        if self.trainer.state.log_history:
            final_log = self.trainer.state.log_history[-1]
            metrics = {k: v for k, v in final_log.items() if isinstance(v, (int, float))}
            
        result = TrainingResult(
            adapter_path=adapter_path,
            model_name=self.config.model_name,
            dataset_name=self.config.dataset_path,
            training_time=training_time,
            final_loss=metrics.get("loss"),
            metrics=metrics
        )
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return result
        
    def cleanup(self):
        """Cleanup resources after training."""
        if self.model:
            del self.model
        if self.trainer:
            del self.trainer
        clear_memory()
        logger.info("Cleaned up training resources")