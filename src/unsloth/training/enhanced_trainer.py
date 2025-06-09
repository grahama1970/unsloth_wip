"""
Module: enhanced_trainer.py
Description: Enhanced training module with advanced monitoring and optimization features

External Dependencies:
- torch: https://pytorch.org/docs/stable/index.html
- transformers: https://huggingface.co/docs/transformers/
- unsloth: https://github.com/unslothai/unsloth
- trl: https://huggingface.co/docs/trl/
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> config = EnhancedTrainingConfig(model_name="unsloth/Phi-3.5-mini-instruct")
>>> trainer = EnhancedTrainer(config)

Expected Output:
>>> result = trainer.train(dataset)
>>> result.final_loss
0.45

Example Usage:
>>> from unsloth.training.enhanced_trainer import EnhancedTrainer
>>> trainer = EnhancedTrainer(config)
>>> trainer.train()
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

from unsloth import FastLanguageModel

from ..core.enhanced_config import EnhancedTrainingConfig
from ..data.loader import ArangoDBDataLoader
from ..data.thinking_enhancer import StudentTeacherConfig, ThinkingEnhancer
from ..utils.memory import clear_memory, log_memory_usage
from .grokking_callback import GrokkingCallback


class TensorBoardCallback(TrainerCallback):
    """Custom callback for enhanced TensorBoard logging."""

    def __init__(self, writer: SummaryWriter, config: EnhancedTrainingConfig):
        self.writer = writer
        self.config = config

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log metrics to TensorBoard."""
        if logs is None:
            return

        # Log scalar metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"training/{key}", value, state.global_step)

        # Log learning rate if scheduler is active
        if self.config.log_learning_rate and "learning_rate" in logs:
            self.writer.add_scalar("training/learning_rate", logs["learning_rate"], state.global_step)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log epoch-level metrics."""
        epoch = state.epoch
        if epoch is not None:
            self.writer.add_scalar("training/epoch", epoch, state.global_step)

            # Log memory usage
            if torch.cuda.is_available():
                self.writer.add_scalar(
                    "system/gpu_memory_allocated_gb",
                    torch.cuda.memory_allocated() / 1024**3,
                    state.global_step
                )

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Finalize TensorBoard logging."""
        self.writer.close()


class EnhancedUnslothTrainer:
    """Enhanced trainer with advanced features and monitoring."""

    def __init__(self, config: EnhancedTrainingConfig, student_teacher_config: StudentTeacherConfig | None = None):
        """Initialize the enhanced trainer."""
        self.config = config
        self.student_teacher_config = student_teacher_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.writer = None
        self.thinking_enhancer = None

        # Setup thinking enhancer if configured
        if student_teacher_config:
            # Will initialize after model is loaded to use same model as student
            self.thinking_enhancer = None
            logger.info("Student-teacher enhancement configured")

        # Setup TensorBoard
        if "tensorboard" in self.config.report_to:
            Path(self.config.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.config.tensorboard_log_dir)
            logger.info(f"TensorBoard logging to: {self.config.tensorboard_log_dir}")

    def setup_model(self) -> None:
        """Setup model with enhanced LoRA configuration."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )

        # Log model info to TensorBoard
        if self.writer:
            model_info = {
                "model_name": self.config.model_name,
                "max_seq_length": self.config.max_seq_length,
                "load_in_4bit": self.config.load_in_4bit,
            }
            self.writer.add_text("model/info", json.dumps(model_info, indent=2))

        # Setup LoRA with enhanced configuration
        lora_config = {
            "r": self.config.lora_r,
            "target_modules": self.config.target_modules,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "bias": self.config.bias,
            "use_gradient_checkpointing": self.config.use_gradient_checkpointing,
            "random_state": self.config.random_state,
            "use_rslora": self.config.use_rslora,
            "loftq_config": self.config.loftq_config,
        }

        # Add advanced LoRA settings if available
        if self.config.use_dora:
            lora_config["use_dora"] = True
        if self.config.modules_to_save:
            lora_config["modules_to_save"] = self.config.modules_to_save
        if self.config.rank_pattern:
            lora_config["rank_pattern"] = self.config.rank_pattern
        if self.config.alpha_pattern:
            lora_config["alpha_pattern"] = self.config.alpha_pattern

        self.model = FastLanguageModel.get_peft_model(self.model, **lora_config)

        # Log LoRA config
        if self.writer:
            self.writer.add_text("lora/config", json.dumps(lora_config, indent=2))

        # Calculate and log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = trainable_params / total_params * 100

        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
        if self.writer:
            self.writer.add_scalar("model/trainable_parameters", trainable_params)
            self.writer.add_scalar("model/trainable_percentage", trainable_percent)

    def compute_metrics(self, eval_preds) -> dict[str, float]:
        """Compute evaluation metrics including perplexity."""
        predictions, labels = eval_preds

        # Ensure tensors
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Calculate perplexity
        loss = F.cross_entropy(
            predictions.view(-1, predictions.size(-1)),
            labels.view(-1),
            reduction='mean',
            ignore_index=-100
        )
        perplexity = torch.exp(loss).item()

        metrics = {
            "perplexity": perplexity,
            "loss": loss.item()
        }

        # Log to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"eval/{key}", value)

        return metrics

    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments with enhanced features."""
        training_args = TrainingArguments(
            # Basic settings
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,

            # Precision and optimization
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,

            # Logging
            logging_steps=self.config.logging_steps,
            logging_first_step=self.config.logging_first_step,
            logging_nan_inf_filter=self.config.logging_nan_inf_filter,
            report_to=self.config.report_to,
            run_name=self.config.run_name,

            # Saving
            output_dir=self.config.output_dir,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,

            # Evaluation
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            eval_accumulation_steps=self.config.eval_accumulation_steps,
            include_inputs_for_metrics=self.config.include_inputs_for_metrics,

            # Model selection
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,

            # Other
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            group_by_length=self.config.group_by_length,
            length_column_name=self.config.length_column_name,

            # Advanced options
            torch_compile=self.config.torch_compile,
            optim_args=self.config.optim_args,
            gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs,

            # NEFTune
            neftune_noise_alpha=self.config.neftune_noise_alpha,
        )

        return training_args

    def train(self, enhance_thinking: bool = False) -> dict[str, Any]:
        """Execute enhanced training with monitoring.
        
        Args:
            enhance_thinking: Whether to enhance dataset with student-teacher thinking
        """
        start_time = time.time()

        # Log initial memory
        log_memory_usage("Before model setup")

        # Setup model
        self.setup_model()
        log_memory_usage("After model setup")

        # Initialize thinking enhancer with the model we're training
        if enhance_thinking and self.student_teacher_config:
            self.thinking_enhancer = ThinkingEnhancer(
                config=self.student_teacher_config,
                base_model_name=self.config.model_name
            )
            logger.info(f"Initialized thinking enhancer with student model: {self.config.model_name}")

        # Load dataset
        dataset = self.load_dataset()

        # Enhance thinking if requested and enhancer is available
        if enhance_thinking and self.thinking_enhancer:
            logger.info("Enhancing dataset with student-teacher thinking...")
            # Save to temp file for enhancement
            temp_input = Path(self.config.output_dir) / "temp_qa_input.jsonl"
            temp_output = Path(self.config.output_dir) / "temp_qa_enhanced.jsonl"

            # Convert dataset to JSONL for enhancement
            self._save_dataset_as_jsonl(dataset, temp_input)

            # Enhance dataset
            import asyncio
            enhancement_stats = asyncio.run(
                self.thinking_enhancer.enhance_dataset(
                    input_path=temp_input,
                    output_path=temp_output,
                    max_samples=self.config.max_samples
                )
            )

            logger.info(f"Enhancement stats: {enhancement_stats}")

            # Reload enhanced dataset
            dataset = self._load_enhanced_dataset(temp_output)

            # Cleanup temp files
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)

        # Prepare dataset
        dataset = self.prepare_dataset(dataset)

        # Split dataset
        if self.config.validation_split > 0 and "train" not in dataset:
            split = dataset.train_test_split(test_size=self.config.validation_split)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset) if eval_dataset else 0}")

        # Log dataset info
        if self.writer:
            dataset_info = {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset) if eval_dataset else 0,
                "max_seq_length": self.config.max_seq_length,
            }
            self.writer.add_text("dataset/info", json.dumps(dataset_info, indent=2))

        # Setup training arguments
        training_args = self.setup_training_args()

        # Setup callbacks
        callbacks = []

        # Add TensorBoard callback
        if self.writer:
            callbacks.append(TensorBoardCallback(self.writer, self.config))

        # Add early stopping if configured
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )

        # Add grokking callback if enabled
        if self.config.grokking and self.config.grokking.enable_grokking:
            callbacks.append(
                GrokkingCallback(
                    grokking_config=self.config.grokking,
                    writer=self.writer,
                    base_weight_decay=self.config.weight_decay
                )
            )
            logger.info(" Grokking mode enabled - extended training with enhanced regularization")

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
            callbacks=callbacks,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            preprocess_logits_for_metrics=self.config.preprocess_logits_for_metrics,
        )

        # Train
        logger.info("Starting enhanced training...")
        train_result = self.trainer.train()

        # Save final model
        final_path = Path(self.config.output_dir) / "final_adapter"
        logger.info(f"Saving final adapter to {final_path}")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        # Calculate training time
        training_time = time.time() - start_time

        # Prepare results
        results = {
            "adapter_path": str(final_path),
            "model_name": self.config.model_name,
            "dataset_path": self.config.dataset_path,
            "training_time": training_time,
            "train_result": train_result.metrics,
            "config": {
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_train_epochs,
            }
        }

        # Log final results
        if self.writer:
            self.writer.add_text("training/final_results", json.dumps(results, indent=2))

        logger.info(f"Training completed in {training_time:.2f} seconds")
        return results

    def load_dataset(self) -> Dataset:
        """Load dataset with enhanced configuration."""
        if self.config.dataset_source == "arangodb":
            loader = ArangoDBDataLoader(
                qa_path=self.config.dataset_path,
                validation_split=0,  # Handle split separately
                max_samples=self.config.max_samples,
                metadata_filters=self.config.metadata_filters
            )
            dataset = loader.load_dataset()
        else:
            dataset = load_dataset(
                self.config.dataset_path,
                split=self.config.dataset_split
            )

        return dataset

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset with enhanced preprocessing."""
        # Get chat template
        chat_template = self._get_chat_template()

        # Apply template
        if "messages" in dataset.column_names:
            dataset = dataset.map(
                lambda x: {
                    "text": self.tokenizer.apply_chat_template(
                        x["messages"],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                },
                num_proc=self.config.dataset_num_proc
            )
        else:
            dataset = dataset.map(
                lambda x: {
                    "text": chat_template.format(
                        question=x.get("question", x.get("prompt", "")),
                        answer=x.get("answer", x.get("response", ""))
                    )
                },
                num_proc=self.config.dataset_num_proc
            )

        # Add length column for grouping
        if self.config.group_by_length:
            dataset = dataset.map(
                lambda x: {self.config.length_column_name: len(x["text"])},
                num_proc=self.config.dataset_num_proc
            )

        return dataset

    def _get_chat_template(self) -> str:
        """Get appropriate chat template."""
        model_lower = self.config.model_name.lower()

        if "phi" in model_lower:
            return "<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
        elif "llama" in model_lower:
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        else:
            return "### Human: {question}\n\n### Assistant: {answer}"

    def _save_dataset_as_jsonl(self, dataset: Dataset, output_path: Path):
        """Save dataset in JSONL format for thinking enhancement."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                # Convert to expected format
                jsonl_example = {
                    "messages": example.get("messages", []),
                    "metadata": example.get("metadata", {})
                }
                f.write(json.dumps(jsonl_example, ensure_ascii=False) + '\n')

    def _load_enhanced_dataset(self, input_path: Path) -> Dataset:
        """Load enhanced dataset from JSONL."""
        examples = []
        with open(input_path, encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        return Dataset.from_list(examples)

    def cleanup(self):
        """Enhanced cleanup with TensorBoard closure."""
        if self.writer:
            self.writer.close()

        if self.model:
            del self.model
        if self.trainer:
            del self.trainer
        if self.thinking_enhancer:
            del self.thinking_enhancer

        clear_memory()
        log_memory_usage("After cleanup")
        logger.info("Enhanced cleanup completed")
