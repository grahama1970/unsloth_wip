"""Configuration classes for Unsloth training pipeline."""
Module: config.py
Description: Configuration management and settings

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainingConfig:
    """Configuration for training with Unsloth."""

    # Model settings
    model_name: str = "unsloth/Phi-3.5-mini-instruct"
    max_seq_length: int = 2048
    dtype: torch.dtype | None = None
    load_in_4bit: bool = True

    # Dataset settings
    dataset_source: str = "arangodb"  # "arangodb" or "huggingface"
    dataset_path: str = "/home/graham/workspace/experiments/arangodb/qa_output"
    dataset_split: str = "train"
    validation_split: float = 0.1
    max_samples: int | None = None
    metadata_filters: dict[str, Any] = field(default_factory=lambda: {})

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: dict[str, Any] | None = None

    # Training arguments
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    fp16: bool = not torch.cuda.is_bf16_supported()
    bf16: bool = torch.cuda.is_bf16_supported()
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 3407

    # Output settings
    output_dir: str = "./outputs"
    save_strategy: str = "steps"
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: str | None = None

    # Processing settings
    dataset_num_proc: int = 2
    packing: bool = False

    # Adapter settings
    adapter_name: str | None = None
    force_retrain: bool = False

    def __post_init__(self):
        """Post initialization processing."""
        # Generate adapter name if not provided
        if not self.adapter_name:
            model_short = self.model_name.split("/")[-1].lower()
            dataset_short = Path(self.dataset_path).stem if "/" in self.dataset_path else self.dataset_path
            self.adapter_name = f"{model_short}_{dataset_short}_lora"

        # Set run name if not provided
        if not self.run_name:
            self.run_name = self.adapter_name


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    model_path: str | Path
    device: str = "cuda"
    load_in_4bit: bool = True
    max_seq_length: int = 2048

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    use_cache: bool = True

    # System prompt
    system_prompt: str | None = None


@dataclass
class UploadConfig:
    """Configuration for model upload."""

    # Required fields
    repo_name: str
    base_model: str
    dataset_name: str

    # Optional fields with defaults
    hf_token: str | None = None
    private: bool = False
    exist_ok: bool = True
    description: str = ""
    tags: list[str] = field(default_factory=lambda: [])
    language: str = "en"
    license: str = "apache-2.0"
