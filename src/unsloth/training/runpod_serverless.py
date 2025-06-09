"""RunPod Serverless training implementation.
Module: runpod_serverless.py
Description: Functions for runpod serverless operations

This is a more practical approach using RunPod's serverless endpoints'
which have better support than direct pod management.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import requests
import runpod
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_folder
from loguru import logger
from transformers import TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel


def download_file(url: str, dest_path: Path) -> None:
    """Download file from URL."""
    logger.info(f"Downloading from {url} to {dest_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def train_handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    Serverless training handler for RunPod.
    
    Expected input format:
    {
        "input": {
            "model_name": "unsloth/Phi-3.5-mini-instruct",
            "dataset_url": "https://example.com/dataset.jsonl",
            "hub_repo_id": "username/model-name",
            "hf_token": "hf_...",
            "training_config": {
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
                ...
            }
        }
    }
    """
    try:
        config = job["input"]
        job_id = job.get("id", "unknown")

        logger.info(f"Starting training job: {job_id}")
        logger.info(f"Model: {config['model_name']}")

        # Setup paths
        work_dir = Path(f"/tmp/training_{job_id}")
        work_dir.mkdir(exist_ok=True)
        output_dir = work_dir / "outputs"
        output_dir.mkdir(exist_ok=True)

        # Download dataset
        dataset_path = work_dir / "dataset.jsonl"
        download_file(config["dataset_url"], dataset_path)

        # Load model
        logger.info("Loading model and tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_name"],
            max_seq_length=config.get("max_seq_length", 2048),
            dtype=None,
            load_in_4bit=config.get("load_in_4bit", True)
        )

        # Apply LoRA
        lora_config = config.get("lora_config", {})
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get("r", 16),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_alpha=lora_config.get("lora_alpha", 16),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", True),
            random_state=lora_config.get("random_state", 3407)
        )

        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset("json", data_files=str(dataset_path))["train"]

        # Prepare dataset
        def formatting_func(examples):
            if "messages" in examples:
                return [tokenizer.apply_chat_template(msgs, tokenize=False)
                       for msgs in examples["messages"]]
            else:
                # Fallback for simple format
                texts = []
                for i in range(len(examples["question"])):
                    text = f"Question: {examples['question'][i]}\nAnswer: {examples['answer'][i]}"
                    texts.append(text)
                return texts

        # Training arguments
        training_config = config.get("training_config", {})
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=training_config.get("batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            num_train_epochs=training_config.get("num_train_epochs", 3),
            learning_rate=training_config.get("learning_rate", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            warmup_ratio=training_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            optim=training_config.get("optim", "adamw_8bit"),
            weight_decay=training_config.get("weight_decay", 0.01),
            max_grad_norm=training_config.get("max_grad_norm", 0.3),
            report_to="none"  # Disable reporting in serverless
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            max_seq_length=config.get("max_seq_length", 2048),
            formatting_func=formatting_func,
            args=training_args
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save model
        final_model_path = output_dir / "final_model"
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Upload to HuggingFace if configured
        hub_url = None
        if config.get("hub_repo_id") and config.get("hf_token"):
            logger.info(f"Uploading to HuggingFace: {config['hub_repo_id']}")

            api = HfApi(token=config["hf_token"])

            # Create repo if needed
            try:
                create_repo(
                    repo_id=config["hub_repo_id"],
                    token=config["hf_token"],
                    private=config.get("private_repo", True),
                    exist_ok=True
                )
            except Exception as e:
                logger.warning(f"Repo creation warning: {e}")

            # Upload
            upload_folder(
                folder_path=str(final_model_path),
                repo_id=config["hub_repo_id"],
                token=config["hf_token"]
            )

            hub_url = f"https://huggingface.co/{config['hub_repo_id']}"
            logger.info(f"Model uploaded to: {hub_url}")

        # Prepare results
        results = {
            "status": "success",
            "job_id": job_id,
            "model_name": config["model_name"],
            "training_loss": train_result.training_loss,
            "epochs_trained": train_result.state.num_train_epochs,
            "total_steps": train_result.state.global_step,
            "hub_url": hub_url
        }

        # Cleanup
        shutil.rmtree(work_dir)

        logger.info(f"Training completed successfully: {results}")
        return results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            "status": "error",
            "job_id": job.get("id", "unknown"),
            "error": str(e)
        }


# For local testing
def test_locally():
    """Test the handler locally."""
    test_job = {
        "id": "test_local",
        "input": {
            "model_name": "unsloth/Phi-3.5-mini-instruct",
            "dataset_url": "file:///path/to/dataset.jsonl",
            "hub_repo_id": "test/model",
            "hf_token": os.getenv("HF_TOKEN"),
            "max_seq_length": 2048,
            "lora_config": {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05
            },
            "training_config": {
                "num_train_epochs": 1,
                "batch_size": 4,
                "learning_rate": 2e-4
            }
        }
    }

    result = train_handler(test_job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # When deployed to RunPod, use their serverless handler
    if os.getenv("RUNPOD_POD_ID"):
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": train_handler})
    else:
        # Local testing
        logger.info("Running local test...")
        test_locally()
