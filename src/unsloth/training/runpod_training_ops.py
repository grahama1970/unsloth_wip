"""RunPod training operations for Unsloth - based on runpod_llm_ops pattern.

This module provides proper RunPod pod management for training large models,
including Docker image deployment, training execution, and result saving.
"""

import os
import asyncio
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import uuid

import runpod
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from yaspin import yaspin
from yaspin.spinners import Spinners
from huggingface_hub import HfApi, upload_folder


# Training-specific configurations
TRAINING_CONFIGS = {
    "unsloth-phi-3.5": {
        "name": "unsloth-phi-3.5-training",
        "image_name": "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04",
        "docker_args": "bash -c 'cd /workspace && python train.py'",
        "cloud_type": "SECURE",
        "volume_in_gb": 100,
        "ports": "8888/http,6006/http",  # Jupyter + TensorBoard
        "container_disk_in_gb": 50,
        "volume_mount_path": "/workspace",
        "env": {
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        },
        "preferred_gpu_names": ["RTX 4090", "RTX A6000", "A100 PCIe"],
    },
    
    "unsloth-llama-13b": {
        "name": "unsloth-llama-13b-training",
        "image_name": "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04",
        "docker_args": "bash -c 'cd /workspace && python train.py'",
        "cloud_type": "SECURE",
        "volume_in_gb": 200,
        "ports": "8888/http,6006/http",
        "container_disk_in_gb": 100,
        "volume_mount_path": "/workspace",
        "env": {
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        },
        "preferred_gpu_names": ["A100 PCIe", "A100 SXM", "RTX A6000"],
    },
    
    "unsloth-llama-70b": {
        "name": "unsloth-llama-70b-training",
        "image_name": "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04",
        "docker_args": "bash -c 'cd /workspace && python train.py'",
        "cloud_type": "SECURE",
        "volume_in_gb": 500,
        "ports": "8888/http,6006/http",
        "container_disk_in_gb": 200,
        "volume_mount_path": "/workspace",
        "env": {
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        },
        "preferred_gpu_names": ["H100 PCIe", "H100 SXM", "H100 NVL", "A100 SXM 80GB"],
    }
}


class RunPodTrainingOps:
    """Manages RunPod training operations for Unsloth."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize RunPod training operations."""
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not found")
            
        runpod.api_key = self.api_key
        self.pod = None
        self.api_base = None
        
    def get_training_config(self, model_size: str) -> Dict[str, Any]:
        """Get training configuration based on model size."""
        if "70b" in model_size.lower():
            return TRAINING_CONFIGS["unsloth-llama-70b"]
        elif "13b" in model_size.lower() or "30b" in model_size.lower():
            return TRAINING_CONFIGS["unsloth-llama-13b"]
        else:
            return TRAINING_CONFIGS["unsloth-phi-3.5"]
            
    async def start_training_pod(
        self,
        model_config: Dict[str, Any],
        reuse_existing: bool = True
    ) -> Dict[str, Any]:
        """Start a RunPod training pod with retry logic."""
        
        # Check for existing pods if reuse is enabled
        if reuse_existing:
            existing_pods = runpod.get_pods()
            matching_pod = next(
                (pod for pod in existing_pods if pod["name"] == model_config["name"]),
                None
            )
            
            if matching_pod:
                if matching_pod["desiredStatus"] == "RUNNING":
                    logger.info(f"Using existing running pod: {matching_pod['id']}")
                    self.pod = matching_pod
                    self.api_base = f"https://{matching_pod['id']}-8888.proxy.runpod.net"
                    return {
                        "id": matching_pod["id"],
                        "status": "reused",
                        "api_base": self.api_base
                    }
                elif matching_pod["desiredStatus"] in ["EXITED", "STOPPED"]:
                    logger.info(f"Cleaning up existing pod: {matching_pod['id']}")
                    runpod.terminate_pod(matching_pod["id"])
                    await asyncio.sleep(5)
        
        # Start new pod
        logger.info(f"Starting new training pod: {model_config['name']}")
        
        # Get available GPUs
        available_gpus = runpod.get_gpus()
        if not available_gpus:
            raise RuntimeError("No available GPUs found")
            
        # Prioritize GPUs
        preferred_gpus = sorted(
            [gpu for gpu in available_gpus if gpu["displayName"] in model_config["preferred_gpu_names"]],
            key=lambda gpu: model_config["preferred_gpu_names"].index(gpu["displayName"])
        )
        fallback_gpus = [gpu for gpu in available_gpus if gpu not in preferred_gpus]
        
        gpus_to_try = preferred_gpus + fallback_gpus
        
        # Try to create pod with available GPUs
        for gpu in gpus_to_try:
            try:
                pod_config = {key: value for key, value in model_config.items() 
                             if key != "preferred_gpu_names"}
                pod_config["gpu_type_id"] = gpu["id"]
                
                # Filter to valid keys only
                valid_keys = [
                    'name', 'image_name', 'gpu_type_id', 'cloud_type',
                    'gpu_count', 'volume_in_gb', 'container_disk_in_gb',
                    'docker_args', 'ports', 'volume_mount_path', 'env'
                ]
                filtered_config = {k: v for k, v in pod_config.items() if k in valid_keys}
                
                logger.info(f"Attempting to start pod with GPU: {gpu['displayName']}")
                self.pod = runpod.create_pod(**filtered_config)
                
                logger.info(f"Successfully created pod: {self.pod['id']}")
                
                # Wait for pod to be ready
                await self._wait_for_pod_ready()
                
                return {
                    "id": self.pod["id"],
                    "status": "created",
                    "gpu": gpu["displayName"],
                    "api_base": self.api_base
                }
                
            except Exception as e:
                logger.warning(f"Failed with GPU {gpu['displayName']}: {e}")
                continue
                
        raise RuntimeError("Failed to start pod with any available GPU")
        
    async def _wait_for_pod_ready(self, max_wait: int = 900):
        """Wait for pod to be ready."""
        start_time = datetime.now(timezone.utc)
        
        with yaspin(Spinners.dots, text="Waiting for pod to initialize...") as spinner:
            while (datetime.now(timezone.utc) - start_time).total_seconds() < max_wait:
                pod = runpod.get_pod(self.pod["id"])
                
                if pod.get("desiredStatus") == "RUNNING":
                    logger.info("Pod has reached RUNNING status")
                    self.pod = pod
                    self.api_base = f"https://{pod['id']}-8888.proxy.runpod.net"
                    
                    # Check if API is ready
                    if await self._check_api_readiness():
                        spinner.succeed("Pod is ready!")
                        return
                        
                await asyncio.sleep(10)
                
        raise TimeoutError("Pod startup timed out")
        
    @retry(
        stop=stop_after_attempt(30),
        wait=wait_fixed(10),
        retry=retry_if_exception_type(Exception)
    )
    async def _check_api_readiness(self) -> bool:
        """Check if pod API is ready."""
        endpoints = ["/health", "/"]
        
        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                try:
                    response = await client.get(
                        f"{self.api_base}{endpoint}",
                        timeout=10
                    )
                    if response.status_code in [200, 404]:  # 404 is ok, means server is up
                        logger.info(f"API readiness confirmed")
                        return True
                except:
                    pass
                    
        raise RuntimeError("API readiness check failed")
        
    async def upload_training_script(
        self,
        training_config: Dict[str, Any],
        dataset_path: Path,
        output_path: Path
    ) -> None:
        """Upload training script and configuration to pod."""
        logger.info("Uploading training script to pod...")
        
        # Create training script content
        training_script = self._create_training_script(
            training_config,
            dataset_path,
            output_path
        )
        
        # Save locally first
        local_script = Path("/tmp/train.py")
        with open(local_script, "w") as f:
            f.write(training_script)
            
        # Upload using SCP or HTTP endpoint
        # Note: This would need actual implementation based on RunPod's file transfer API
        logger.info("Training script prepared (actual upload would happen here)")
        
    def _create_training_script(
        self,
        config: Dict[str, Any],
        dataset_path: Path,
        output_path: Path
    ) -> str:
        """Create the training script content."""
        return f'''#!/usr/bin/env python3
"""Auto-generated training script for RunPod execution."""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from huggingface_hub import HfApi, upload_folder
from loguru import logger

# Install dependencies
os.system("pip install unsloth[colab-new] loguru datasets trl")

# Configuration
config = {json.dumps(config, indent=2)}

# Setup paths
output_dir = Path("/workspace/outputs")
output_dir.mkdir(exist_ok=True)

# Progress file for monitoring
progress_file = Path("/workspace/training_progress.json")

def update_progress(status, **kwargs):
    """Update training progress."""
    progress = {{
        "status": status,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }}
    with open(progress_file, "w") as f:
        json.dump(progress, f)

try:
    update_progress("starting")
    
    # Load model
    logger.info(f"Loading model: {{config['model_name']}}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config.get("load_in_4bit", True)
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing=config.get("gradient_checkpointing", True)
    )
    
    update_progress("model_loaded")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files="{dataset_path}")["train"]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        optim=config.get("optim", "adamw_8bit"),
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        args=training_args
    )
    
    # Train
    update_progress("training")
    trainer.train()
    
    # Save
    update_progress("saving")
    final_path = output_dir / "final_adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Upload to HuggingFace if configured
    if config.get("hub_model_id") and os.getenv("HF_TOKEN"):
        update_progress("uploading")
        api = HfApi(token=os.getenv("HF_TOKEN"))
        upload_folder(
            folder_path=str(final_path),
            repo_id=config["hub_model_id"],
            token=os.getenv("HF_TOKEN")
        )
        logger.info(f"Uploaded to: {{config['hub_model_id']}}")
    
    update_progress("completed", adapter_path=str(final_path))
    logger.info("Training completed successfully!")
    
except Exception as e:
    logger.error(f"Training failed: {{e}}")
    update_progress("failed", error=str(e))
    raise
'''
        
    async def monitor_training(self) -> AsyncIterator[Dict[str, Any]]:
        """Monitor training progress."""
        progress_url = f"{self.api_base}/workspace/training_progress.json"
        
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.get(progress_url, timeout=10)
                    if response.status_code == 200:
                        progress = response.json()
                        yield progress
                        
                        if progress.get("status") in ["completed", "failed"]:
                            break
                            
                except Exception as e:
                    logger.debug(f"Progress check error: {e}")
                    
                await asyncio.sleep(30)
                
    async def download_results(self, output_path: Path) -> Path:
        """Download training results from pod."""
        logger.info("Downloading training results...")
        
        # Create local directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Note: Actual download would use RunPod's file transfer API
        # This is a placeholder showing the intended functionality
        adapter_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model"
        ]
        
        logger.info(f"Results would be downloaded to: {output_path}")
        return output_path
        
    @retry(
        stop=stop_after_attempt(30),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(Exception)
    )
    async def stop_pod(self, terminate: bool = True) -> None:
        """Stop the training pod."""
        if not self.pod:
            return
            
        pod_id = self.pod["id"]
        logger.info(f"Stopping pod: {pod_id}")
        
        try:
            if terminate:
                response = runpod.terminate_pod(pod_id)
                logger.info(f"Pod terminated: {response}")
            else:
                response = runpod.stop_pod(pod_id)
                
                # Verify it stopped
                pod = runpod.get_pod(pod_id)
                if pod.get("desiredStatus") not in ["EXITED", "STOPPED"]:
                    raise RuntimeError(f"Pod still in state: {pod.get('desiredStatus')}")
                    
            logger.info("Pod stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop pod: {e}")
            if terminate:
                raise


async def run_training_on_runpod(
    model_name: str,
    dataset_path: Path,
    training_config: Dict[str, Any],
    hub_model_id: Optional[str] = None
) -> Dict[str, Any]:
    """Complete training workflow on RunPod."""
    
    ops = RunPodTrainingOps()
    
    try:
        # Get appropriate config
        model_size = model_name.split("-")[-1] if "-" in model_name else "7b"
        pod_config = ops.get_training_config(model_size)
        
        # Start pod
        logger.info(f"Starting RunPod for {model_name}")
        pod_info = await ops.start_training_pod(pod_config)
        
        # Upload training script
        await ops.upload_training_script(
            training_config,
            dataset_path,
            Path("/workspace/outputs")
        )
        
        # Monitor training
        logger.info("Monitoring training progress...")
        final_status = None
        
        async for progress in ops.monitor_training():
            logger.info(f"Training status: {progress.get('status')}")
            final_status = progress
            
        # Download results if successful
        if final_status and final_status.get("status") == "completed":
            local_path = Path("./outputs/runpod_results")
            adapter_path = await ops.download_results(local_path)
            
            return {
                "status": "success",
                "adapter_path": str(adapter_path),
                "pod_id": pod_info["id"],
                "training_time": final_status.get("training_time")
            }
        else:
            return {
                "status": "failed",
                "error": final_status.get("error", "Unknown error"),
                "pod_id": pod_info["id"]
            }
            
    finally:
        # Always clean up
        await ops.stop_pod(terminate=True)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test():
        result = await run_training_on_runpod(
            model_name="unsloth/Phi-3.5-mini-instruct",
            dataset_path=Path("./data/qa_enhanced.jsonl"),
            training_config={
                "model_name": "unsloth/Phi-3.5-mini-instruct",
                "max_seq_length": 2048,
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "hub_model_id": "username/test-model"
            }
        )
        print(result)
        
    asyncio.run(test())