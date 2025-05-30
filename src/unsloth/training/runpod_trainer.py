"""RunPod integration for training large models that exceed local GPU capacity."""

import os
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncIterator
from datetime import datetime
from dataclasses import dataclass, field

import runpod
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import torch

from ..core.enhanced_config import EnhancedTrainingConfig


@dataclass
class RunPodTrainingConfig(EnhancedTrainingConfig):
    """Extended configuration for RunPod training."""
    
    # RunPod specific settings
    runpod_api_key: Optional[str] = None
    pod_name: str = "unsloth-training"
    gpu_type: str = "RTX A6000"  # Default for 7B models
    preferred_gpus: List[str] = field(default_factory=lambda: [])
    fallback_gpus: List[str] = field(default_factory=lambda: [])
    
    # Pod configuration
    cloud_type: str = "SECURE"
    volume_size_gb: int = 100
    container_disk_gb: int = 50
    output_mount_path: str = "/workspace/outputs"
    
    # Resource optimization
    use_flash_attention_2: bool = True
    use_fsdp: bool = False  # For very large models
    fsdp_config: Dict[str, Any] = field(default_factory=dict)
    cpu_offload: bool = False
    optimizer_offload: bool = False
    
    # Monitoring
    monitor_interval: int = 30  # seconds
    checkpoint_interval: int = 500  # steps
    
    def __post_init__(self):
        """Initialize RunPod-specific settings."""
        super().__post_init__()
        
        # Set API key
        if not self.runpod_api_key:
            self.runpod_api_key = os.getenv("RUNPOD_API_KEY")
        if not self.runpod_api_key:
            raise ValueError("RUNPOD_API_KEY not found")
            
        # Set preferred GPUs based on model size
        if not self.preferred_gpus:
            model_size = self._estimate_model_size()
            self.preferred_gpus = self._get_recommended_gpus(model_size)
            
    def _estimate_model_size(self) -> str:
        """Estimate model size from name."""
        model_lower = self.model_name.lower()
        if "70b" in model_lower:
            return "70B"
        elif "30b" in model_lower or "33b" in model_lower:
            return "30B"
        elif "13b" in model_lower:
            return "13B"
        elif "7b" in model_lower:
            return "7B"
        else:
            return "7B"  # Default
            
    def _get_recommended_gpus(self, model_size: str) -> List[str]:
        """Get recommended GPUs for model size."""
        gpu_recommendations = {
            "7B": ["RTX 4090", "RTX A6000", "A100 PCIe"],
            "13B": ["A100 PCIe", "A100 SXM", "RTX A6000"],
            "30B": ["A100 SXM", "H100 PCIe"],
            "70B": ["H100 PCIe", "H100 SXM", "H100 NVL"]
        }
        return gpu_recommendations.get(model_size, ["RTX A6000"])


class RunPodTrainer:
    """Trainer for running Unsloth on RunPod infrastructure."""
    
    def __init__(self, config: RunPodTrainingConfig):
        """Initialize RunPod trainer."""
        self.config = config
        self.pod = None
        self.api_base = None
        
        # Set RunPod API key
        runpod.api_key = config.runpod_api_key
        
        # Create training script
        self.training_script = self._create_training_script()
        
    async def start_training_pod(self) -> Dict[str, Any]:
        """Start a RunPod training pod."""
        logger.info(f"Starting RunPod training pod: {self.config.pod_name}")
        
        # Get available GPUs
        available_gpus = await asyncio.to_thread(runpod.get_gpus)
        
        # Select GPU
        gpu_name = self._select_gpu(available_gpus)
        if not gpu_name:
            raise RuntimeError("No suitable GPU available")
            
        # Create pod using the simplified API
        # Note: The API seems to have changed - using the pattern from examples
        try:
            # Environment variables
            env_vars = {
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONPATH": "/workspace"
            }
            
            # Create pod with basic parameters
            # The create_pod function takes: name, image, gpu_type
            self.pod = await asyncio.to_thread(
                runpod.create_pod,
                self.config.pod_name,
                "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel",
                gpu_name
            )
            
            logger.info(f"Pod created: {self.pod['id']}")
            
            # Note: For advanced configuration (volumes, ports, etc.)
            # we may need to use the GraphQL API directly or runpodctl
            
            # Wait for pod to be ready
            await self._wait_for_pod_ready()
            
            return {
                "id": self.pod["id"],
                "name": self.config.pod_name,
                "gpu": gpu_name,
                "status": "running"
            }
            
        except Exception as e:
            logger.error(f"Failed to start pod: {e}")
            raise
            
    def _select_gpu(self, available_gpus: List[Dict]) -> Optional[str]:
        """Select best available GPU."""
        # Try preferred GPUs first
        for gpu_name in self.config.preferred_gpus:
            for gpu in available_gpus:
                if gpu.get("name") == gpu_name or gpu.get("displayName") == gpu_name:
                    if gpu.get("available", True):  # Check availability
                        logger.info(f"Selected preferred GPU: {gpu_name}")
                        return gpu_name
                    
        # Try fallback GPUs
        for gpu_name in self.config.fallback_gpus:
            for gpu in available_gpus:
                if gpu.get("name") == gpu_name or gpu.get("displayName") == gpu_name:
                    if gpu.get("available", True):
                        logger.info(f"Selected fallback GPU: {gpu_name}")
                        return gpu_name
                    
        # Try any available GPU
        for gpu in available_gpus:
            if gpu.get("available", True):
                gpu_name = gpu.get("name") or gpu.get("displayName")
                logger.warning(f"Using any available GPU: {gpu_name}")
                return gpu_name
                
        return None
        
    async def _wait_for_pod_ready(self, max_wait: int = 900) -> None:
        """Wait for pod to be ready."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait:
            # Get pod status
            pod = await asyncio.to_thread(runpod.get_pod, self.pod["id"])
            
            # Check if pod is running
            status = pod.get("status") or pod.get("desiredStatus")
            if status in ["RUNNING", "running"]:
                logger.info("Pod is ready!")
                self.pod = pod
                return
                    
            await asyncio.sleep(10)
            logger.info(f"Waiting for pod... Current status: {status}")
            
        raise TimeoutError("Pod failed to start within timeout")
        
    @retry(
        stop=stop_after_attempt(30),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _check_api_ready(self, api_base: str) -> bool:
        """Check if pod API is ready."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_base}/health", timeout=10)
                return response.status_code == 200
            except:
                return False
                
    async def upload_dataset(self) -> None:
        """Upload dataset to pod."""
        logger.info("Uploading dataset to pod...")
        
        # Create upload script
        upload_script = f"""
import os
import shutil
from pathlib import Path

# Copy dataset to pod
dataset_path = Path("{self.config.dataset_path}")
pod_dataset_path = Path("{self.config.output_mount_path}/data")
pod_dataset_path.mkdir(parents=True, exist_ok=True)

if dataset_path.is_file():
    shutil.copy2(dataset_path, pod_dataset_path / dataset_path.name)
else:
    shutil.copytree(dataset_path, pod_dataset_path / dataset_path.name)
    
print(f"Dataset uploaded to {{pod_dataset_path}}")
"""
        
        # Execute upload
        await self._execute_on_pod(upload_script)
        
    async def start_training(self) -> str:
        """Start training job on pod."""
        logger.info("Starting training job...")
        
        # Upload training script
        await self._upload_file(
            self.training_script,
            f"{self.config.output_mount_path}/train.py"
        )
        
        # Start training
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute training in background
        training_command = f"""
nohup python {self.config.output_mount_path}/train.py \
    --job_id {job_id} \
    > {self.config.output_mount_path}/training.log 2>&1 &
echo $! > {self.config.output_mount_path}/{job_id}.pid
"""
        
        await self._execute_on_pod(training_command)
        logger.info(f"Training started with job ID: {job_id}")
        
        return job_id
        
    async def monitor_training(self, job_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Monitor training progress."""
        log_path = f"{self.config.output_mount_path}/training_progress.json"
        
        while True:
            try:
                # Read progress file
                progress_data = await self._read_file(log_path)
                if progress_data:
                    progress = json.loads(progress_data)
                    yield progress
                    
                    if progress.get("status") == "completed":
                        break
                        
            except Exception as e:
                logger.warning(f"Error reading progress: {e}")
                
            await asyncio.sleep(self.config.monitor_interval)
            
    async def download_adapter(self) -> Path:
        """Download trained adapter from pod."""
        logger.info("Downloading adapter...")
        
        # Create local directory
        local_path = Path(self.config.output_dir) / "runpod_adapter"
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Download adapter files
        adapter_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model"
        ]
        
        for file_name in adapter_files:
            remote_path = f"{self.config.output_mount_path}/final_adapter/{file_name}"
            local_file = local_path / file_name
            
            try:
                content = await self._read_file(remote_path, binary=True)
                if content:
                    with open(local_file, 'wb') as f:
                        f.write(content)
                    logger.info(f"Downloaded {file_name}")
            except Exception as e:
                logger.warning(f"Failed to download {file_name}: {e}")
                
        return local_path
        
    async def stop_pod(self, terminate: bool = False) -> None:
        """Stop the training pod."""
        if self.pod:
            pod_id = self.pod.get("id") or self.pod.get("pod_id")
            logger.info(f"Stopping pod {pod_id}")
            
            try:
                if terminate:
                    await asyncio.to_thread(runpod.terminate_pod, pod_id)
                else:
                    await asyncio.to_thread(runpod.stop_pod, pod_id)
                    
                logger.info("Pod stopped successfully")
            except Exception as e:
                logger.error(f"Failed to stop pod: {e}")
                
    def _create_training_script(self) -> str:
        """Create the training script to run on pod."""
        return f'''
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Install requirements
os.system("pip install unsloth[colab-new] loguru")

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
from loguru import logger

# Configuration
config = {json.dumps(self.config.__dict__, default=str, indent=2)}

# Setup paths
output_dir = Path(config["output_mount_path"]) / "final_adapter"
output_dir.mkdir(parents=True, exist_ok=True)

# Progress tracking
progress_file = Path(config["output_mount_path"]) / "training_progress.json"

def update_progress(epoch, step, loss, status="training"):
    """Update training progress file."""
    progress = {{
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }}
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

try:
    # Load model
    logger.info(f"Loading model: {{config['model_name']}}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"]
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
        random_state=config["random_state"]
    )
    
    # Load dataset
    dataset_path = Path(config["output_mount_path"]) / "data" / Path(config["dataset_path"]).name
    logger.info(f"Loading dataset from: {{dataset_path}}")
    
    if dataset_path.suffix == ".jsonl":
        dataset = load_dataset("json", data_files=str(dataset_path))["train"]
    else:
        dataset = load_dataset(str(dataset_path))["train"]
        
    # Prepare dataset
    def formatting_func(examples):
        return [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
        
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
        save_steps=config["checkpoint_interval"],
        evaluation_strategy="no",
        save_strategy="steps",
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config["optim"],
        weight_decay=config["weight_decay"]
    )
    
    # Custom callback for progress tracking
    from transformers import TrainerCallback
    
    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                update_progress(
                    epoch=state.epoch or 0,
                    step=state.global_step,
                    loss=logs.get("loss", 0),
                    status="training"
                )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        formatting_func=formatting_func,
        args=training_args,
        callbacks=[ProgressCallback()]
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    update_progress(
        epoch=config["num_train_epochs"],
        step=trainer.state.global_step,
        loss=trainer.state.best_metric or 0,
        status="completed"
    )
    
    logger.info("Training completed successfully!")
    
except Exception as e:
    logger.error(f"Training failed: {{e}}")
    update_progress(0, 0, 0, status=f"failed: {{str(e)}}")
    raise
'''
        
    async def _execute_on_pod(self, command: str) -> str:
        """Execute command on pod."""
        # Note: RunPod doesn't have a direct exec API in the Python SDK
        # Options:
        # 1. Use SSH if enabled on the pod
        # 2. Use a web API endpoint on the pod
        # 3. Use runpodctl CLI wrapped in subprocess
        logger.warning("Direct command execution not available in current SDK")
        logger.info(f"Would execute: {command[:100]}...")
        return ""
        
    async def _upload_file(self, content: str, remote_path: str) -> None:
        """Upload file to pod."""
        # Note: File upload requires either:
        # 1. Using runpodctl CLI
        # 2. Setting up an HTTP endpoint on the pod
        # 3. Using network volumes
        logger.warning("Direct file upload not available in current SDK")
        logger.info(f"Would upload to: {remote_path}")
        
    async def _read_file(self, remote_path: str, binary: bool = False) -> Optional[bytes]:
        """Read file from pod."""
        # Note: File reading requires similar approach as upload
        logger.warning("Direct file read not available in current SDK")
        logger.info(f"Would read from: {remote_path}")
        return None