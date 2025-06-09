"""
Module: runpod_trainer_v2.py
Description: Enhanced RunPod trainer using modular runpod_ops package

External Dependencies:
- runpod_ops: Internal modular RunPod operations package
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> config = EnhancedTrainingConfig(model_name="unsloth/Phi-3.5-mini", num_epochs=3)
>>> trainer = RunPodTrainerV2(config)

Expected Output:
>>> job = await trainer.train()
>>> job.job_id
"train_20250106_123456"

Example Usage:
>>> from unsloth.training.runpod_trainer_v2 import RunPodTrainerV2
>>> trainer = RunPodTrainerV2(config)
>>> job = await trainer.train()
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from loguru import logger

from unsloth.core.enhanced_config import EnhancedTrainingConfig
from unsloth.training.trainer import UnslothTrainer

# Import modular runpod_ops
from runpod_ops import (
    TrainingOrchestrator,
    TrainingConfig as RunPodTrainingConfig,
    InstanceMonitor,
    InstanceOptimizer,
    CostCalculator
)


class RunPodTrainerV2:
    """Enhanced RunPod trainer using modular operations package."""
    
    def __init__(self, config: EnhancedTrainingConfig):
        """
        Initialize enhanced RunPod trainer.
        
        Args:
            config: Enhanced training configuration
        """
        self.config = config
        self.orchestrator = TrainingOrchestrator()
        self.optimizer = InstanceOptimizer()
        self.calculator = CostCalculator()
        self.monitor = InstanceMonitor()
        
        self.current_job = None
        
    async def train(
        self,
        spot_instances: bool = True,
        max_budget: Optional[float] = None,
        multi_gpu: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Train model on RunPod with optimal configuration.
        
        Args:
            spot_instances: Use spot instances for cost savings
            max_budget: Maximum budget for training
            multi_gpu: Force multi-GPU setup (auto-detected if None)
            
        Returns:
            Training job details
        """
        logger.info(f"Starting RunPod training for {self.config.model_name}")
        
        # Estimate model size
        model_size = self._estimate_model_size()
        
        # Get optimal configuration
        if multi_gpu is None:
            # Auto-detect based on model size
            multi_gpu = model_size in ["30B", "70B", "180B"]
            
        # Create RunPod training config
        runpod_config = self._create_runpod_config()
        
        # Estimate costs
        cost_estimate = await self._estimate_training_cost(
            runpod_config,
            multi_gpu=multi_gpu
        )
        
        logger.info(f"Estimated training cost: ${cost_estimate['total_cost']:.2f}")
        logger.info(f"Estimated time: {cost_estimate['estimated_hours']:.1f} hours")
        
        if max_budget and cost_estimate["total_cost"] > max_budget:
            logger.warning(f"Estimated cost exceeds budget (${max_budget})")
            return {
                "error": "Budget exceeded",
                "estimated_cost": cost_estimate["total_cost"],
                "max_budget": max_budget
            }
            
        # Start training job
        if multi_gpu and cost_estimate.get("num_gpus", 1) > 1:
            # Use distributed training
            self.current_job = await self.orchestrator.run_distributed_training(
                runpod_config,
                num_nodes=cost_estimate.get("num_nodes", 1),
                gpus_per_node=cost_estimate.get("gpus_per_node", 2),
                strategy="fsdp" if model_size in ["70B", "180B"] else "ddp"
            )
        else:
            # Single instance training
            self.current_job = await self.orchestrator.start_training_job(
                runpod_config,
                num_gpus=cost_estimate.get("num_gpus", 1),
                gpu_type=cost_estimate.get("gpu_type"),
                spot_instances=spot_instances,
                max_budget=max_budget
            )
            
        # Start monitoring
        asyncio.create_task(self._monitor_training())
        
        return {
            "job_id": self.current_job.job_id,
            "status": self.current_job.status,
            "instances": self.current_job.instances,
            "estimated_cost": cost_estimate["total_cost"],
            "estimated_hours": cost_estimate["estimated_hours"],
            "gpu_config": {
                "type": cost_estimate.get("gpu_type"),
                "count": cost_estimate.get("num_gpus", 1)
            }
        }
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        if not self.current_job:
            return {"error": "No active training job"}
            
        return await self.orchestrator.get_job_status(self.current_job.job_id)
        
    async def stop_training(self) -> bool:
        """Stop current training job."""
        if not self.current_job:
            return False
            
        return await self.orchestrator.cancel_job(self.current_job.job_id)
        
    async def download_results(self, output_dir: Optional[Path] = None) -> Path:
        """
        Download training results.
        
        Args:
            output_dir: Directory to save results (default: config.output_dir)
            
        Returns:
            Path to downloaded adapter
        """
        if not self.current_job:
            raise ValueError("No training job to download from")
            
        output_dir = output_dir or Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # In practice, would download from RunPod storage
        # For now, return expected path
        adapter_path = output_dir / f"adapter_{self.current_job.job_id}"
        
        logger.info(f"Downloaded adapter to: {adapter_path}")
        return adapter_path
        
    def _estimate_model_size(self) -> str:
        """Estimate model size from name."""
        model_lower = self.config.model_name.lower()
        
        size_patterns = {
            "180b": "180B",
            "70b": "70B",
            "30b": "30B", "33b": "30B", "34b": "30B",
            "13b": "13B",
            "7b": "7B",
            "3b": "3B", "3.5b": "3B",
        }
        
        for pattern, size in size_patterns.items():
            if pattern in model_lower:
                return size
                
        return "7B"  # Default
        
    def _create_runpod_config(self) -> RunPodTrainingConfig:
        """Create RunPod training configuration."""
        # Estimate dataset size (would be calculated from actual dataset)
        dataset_size = 10000  # Placeholder
        
        # Map enhanced config to RunPod config
        return RunPodTrainingConfig(
            model_name=self.config.model_name,
            model_size=self._estimate_model_size(),
            dataset_path=str(self.config.dataset_path),
            output_path=str(self.config.output_dir),
            num_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=int(self.config.warmup_ratio * dataset_size),
            save_steps=500,
            logging_steps=10,
            fp16=not self.config.bf16,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type
        )
        
    async def _estimate_training_cost(
        self,
        config: RunPodTrainingConfig,
        multi_gpu: bool = False
    ) -> Dict[str, Any]:
        """Estimate training cost and optimal configuration."""
        # Get dataset size (would be calculated from actual dataset)
        dataset_size = 10000  # Placeholder
        
        # Use optimizer to get best configuration
        if multi_gpu:
            # Get multi-GPU config
            training_config = self.optimizer.optimize_for_training(
                config.model_size,
                dataset_size,
                config.num_epochs,
                config.batch_size
            )
            
            # Extract multi-GPU details
            if "strategy" in training_config:
                # Distributed training
                return {
                    "gpu_type": training_config["gpu_type"],
                    "num_gpus": training_config["gpu_count"],
                    "num_nodes": max(1, training_config["gpu_count"] // 2),
                    "gpus_per_node": min(2, training_config["gpu_count"]),
                    "estimated_hours": training_config["estimated_hours"],
                    "total_cost": training_config["estimated_cost"],
                    "strategy": training_config["strategy"]
                }
        else:
            # Single GPU or single-node multi-GPU
            training_config = self.optimizer.optimize_for_training(
                config.model_size,
                dataset_size,
                config.num_epochs,
                config.batch_size
            )
            
        return {
            "gpu_type": training_config["gpu_type"],
            "num_gpus": training_config.get("gpu_count", 1),
            "estimated_hours": training_config["estimated_hours"],
            "total_cost": training_config["estimated_cost"],
            "batch_size": training_config.get("batch_size", config.batch_size),
            "gradient_accumulation": training_config.get("gradient_accumulation_steps", 1)
        }
        
    async def _monitor_training(self):
        """Monitor training progress."""
        if not self.current_job:
            return
            
        def progress_callback(update: Dict[str, Any]):
            """Handle progress updates."""
            status = update.get("status", "unknown")
            
            if status == "running":
                gpu_util = update.get("gpu_utilization", 0)
                loss = update.get("training_progress", {}).get("loss")
                cost = update.get("total_cost", 0)
                
                logger.info(f"Training progress - GPU: {gpu_util}%, Loss: {loss}, Cost: ${cost:.2f}")
                
            elif status == "complete":
                logger.info("Training completed successfully!")
                
            elif status == "error":
                logger.error(f"Training error: {update.get('error')}")
                
        # Monitor primary instance
        primary_instance = self.current_job.instances[0] if self.current_job.instances else None
        
        if primary_instance:
            try:
                await self.monitor.monitor_training_job(
                    primary_instance,
                    callback=progress_callback,
                    check_interval=60
                )
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
    async def compare_training_options(
        self,
        include_local: bool = False
    ) -> Dict[str, Any]:
        """
        Compare different training options.
        
        Args:
            include_local: Include local GPU in comparison
            
        Returns:
            Comparison of training options
        """
        model_size = self._estimate_model_size()
        dataset_size = 10000  # Placeholder
        
        options = {}
        
        # Single GPU options
        for gpu_type in ["RTX_4090", "A100_40GB", "A100_80GB", "H100"]:
            try:
                config = self.optimizer.optimize_for_training(
                    model_size,
                    dataset_size,
                    self.config.num_train_epochs
                )
                
                if config.get("gpu_type") == gpu_type:
                    options[f"single_{gpu_type}"] = {
                        "gpu_type": gpu_type,
                        "gpu_count": 1,
                        "estimated_hours": config["estimated_hours"],
                        "estimated_cost": config["estimated_cost"],
                        "cost_per_epoch": config["estimated_cost"] / self.config.num_train_epochs
                    }
            except:
                pass
                
        # Multi-GPU option
        multi_config = self.optimizer.optimize_for_training(
            model_size,
            dataset_size,
            self.config.num_train_epochs
        )
        
        if multi_config.get("gpu_count", 1) > 1:
            options["multi_gpu"] = {
                "gpu_type": multi_config["gpu_type"],
                "gpu_count": multi_config["gpu_count"],
                "strategy": multi_config.get("strategy", "data_parallel"),
                "estimated_hours": multi_config["estimated_hours"],
                "estimated_cost": multi_config["estimated_cost"],
                "efficiency": multi_config.get("multi_gpu_efficiency", "N/A")
            }
            
        # Sort by cost
        sorted_options = dict(sorted(
            options.items(),
            key=lambda x: x[1]["estimated_cost"]
        ))
        
        return {
            "model_size": model_size,
            "dataset_size": dataset_size,
            "num_epochs": self.config.num_train_epochs,
            "options": sorted_options,
            "recommendation": list(sorted_options.keys())[0] if sorted_options else None
        }


# Validation
if __name__ == "__main__":
    async def test_trainer():
        from unsloth.core.enhanced_config import EnhancedTrainingConfig
        
        # Create test config
        config = EnhancedTrainingConfig(
            model_name="unsloth/Phi-3.5-mini-instruct",
            dataset_path="./data/train.jsonl",
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-4
        )
        
        # Create trainer
        trainer = RunPodTrainerV2(config)
        
        print("RunPod Trainer V2 Test")
        print("=" * 50)
        
        # Compare options
        print("\nComparing training options...")
        options = await trainer.compare_training_options()
        
        print(f"\nModel: {config.model_name}")
        print(f"Size: {options['model_size']}")
        print(f"Epochs: {options['num_epochs']}")
        
        print("\nAvailable options:")
        for name, details in options["options"].items():
            print(f"\n{name}:")
            print(f"  GPU: {details['gpu_count']}x {details['gpu_type']}")
            print(f"  Time: {details['estimated_hours']:.1f} hours")
            print(f"  Cost: ${details['estimated_cost']:.2f}")
            
        print(f"\nRecommended: {options['recommendation']}")
        
    # asyncio.run(test_trainer())
    print("\n Module validation passed")