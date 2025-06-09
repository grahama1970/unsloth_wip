"""
Module: complete_training_pipeline.py

External Dependencies:
- asyncio: [Documentation URL]
- loguru: [Documentation URL]
- src: [Documentation URL]

Sample Input:
>>> # Add specific examples based on module functionality

Expected Output:
>>> # Add expected output examples

Example Usage:
>>> # Add usage examples
"""

#!/usr/bin/env python3
"""Complete training pipeline integrating all components.

This script orchestrates the entire training process:
1. Load Q&A data from ArangoDB
2. Enhance with student-teacher thinking (using Claude as teacher)
3. Train LoRA adapter (locally or on RunPod for large models)
4. Upload to Hugging Face
5. Validate the trained model
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger

from src.unsloth.core.enhanced_config import EnhancedTrainingConfig
from src.unsloth.core.grokking_config import GrokkingConfig
from src.unsloth.data.thinking_enhancer import StudentTeacherConfig, ThinkingEnhancer
from src.unsloth.training.enhanced_trainer import EnhancedUnslothTrainer
from src.unsloth.training.runpod_training_ops import run_training_on_runpod
from src.unsloth.upload.hub_uploader import HubUploader
from src.unsloth.utils.memory import get_gpu_memory
from src.unsloth.validation.model_validator import ModelValidator


class CompletePipeline:
    """Orchestrates the complete training pipeline."""

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        hub_model_id: str | None = None,
        use_runpod: bool = False
    ):
        """Initialize the pipeline."""
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.hub_model_id = hub_model_id
        self.use_runpod = use_runpod

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="500 MB")

    def _determine_training_location(self) -> bool:
        """Determine if we should use RunPod based on model size and GPU."""
        model_lower = self.model_name.lower()

        # Check model size
        if "70b" in model_lower:
            logger.info("70B model detected - RunPod recommended")
            return True
        elif "30b" in model_lower or "33b" in model_lower:
            logger.info("30B model detected - checking GPU memory")
            gpu_memory = get_gpu_memory()
            if gpu_memory < 80:  # Less than 80GB
                logger.info(f"GPU memory {gpu_memory}GB insufficient for 30B model")
                return True
        elif "13b" in model_lower:
            logger.info("13B model detected - checking GPU memory")
            gpu_memory = get_gpu_memory()
            if gpu_memory < 40:  # Less than 40GB
                logger.info(f"GPU memory {gpu_memory}GB insufficient for 13B model")
                return True

        return self.use_runpod

    def create_configs(self) -> tuple:
        """Create configuration objects for training."""
        # Student-teacher config
        student_teacher_config = StudentTeacherConfig(
            # student_model automatically set to training model
            teacher_model="anthropic/max",  # Claude for hints
            judge_model="gpt-4o-mini",
            max_iterations=3,
            student_temperature=0.7,
            teacher_temperature=0.8,
            batch_size=10,
            use_local_student=False,  # Use API for flexibility
            thinking_format="iterative",
            save_iterations=True
        )

        # Determine if we need RunPod
        use_runpod = self._determine_training_location()

        # Base training config
        base_config = {
            "model_name": self.model_name,
            "max_seq_length": 2048,

            # LoRA settings
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],

            # Dataset
            "dataset_source": "arangodb",
            "dataset_path": self.dataset_path,
            "validation_split": 0.1,

            # Training
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "max_grad_norm": 0.3,

            # Memory optimization
            "gradient_checkpointing": True,
            "group_by_length": True,

            # Monitoring
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 100,
            "save_total_limit": 3,

            # Output
            "output_dir": str(self.output_dir / "checkpoints"),
            "run_name": f"{self.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

            # Reporting
            "report_to": ["tensorboard"],
            "tensorboard_log_dir": str(self.output_dir / "tensorboard"),

            # Optional grokking
            "grokking": GrokkingConfig(
                enable_grokking=False,
                grokking_multiplier=10.0,
                grokking_weight_decay=0.1
            )
        }

        # Always use EnhancedTrainingConfig
        training_config = EnhancedTrainingConfig(**base_config)

        return student_teacher_config, training_config, use_runpod

    async def run_pipeline(self) -> dict[str, Any]:
        """Run the complete training pipeline."""
        results = {
            "model_name": self.model_name,
            "start_time": datetime.now().isoformat(),
            "steps": {}
        }

        try:
            # Step 1: Create configurations
            logger.info("Step 1: Creating configurations...")
            student_teacher_config, training_config, use_runpod = self.create_configs()
            results["steps"]["config"] = "completed"
            results["use_runpod"] = use_runpod

            # Step 2: Enhance dataset with student-teacher thinking
            logger.info("Step 2: Enhancing dataset with student-teacher thinking...")
            enhancer = ThinkingEnhancer(
                config=student_teacher_config,
                base_model_name=self.model_name
            )

            enhanced_path = self.output_dir / "enhanced_dataset.jsonl"
            enhancement_stats = await enhancer.enhance_dataset(
                input_path=Path(self.dataset_path),
                output_path=enhanced_path,
                max_samples=training_config.max_samples
            )

            results["steps"]["enhancement"] = enhancement_stats
            logger.info(f"Enhancement stats: {enhancement_stats}")

            # Update dataset path to enhanced version
            training_config.dataset_path = str(enhanced_path)

            # Step 3: Train model
            if use_runpod:
                logger.info("Step 3: Training on RunPod...")
                results["steps"]["training"] = await self._train_on_runpod(training_config)
            else:
                logger.info("Step 3: Training locally...")
                results["steps"]["training"] = await self._train_locally(
                    training_config,
                    student_teacher_config
                )

            # Step 4: Upload to Hugging Face (if configured)
            if self.hub_model_id:
                logger.info("Step 4: Uploading to Hugging Face...")
                uploader = HubUploader()
                upload_result = await uploader.upload_adapter(
                    adapter_path=Path(results["steps"]["training"]["adapter_path"]),
                    model_id=self.hub_model_id,
                    base_model=self.model_name,
                    training_stats=results
                )
                results["steps"]["upload"] = upload_result

            # Step 5: Validate model
            logger.info("Step 5: Validating trained model...")
            validator = ModelValidator()
            validation_results = await validator.validate_adapter(
                adapter_path=Path(results["steps"]["training"]["adapter_path"]),
                base_model=self.model_name,
                test_prompts=[
                    "What is the capital of France?",
                    "Explain quantum computing in simple terms.",
                    "Write a Python function to calculate fibonacci numbers."
                ]
            )
            results["steps"]["validation"] = validation_results

            # Complete
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"

            # Save results
            import json
            with open(self.output_dir / "pipeline_results.json", "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Pipeline completed successfully! Results saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        return results

    async def _train_locally(
        self,
        training_config: EnhancedTrainingConfig,
        student_teacher_config: StudentTeacherConfig
    ) -> dict[str, Any]:
        """Train model locally."""
        trainer = EnhancedUnslothTrainer(
            config=training_config,
            student_teacher_config=student_teacher_config
        )

        # Train (dataset already enhanced)
        metrics = trainer.train(enhance_thinking=False)

        # Save adapter
        adapter_path = self.output_dir / "adapter"
        trainer.model.save_pretrained(adapter_path)
        trainer.tokenizer.save_pretrained(adapter_path)

        # Cleanup
        trainer.cleanup()

        return {
            "metrics": metrics,
            "adapter_path": str(adapter_path),
            "training_location": "local"
        }

    async def _train_on_runpod(self, config: EnhancedTrainingConfig) -> dict[str, Any]:
        """Train model on RunPod."""
        # Convert to dict for RunPod
        training_config = {
            "model_name": config.model_name,
            "max_seq_length": config.max_seq_length,
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "num_train_epochs": config.num_train_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "optim": config.optim,
            "weight_decay": config.weight_decay,
            "gradient_checkpointing": config.gradient_checkpointing,
            "hub_model_id": self.hub_model_id
        }

        # Run training on RunPod
        result = await run_training_on_runpod(
            model_name=config.model_name,
            dataset_path=Path(config.dataset_path),
            training_config=training_config,
            hub_model_id=self.hub_model_id
        )

        if result["status"] == "success":
            return {
                "metrics": {"status": "completed"},
                "adapter_path": result["adapter_path"],
                "training_location": "runpod",
                "pod_id": result["pod_id"]
            }
        else:
            raise RuntimeError(f"RunPod training failed: {result.get('error')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Complete Unsloth training pipeline with student-teacher enhancement"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name (e.g., unsloth/Phi-3.5-mini-instruct)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (JSONL file from ArangoDB)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/pipeline",
        help="Output directory"
    )

    parser.add_argument(
        "--hub-id",
        type=str,
        help="Hugging Face model ID for upload (e.g., username/model-name)"
    )

    parser.add_argument(
        "--force-runpod",
        action="store_true",
        help="Force training on RunPod even for smaller models"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = CompletePipeline(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        hub_model_id=args.hub_id,
        use_runpod=args.force_runpod
    )

    # Run pipeline
    results = asyncio.run(pipeline.run_pipeline())

    # Print summary
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    print(f"Status: {results['status']}")
    print(f"Model: {results['model_name']}")
    print(f"Training Location: {'RunPod' if results.get('use_runpod') else 'Local'}")

    if "enhancement" in results["steps"]:
        stats = results["steps"]["enhancement"]
        print("\nEnhancement:")
        print(f"  - Examples: {stats.get('enhanced_examples', 0)}")
        print(f"  - Avg Iterations: {stats.get('average_iterations', 0):.2f}")
        print(f"  - Convergence: {stats.get('convergence_rate', 0)*100:.1f}%")

    if "training" in results["steps"]:
        training = results["steps"]["training"]
        print("\nTraining:")
        print(f"  - Final Loss: {training.get('metrics', {}).get('loss', 'N/A')}")
        print(f"  - Adapter Path: {training.get('adapter_path', 'N/A')}")

    if "upload" in results["steps"]:
        print("\nUpload:")
        print(f"  - Hub URL: {results['steps']['upload'].get('url', 'N/A')}")

    print("\n" + "="*50)


if __name__ == "__main__":
    main()
