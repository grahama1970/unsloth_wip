#!/usr/bin/env python3
"""Example script for training with student-teacher thinking enhancement.

This script demonstrates how to:
1. Load Q&A data from ArangoDB
2. Enhance it with student-teacher iterative thinking
3. Train a LoRA adapter with the enhanced data
4. Monitor training progress with TensorBoard
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.unsloth.core.enhanced_config import EnhancedTrainingConfig
from src.unsloth.core.grokking_config import GrokkingConfig
from src.unsloth.data.thinking_enhancer import StudentTeacherConfig
from src.unsloth.training.enhanced_trainer import EnhancedUnslothTrainer
from src.unsloth.utils.memory import log_memory_usage
from loguru import logger


def main():
    """Run training with student-teacher enhancement."""
    
    # Configure student-teacher enhancement
    student_teacher_config = StudentTeacherConfig(
        # Models
        # Note: student_model will be automatically set to the model we're training!
        # This ensures the enhanced thinking captures model-specific reasoning patterns
        teacher_model="anthropic/max",  # Claude for pedagogical hints
        judge_model="gpt-4o-mini",  # For verifying correctness
        
        # Generation parameters
        max_iterations=3,  # Give student up to 3 attempts
        student_temperature=0.7,
        teacher_temperature=0.8,
        judge_temperature=0.0,
        
        # Processing
        batch_size=10,
        use_local_student=False,  # Use API for consistency
        max_new_tokens=500,
        
        # Output format
        thinking_format="iterative",  # Show clear iterations with "Aha!" moments
        save_iterations=True
    )
    
    # Configure training
    training_config = EnhancedTrainingConfig(
        # Model configuration
        model_name="unsloth/Phi-3.5-mini-instruct",
        max_seq_length=2048,
        
        # LoRA configuration
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Dataset configuration
        dataset_source="arangodb",
        dataset_path="/home/graham/workspace/experiments/arangodb/qa_output/qa_unsloth_latest.jsonl",
        max_samples=1000,  # Start with small dataset
        
        # Training configuration
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        
        # Optimization
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # Monitoring
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        
        # Output
        output_dir=f"outputs/student_teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_name="student_teacher_enhanced",
        
        # TensorBoard
        tensorboard_log_dir="outputs/tensorboard/student_teacher",
        report_to=["tensorboard"],
        
        # Optional: Enable grokking for better generalization
        grokking=GrokkingConfig(
            enable_grokking=False,  # Can enable for extended training
            grokking_multiplier=10.0,
            grokking_weight_decay=0.1
        )
    )
    
    # Initialize trainer
    trainer = EnhancedUnslothTrainer(
        config=training_config,
        student_teacher_config=student_teacher_config
    )
    
    try:
        # Log initial state
        log_memory_usage("Before training")
        
        # Train with thinking enhancement
        logger.info("Starting training with student-teacher thinking enhancement...")
        metrics = trainer.train(enhance_thinking=True)
        
        # Log results
        logger.info(f"Training completed! Metrics: {metrics}")
        
        # Save the model
        save_path = Path(training_config.output_dir) / "final_model"
        trainer.model.save_pretrained(save_path)
        trainer.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to: {save_path}")
        
        # Generate model card
        model_card = f"""# Student-Teacher Enhanced Model

This model was trained using student-teacher iterative thinking enhancement.

## Training Configuration
- Base Model: {training_config.model_name}
- Student Model: {training_config.model_name} (same as base - captures model-specific patterns)
- Teacher Model: {student_teacher_config.teacher_model}
- Max Iterations: {student_teacher_config.max_iterations}
- Dataset: {training_config.dataset_path}
- Samples: {training_config.max_samples}

## Key Features
- Student model attempts to solve problems independently
- When incorrect, receives pedagogical hints from teacher (Claude)
- Shows iterative reasoning with "Aha!" moments
- Creates richer training data with self-correction patterns

## Training Metrics
{metrics}

## Usage
This model is particularly good at:
- Showing step-by-step reasoning
- Self-correcting when making mistakes
- Incorporating hints to improve answers
"""
        
        with open(save_path / "README.md", "w") as f:
            f.write(model_card)
            
    finally:
        # Cleanup
        trainer.cleanup()
        log_memory_usage("After cleanup")


if __name__ == "__main__":
    main()