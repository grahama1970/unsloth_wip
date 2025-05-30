"""Example script for training on ArangoDB Q&A data."""

from pathlib import Path
from unsloth.core.config import TrainingConfig
from unsloth.training.trainer import UnslothTrainer
from unsloth.utils.logging import setup_logging


def main():
    # Setup logging
    setup_logging(log_file=Path("training.log"))
    
    # Configure training
    config = TrainingConfig(
        # Model settings
        model_name="unsloth/Phi-3.5-mini-instruct",
        max_seq_length=2048,
        
        # Dataset settings
        dataset_source="arangodb",
        dataset_path="/home/graham/workspace/experiments/arangodb/qa_output",
        validation_split=0.1,
        
        # Filter for high-quality Q&A pairs
        metadata_filters={
            "confidence": {"min": 0.9},
            "validated": True,
            "question_type": ["FACTUAL", "COMPARATIVE", "PROCEDURAL"]
        },
        
        # LoRA configuration
        lora_r=16,
        lora_alpha=32,
        
        # Training settings
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        
        # Output
        output_dir="./outputs/arangodb_qa_model",
        run_name="phi35-arangodb-qa"
    )
    
    # Initialize trainer
    trainer = UnslothTrainer(config)
    
    # Train
    result = trainer.train()
    
    print(f"Training completed!")
    print(f"Adapter saved to: {result.adapter_path}")
    print(f"Training time: {result.training_time:.2f} seconds")
    
    if result.metrics:
        print("\nFinal metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
    
    # Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()