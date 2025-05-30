"""Example of training with grokking technique for superior generalization.

Grokking involves extended training (30×+ epochs) with strong regularization,
leading to initial memorization followed by sudden generalization.

This is particularly useful for:
- Complex reasoning tasks
- Small datasets where deep understanding is crucial
- Achieving state-of-the-art performance with patience
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.unsloth.core.enhanced_config import EnhancedTrainingConfig
from src.unsloth.core.grokking_config import GrokkingConfig
from src.unsloth.training.enhanced_trainer import EnhancedUnslothTrainer
from src.unsloth.utils.logging import setup_logging


def visualize_grokking_phases(log_dir: Path):
    """Create visualization of grokking phases from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Load TensorBoard logs
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        # Extract train and eval losses
        train_loss = [(s.step, s.value) for s in ea.Scalars('training/loss')]
        eval_loss = [(s.step, s.value) for s in ea.Scalars('eval/eval_loss')]
        
        if train_loss and eval_loss:
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot losses
            train_steps, train_values = zip(*train_loss)
            eval_steps, eval_values = zip(*eval_loss)
            
            ax.plot(train_steps, train_values, label='Training Loss', alpha=0.8)
            ax.plot(eval_steps, eval_values, label='Validation Loss', alpha=0.8)
            
            # Mark phases (if available)
            ax.axvspan(0, 1000, alpha=0.1, color='gray', label='Initialization')
            ax.axvspan(1000, 5000, alpha=0.1, color='red', label='Memorization')
            ax.axvspan(5000, 10000, alpha=0.1, color='green', label='Grokking')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.set_title('Grokking Training Phases')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            output_path = log_dir.parent / "grokking_phases.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved grokking visualization to {output_path}")
            
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


def main():
    """Demonstrate grokking-based training."""
    setup_logging(log_file=Path("logs/grokking_training.log"))
    
    print("🧠 Grokking Training Example")
    print("=" * 50)
    print("This will train for MANY epochs (90 instead of 3)")
    print("Initial performance may seem poor, but patience leads to superior results!")
    print("=" * 50)
    
    # Create grokking configuration
    grokking_config = GrokkingConfig(
        enable_grokking=True,
        grokking_multiplier=30.0,  # 3 epochs × 30 = 90 epochs
        
        # Strong weight decay is crucial
        grokking_weight_decay=0.1,
        grokking_weight_decay_schedule="cosine",
        
        # Learning rate schedule for extended training
        grokking_lr_schedule="cosine_with_restarts",
        grokking_lr_min=1e-6,
        
        # Additional regularization
        grokking_label_smoothing=0.1,
        grokking_gradient_noise=0.01,
        grokking_dropout_schedule="constant",
        
        # Monitoring
        track_memorization=True,
        track_validation_phases=True,
        save_memorization_checkpoint=True,
        
        # Disable early stopping - grokking needs patience!
        disable_early_stopping=True,
        
        # Advanced techniques
        use_grokfast=False,  # Can enable for 50× speedup
        use_layer_swapping=False  # Experimental
    )
    
    # Create training configuration
    config = EnhancedTrainingConfig(
        # Model
        model_name="unsloth/Phi-3.5-mini-instruct",
        max_seq_length=2048,
        
        # Dataset - use high-quality filtered data for grokking
        dataset_source="arangodb",
        dataset_path=os.getenv("ARANGODB_QA_PATH", "/home/graham/workspace/experiments/arangodb/qa_output"),
        metadata_filters={
            "confidence": {"min": 0.9},  # Only highest quality
            "validated": True,
            "question_type": ["FACTUAL", "PROCEDURAL"]  # Complex reasoning
        },
        max_samples=5000,  # Smaller dataset for grokking demo
        
        # LoRA settings optimized for grokking
        lora_r=32,  # Higher rank for complex patterns
        lora_alpha=64,  # 2×rank
        lora_dropout=0.1,  # Higher dropout for grokking
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"
        ],
        
        # Base training settings (will be modified by grokking)
        num_train_epochs=3,  # Will become 90 with grokking
        per_device_train_batch_size=4,  # Larger batch for stability
        gradient_accumulation_steps=2,
        learning_rate=5e-4,  # Higher initial LR
        warmup_ratio=0.02,  # Shorter warmup for extended training
        
        # Evaluation
        eval_steps=50,  # More frequent for phase monitoring
        save_steps=1000,
        logging_steps=10,
        
        # Output
        output_dir="./outputs/grokking_model",
        tensorboard_log_dir="./outputs/tensorboard/grokking_run",
        run_name="phi35-grokking-demo",
        
        # Enable grokking
        grokking=grokking_config,
        use_grokking=True
    )
    
    print("\n📋 Grokking Configuration:")
    print(f"   - Original epochs: 3")
    print(f"   - Grokking epochs: {config.num_train_epochs}")
    print(f"   - Weight decay: {config.weight_decay}")
    print(f"   - Regularization: Label smoothing + gradient noise")
    print(f"   - Phase tracking: Enabled")
    
    print("\n⚠️  Expected behavior:")
    print("   1. Initial phase: Quick drop in training loss")
    print("   2. Memorization: Training loss << validation loss")
    print("   3. Long plateau: May last thousands of steps")
    print("   4. Grokking: Sudden drop in validation loss")
    print("   5. Convergence: Both losses stabilize at low values")
    
    print("\n🚀 Starting grokking training...")
    print(f"   Monitor progress: tensorboard --logdir {config.tensorboard_log_dir}")
    
    # Initialize trainer
    trainer = EnhancedUnslothTrainer(config)
    
    try:
        # Train with grokking
        results = trainer.train()
        
        print("\n✅ Grokking training completed!")
        print(f"   - Total training time: {results['training_time'] / 3600:.2f} hours")
        print(f"   - Final train loss: {results['train_result'].get('train_loss', 'N/A')}")
        print(f"   - Final eval loss: {results['train_result'].get('eval_loss', 'N/A')}")
        
        # Create visualization
        visualize_grokking_phases(Path(config.tensorboard_log_dir))
        
        print("\n🎯 Grokking Benefits:")
        print("   - Superior generalization compared to early stopping")
        print("   - Better handling of edge cases")
        print("   - More robust to distribution shifts")
        print("   - Deeper understanding of patterns")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted - this is common with grokking!")
        print("   Grokking requires patience. Consider:")
        print("   - Running overnight or on dedicated hardware")
        print("   - Using Grokfast for 50× speedup")
        print("   - Monitoring phases to estimate completion")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
        
    finally:
        trainer.cleanup()
        
    print("\n📚 Further Reading:")
    print("   - Original paper: 'Grokking: Generalization Beyond Overfitting'")
    print("   - Recent advances: docs/correspondence/grokking_technique.md")
    print("   - TensorBoard logs: View phases and transitions")


if __name__ == "__main__":
    main()