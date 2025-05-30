"""Full pipeline example: evaluation set creation, training with monitoring, and analysis.

This example demonstrates:
1. Creating an evaluation set with rephrased questions
2. Training with enhanced configuration and TensorBoard monitoring
3. Evaluating the model on the custom evaluation set
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.unsloth.core.enhanced_config import EnhancedTrainingConfig, EvaluationConfig
from src.unsloth.training.enhanced_trainer import EnhancedUnslothTrainer
from src.unsloth.data.evaluation import EvaluationSetGenerator
from src.unsloth.utils.logging import setup_logging
from src.unsloth.inference.generate import InferenceEngine, GenerationConfig


def main():
    """Run the full pipeline."""
    # Setup logging
    setup_logging(log_file=Path("logs/full_pipeline.log"), level="INFO")
    
    print("🚀 Starting full Unsloth pipeline example")
    
    # ========================================
    # Step 1: Create Evaluation Set
    # ========================================
    print("\n📊 Step 1: Creating evaluation set with rephrased questions...")
    
    eval_config = EvaluationConfig(
        eval_set_size=0.05,  # 5% of data
        rephrase_model="gpt-4o-mini",
        rephrase_temperature=0.7,
        min_eval_samples=100,
        max_eval_samples=1000
    )
    
    # Create evaluation set
    eval_generator = EvaluationSetGenerator(eval_config)
    source_path = Path(os.getenv("ARANGODB_QA_PATH", "/home/graham/workspace/experiments/arangodb/qa_output"))
    eval_output_path = Path("./data/evaluation_sets")
    
    try:
        eval_file, eval_stats = eval_generator.create_evaluation_set(
            source_path,
            eval_output_path
        )
        print(f"✅ Evaluation set created: {eval_file}")
        print(f"   - Total examples: {eval_stats['total_examples']}")
        print(f"   - Evaluation examples: {eval_stats['evaluation_examples']}")
        print(f"   - Rephrasing success: {eval_stats['rephrasing_success_rate']:.1f}%")
    except Exception as e:
        print(f"⚠️  Warning: Could not create evaluation set: {e}")
        print("   Continuing with standard validation split...")
        eval_file = None
    
    # ========================================
    # Step 2: Configure Enhanced Training
    # ========================================
    print("\n⚙️  Step 2: Configuring enhanced training...")
    
    config = EnhancedTrainingConfig(
        # Model settings
        model_name="unsloth/Phi-3.5-mini-instruct",
        max_seq_length=2048,
        
        # Dataset settings
        dataset_source="arangodb",
        dataset_path=source_path,
        validation_split=0.1,  # Will use eval set if available
        
        # Enhanced LoRA settings (based on best practices)
        lora_r=16,  # Higher rank for better capacity
        lora_alpha=32,  # 2x rank as recommended
        lora_dropout=0.05,  # Small dropout for regularization
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"  # Include lm_head for better performance
        ],
        
        # Filter for high-quality data
        metadata_filters={
            "confidence": {"min": 0.85},
            "validated": True,
            "question_type": ["FACTUAL", "COMPARATIVE", "PROCEDURAL"]
        },
        
        # Training settings optimized for quality
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        learning_rate=2e-4,  # Higher LR for LoRA
        warmup_ratio=0.05,
        
        # Enhanced features
        neftune_noise_alpha=5,  # NEFTune for better convergence
        group_by_length=True,  # Efficient batching
        log_gradient_norm=True,
        
        # Early stopping
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
        
        # More frequent evaluation
        eval_steps=100,
        save_steps=500,
        logging_steps=10,
        
        # Output configuration
        output_dir="./outputs/enhanced_model",
        tensorboard_log_dir="./outputs/tensorboard/enhanced_run",
        run_name="phi35-arangodb-enhanced"
    )
    
    print(f"📋 Configuration:")
    print(f"   - Model: {config.model_name}")
    print(f"   - LoRA rank: {config.lora_r} (alpha: {config.lora_alpha})")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"   - TensorBoard: {config.tensorboard_log_dir}")
    
    # ========================================
    # Step 3: Train with Enhanced Monitoring
    # ========================================
    print("\n🎓 Step 3: Training with enhanced monitoring...")
    print(f"   💡 Monitor training in TensorBoard:")
    print(f"      tensorboard --logdir {config.tensorboard_log_dir}")
    
    # Initialize trainer
    trainer = EnhancedUnslothTrainer(config)
    
    # Train
    try:
        results = trainer.train()
        
        print("\n✅ Training completed successfully!")
        print(f"   - Training time: {results['training_time']:.2f} seconds")
        print(f"   - Final loss: {results['train_result'].get('train_loss', 'N/A')}")
        print(f"   - Adapter saved to: {results['adapter_path']}")
        
        # Log key metrics
        if 'eval_loss' in results['train_result']:
            print(f"   - Eval loss: {results['train_result']['eval_loss']:.4f}")
        if 'eval_perplexity' in results['train_result']:
            print(f"   - Eval perplexity: {results['train_result']['eval_perplexity']:.2f}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        trainer.cleanup()
    
    # ========================================
    # Step 4: Test Inference
    # ========================================
    print("\n🔍 Step 4: Testing inference with trained model...")
    
    # Initialize inference engine
    engine = InferenceEngine(
        model_path=results['adapter_path'],
        max_seq_length=2048
    )
    
    # Test prompts
    test_prompts = [
        "What is ArangoDB and what makes it unique?",
        "How does ArangoDB handle graph queries?",
        "What are the main components of ArangoDB architecture?",
    ]
    
    # Generate responses
    gen_config = GenerationConfig(
        temperature=0.7,
        max_new_tokens=200,
        top_p=0.9
    )
    
    print("\n📝 Sample generations:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Question: {prompt}")
        response = engine.generate(prompt, gen_config)
        print(f"   Answer: {response}")
    
    # ========================================
    # Step 5: Summary and Next Steps
    # ========================================
    print("\n📊 Pipeline Summary:")
    print(f"   1. Evaluation set: {'Created' if eval_file else 'Using standard split'}")
    print(f"   2. Model trained: {config.model_name}")
    print(f"   3. Adapter location: {results['adapter_path']}")
    print(f"   4. TensorBoard logs: {config.tensorboard_log_dir}")
    
    print("\n🚀 Next steps:")
    print("   1. View training metrics in TensorBoard")
    print("   2. Upload adapter to HuggingFace Hub:")
    print(f"      unsloth-cli upload {results['adapter_path']} <username/model-name>")
    print("   3. Run comprehensive evaluation:")
    print(f"      python examples/evaluate_model.py --adapter {results['adapter_path']}")
    
    print("\n✨ Full pipeline completed successfully!")


if __name__ == "__main__":
    main()