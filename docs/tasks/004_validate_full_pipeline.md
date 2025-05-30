# Task 004: Validate Complete Enhancement Pipeline

**Test ID**: full_pipeline_validation_004
**Module**: Full integration test
**Goal**: Ensure entire pipeline works end-to-end

## Working Code Example

```python
# COPY THIS WORKING PATTERN:
import asyncio
import json
from pathlib import Path
from unsloth.data.thinking_enhancer import ThinkingEnhancer, StudentTeacherConfig
from unsloth.core.enhanced_config import EnhancedTrainingConfig
from unsloth.training.enhanced_trainer import EnhancedUnslothTrainer
from unsloth.inference.generate import InferenceEngine

async def validate_full_pipeline():
    """Complete pipeline: enhance → train → test"""
    
    # Step 1: Enhance thinking
    print("Step 1: Enhancing thinking fields...")
    enhancer_config = StudentTeacherConfig(
        student_model="unsloth/Phi-3.5-mini-instruct",
        teacher_model="gpt-4o-mini",
        max_iterations=2,  # Faster for testing
        batch_size=5
    )
    
    enhancer = ThinkingEnhancer(enhancer_config)
    enhance_stats = await enhancer.enhance_dataset(
        input_path=Path("./test_data/qa_small.jsonl"),
        output_path=Path("./test_data/qa_enhanced.jsonl"),
        max_samples=50  # Small test set
    )
    
    print(f"✓ Enhanced {enhance_stats['enhanced_examples']} examples")
    
    # Step 2: Train on enhanced data
    print("\nStep 2: Training on enhanced data...")
    train_config = EnhancedTrainingConfig(
        model_name="unsloth/Phi-3.5-mini-instruct",
        dataset_path="./test_data/qa_enhanced.jsonl",
        dataset_source="arangodb",
        
        # Quick training for validation
        num_train_epochs=1,
        eval_steps=10,
        save_steps=50,
        lora_r=16,  # Smaller for speed
        
        output_dir="./test_outputs/pipeline_test"
    )
    
    trainer = EnhancedUnslothTrainer(train_config)
    train_results = trainer.train()
    trainer.cleanup()
    
    print(f"✓ Training complete in {train_results['training_time']:.1f}s")
    
    # Step 3: Test inference quality
    print("\nStep 3: Testing inference quality...")
    engine = InferenceEngine(train_results['adapter_path'])
    
    # Test with a sample that should trigger iterative thinking
    test_prompt = "Explain the security risks in IoT devices. First, what makes them vulnerable?"
    response = engine.generate(test_prompt)
    
    quality_checks = {
        "has_content": len(response) > 100,
        "multi_paragraph": response.count('\n') > 2,
        "shows_thinking": any(marker in response.lower() for marker in 
                            ['first', 'then', 'actually', 'wait', 'hmm'])
    }
    
    print(f"✓ Inference quality: {sum(quality_checks.values())}/3 checks passed")
    
    return {
        "enhance_stats": enhance_stats,
        "train_results": train_results,
        "quality_checks": quality_checks,
        "all_passed": all(quality_checks.values())
    }

# Run it:
results = asyncio.run(validate_full_pipeline())
print(f"\nPipeline validation: {'PASSED' if results['all_passed'] else 'FAILED'}")
```

## Test Details

**Pipeline Data Flow**:
```
qa_small.jsonl (50 samples)
    ↓ [Thinking Enhancement]
qa_enhanced.jsonl (50 enhanced)
    ↓ [Training]
pipeline_test/final_adapter
    ↓ [Inference]
Quality validated output
```

**Expected Timings (GPU)**:
- Enhancement: 2-5 minutes (API calls)
- Training: 5-10 minutes (1 epoch, small data)
- Inference: <1 second per prompt

**Run Command**:
```bash
# Full pipeline test
python test_full_pipeline.py

# Or using pytest
pytest tests/integration/test_enhancement_pipeline.py -v
```

## Common Issues & Solutions

### Issue 1: Enhancement timeout
```python
# Solution: Reduce concurrent API calls
enhancer_config = StudentTeacherConfig(
    batch_size=2,  # Smaller batches
    max_iterations=2  # Fewer iterations
)
```

### Issue 2: Training fails on small dataset
```python
# Solution: Adjust validation split
train_config = EnhancedTrainingConfig(
    validation_split=0.2,  # Less data for validation
    # Or disable validation
    evaluation_strategy="no"
)
```

### Issue 3: Pipeline state management
```python
# Solution: Clean up between runs
import shutil

def cleanup_test_artifacts():
    paths = [
        "./test_data/qa_enhanced.jsonl",
        "./test_outputs/pipeline_test"
    ]
    for path in paths:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()
```

## Validation Requirements

```python
# Pipeline validation passes when:

# 1. Enhancement completes
assert enhance_stats['enhanced_examples'] == 50, "All examples enhanced"
assert enhance_stats['average_iterations'] > 1.0, "Multi-iteration enhancement"

# 2. Training produces adapter
assert Path(train_results['adapter_path']).exists(), "Adapter created"
assert Path(train_results['adapter_path'] / "adapter_config.json").exists()

# 3. Inference shows improvement
assert all(quality_checks.values()), "All quality checks pass"

# 4. No data leakage
with open("./test_data/qa_enhanced.jsonl") as f:
    sample = json.loads(f.readline())
    # Enhanced thinking should be in metadata, not messages
    assert sample['metadata']['thinking'] != sample['metadata']['original_thinking']
    assert len(sample['messages']) == 3  # System, user, assistant unchanged
```