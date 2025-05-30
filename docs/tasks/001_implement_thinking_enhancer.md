# Task 001: Implement Student-Teacher Thinking Enhancement

**Test ID**: thinking_enhancement_001
**Module**: unsloth.data.thinking_enhancer
**Goal**: Enhance Q&A thinking fields with student-teacher iterations

## Working Code Example

```python
# COPY THIS WORKING PATTERN:
import asyncio
from pathlib import Path
from unsloth.data.thinking_enhancer import ThinkingEnhancer, StudentTeacherConfig

async def enhance_qa_thinking():
    # Configure student-teacher
    config = StudentTeacherConfig(
        student_model="unsloth/Phi-3.5-mini-instruct",  # Local small model
        teacher_model="gpt-4o-mini",                     # API teacher
        max_iterations=3,
        use_local_student=True
    )
    
    # Initialize enhancer
    enhancer = ThinkingEnhancer(config)
    
    # Enhance dataset
    stats = await enhancer.enhance_dataset(
        input_path=Path("./data/qa_sample.jsonl"),
        output_path=Path("./data/qa_enhanced.jsonl"),
        max_samples=10  # Start small for testing
    )
    
    return stats

# Run it:
result = asyncio.run(enhance_qa_thinking())
print(f"Enhanced {result['enhanced_examples']} examples")
print(f"Average iterations: {result['average_iterations']:.2f}")
```

## Test Details

**Input Q&A Format (from ArangoDB)**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Document: space_cybersecurity. Section: introduction"
    },
    {
      "role": "user", 
      "content": "What are the main vulnerabilities in satellite communications?"
    },
    {
      "role": "assistant",
      "content": "The main vulnerabilities include signal interception, jamming, and spoofing attacks."
    }
  ],
  "metadata": {
    "thinking": "Basic single-pass reasoning",
    "question_type": "FACTUAL",
    "confidence": 0.95
  }
}
```

**Enhanced Output Format**:
```json
{
  "messages": [...],
  "metadata": {
    "original_thinking": "Basic single-pass reasoning",
    "thinking": "Let me analyze satellite vulnerabilities step by step...\n\nAha! I should consider both technical and operational aspects...\n\nThe main vulnerabilities include...",
    "thinking_iterations": 2,
    "thinking_converged": true
  }
}
```

**Run Command**:
```bash
# CLI usage
unsloth-cli enhance-thinking \
  ./data/qa_sample.jsonl \
  ./data/qa_enhanced.jsonl \
  --student unsloth/Phi-3.5-mini-instruct \
  --teacher gpt-4o-mini \
  --iterations 3
```

## Common Issues & Solutions

### Issue 1: Student model loading fails
```python
# Solution: Ensure 4-bit quantization support
from unsloth import FastLanguageModel

# Test model loading separately
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

### Issue 2: API rate limits with teacher model
```python
# Solution: Already handled with retry logic and batching
# Adjust batch_size if needed:
config = StudentTeacherConfig(
    batch_size=5,  # Reduce from default 10
    # Add delay between batches
)
```

### Issue 3: Memory issues with local student
```python
# Solution: Use smaller batches or API student
config = StudentTeacherConfig(
    use_local_student=False,  # Use API for both
    student_model="gpt-3.5-turbo",  # Smaller API model
)
```

## Validation Requirements

```python
# This enhancement passes when:
assert stats['enhanced_examples'] == stats['total_examples'], "All examples enhanced"
assert stats['average_iterations'] > 1.0, "Multiple iterations occurred"
assert stats['convergence_rate'] > 0.7, "Most examples converged"

# Check enhanced thinking quality
with open("./data/qa_enhanced.jsonl") as f:
    sample = json.loads(f.readline())
    thinking = sample['metadata']['thinking']
    assert "Aha!" in thinking, "Contains teacher hints"
    assert len(thinking) > 200, "Richer than original"
```