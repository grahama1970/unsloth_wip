# Task 003: Test Inference with Enhanced Model

**Test ID**: inference_quality_003
**Module**: unsloth.inference.generate
**Goal**: Verify enhanced model shows self-correction behavior

## Working Code Example

```python
# COPY THIS WORKING PATTERN:
from pathlib import Path
from unsloth.inference.generate import InferenceEngine, GenerationConfig

def test_self_correction_ability():
    # Load enhanced model
    engine = InferenceEngine(
        model_path="./outputs/enhanced_thinking_model/final_adapter",
        max_seq_length=2048
    )
    
    # Test prompts that benefit from iterative thinking
    test_prompts = [
        "Explain why satellite communications are vulnerable to attacks, but first list what you know about satellites.",
        "Calculate the orbital period of a satellite at 400km altitude. Show your work step by step.",
        "What security measures protect satellite uplinks? Think through each layer of protection."
    ]
    
    # Generate with streaming to see thinking process
    gen_config = GenerationConfig(
        temperature=0.7,
        max_new_tokens=500,
        stream=True  # Watch reasoning unfold
    )
    
    results = []
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("Response: ", end="", flush=True)
        
        response = engine.generate(prompt, gen_config)
        results.append({
            "prompt": prompt,
            "response": response,
            "shows_iteration": "actually" in response or "wait" in response or "hmm" in response
        })
        
    return results

# Run it:
results = test_self_correction_ability()
correction_rate = sum(1 for r in results if r['shows_iteration']) / len(results)
print(f"\nSelf-correction indicators: {correction_rate:.1%}")
```

## Test Details

**Expected Behavior Differences**:

**Standard Model Response**:
```
The main vulnerabilities in satellite communications include signal interception, jamming, and spoofing attacks.
```

**Enhanced Model Response**:
```
Let me think through the vulnerabilities in satellite communications step by step.

First, I should consider the unique characteristics of satellites - they're in space, use radio signals, and are often unmanned.

Actually, I realize I should categorize these vulnerabilities:

1. Signal-based attacks:
   - Interception: Signals travel through open space
   - Jamming: Overwhelming legitimate signals
   - Wait, I should also mention spoofing...
   
2. Physical vulnerabilities:
   - Hmm, satellites can't be easily serviced...
```

**Run Command**:
```bash
# Test single prompt
python -c "
from unsloth.inference.generate import InferenceEngine
engine = InferenceEngine('./outputs/enhanced_thinking_model/final_adapter')
print(engine.generate('Explain how GPS spoofing works. Think carefully.'))
"
```

## Common Issues & Solutions

### Issue 1: Model doesn't show iterative thinking
```python
# Solution: Adjust generation parameters
gen_config = GenerationConfig(
    temperature=0.8,  # Higher for more variation
    top_p=0.95,      # Allow more diverse tokens
    repetition_penalty=1.0,  # Don't penalize thinking markers
)
```

### Issue 2: Responses are too verbose
```python
# Solution: Use stop sequences
gen_config = GenerationConfig(
    max_new_tokens=300,
    stop_sequences=["\n\nQuestion:", "\n\nUser:"],
)
```

### Issue 3: Model overfits to "Aha!" pattern
```python
# Solution: Vary prompts to not trigger training patterns
prompts = [
    "What are your thoughts on X?",
    "Can you analyze Y?", 
    "Please explain Z.",
    # Avoid: "Think step by step about..."
]
```

## Validation Requirements

```python
# Inference quality passes when:
# 1. Model loads successfully
assert engine.model is not None, "Model loaded"

# 2. Generates coherent responses
for result in results:
    assert len(result['response']) > 50, "Substantial response"
    assert result['response'].count('\n') > 2, "Multi-paragraph thinking"

# 3. Shows self-correction behavior (at least sometimes)
assert correction_rate > 0.3, "Shows iterative thinking in 30%+ responses"

# 4. Compare perplexity with base model
# Enhanced model should have similar or better perplexity on test set
```