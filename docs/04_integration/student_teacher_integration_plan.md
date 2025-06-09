# Student-Teacher Integration Plan for ArangoDB Q&A Generation

## Overview

Enhance the `thinking` field generation in ArangoDB Q&A module by implementing a student-teacher iterative reasoning approach, creating richer chain-of-thought data for fine-tuning.

## Current vs. Proposed Architecture

### Current (Single-Pass)
```
Question → Single LLM → {question, thinking, answer}
```

### Proposed (Student-Teacher)
```
Question → Student Model → Initial Reasoning
    ↓ (if incorrect)
Teacher Hint → Student Retry → Refined Reasoning
    ↓ (repeat up to N times)
Final Output → {question, rich_thinking_chain, answer}
```

## Implementation Strategy

### 1. **Modify ArangoDB Q&A Generation**

```python
# In arangodb/src/arangodb/qa_generation/generator.py

class StudentTeacherConfig(BaseModel):
    """Configuration for student-teacher thinking generation."""
    enable_student_teacher: bool = True
    student_model: str = "unsloth/Phi-3.5-mini-instruct"  # Smaller model
    teacher_model: str = "gpt-4o-mini"  # Current model
    max_iterations: int = 3
    hint_temperature: float = 0.8
    student_temperature: float = 0.7
    
class ThinkingChain(BaseModel):
    """Structured thinking with iterations."""
    iterations: List[Dict[str, str]] = Field(default_factory=list)
    final_reasoning: str
    iteration_count: int
    hints_received: List[str] = Field(default_factory=list)
```

### 2. **Enhanced Thinking Generation**

```python
async def _generate_thinking_with_student_teacher(
    self,
    question: str,
    correct_answer: str,
    content: str,
    config: StudentTeacherConfig
) -> ThinkingChain:
    """Generate thinking field using student-teacher approach."""
    
    thinking_chain = ThinkingChain()
    current_reasoning = ""
    
    for iteration in range(config.max_iterations):
        # Student attempt
        student_response = await self._student_attempt(
            question=question,
            content=content,
            previous_reasoning=current_reasoning,
            model=config.student_model,
            temperature=config.student_temperature
        )
        
        # Extract reasoning and answer
        student_reasoning = student_response.get("reasoning", "")
        student_answer = student_response.get("answer", "")
        
        thinking_chain.iterations.append({
            "iteration": iteration + 1,
            "reasoning": student_reasoning,
            "answer": student_answer
        })
        
        # Check if correct
        if self._is_answer_correct(student_answer, correct_answer):
            thinking_chain.final_reasoning = student_reasoning
            thinking_chain.iteration_count = iteration + 1
            break
            
        # Get teacher hint (if not last iteration)
        if iteration < config.max_iterations - 1:
            hint = await self._get_teacher_hint(
                question=question,
                student_reasoning=student_reasoning,
                student_answer=student_answer,
                model=config.teacher_model,
                temperature=config.hint_temperature
            )
            thinking_chain.hints_received.append(hint)
            current_reasoning = f"{student_reasoning}\n\nAha! {hint}"
    
    # Format final thinking field
    thinking_text = self._format_thinking_chain(thinking_chain)
    return thinking_text
```

### 3. **Benefits of This Approach**

#### **Richer Training Data**
- Multi-step reasoning chains
- Self-correction patterns
- "Aha!" moments that teach error recovery

#### **Better Model Alignment**
- Student model matches deployment model size
- Thinking patterns match actual inference behavior
- Natural incorporation of hints and corrections

#### **Quality Control**
- Teacher ensures correctness
- Multiple attempts increase robustness
- Validation against source content

### 4. **Integration with Unsloth Training**

The enhanced thinking field will:
- Provide better chain-of-thought examples
- Teach self-correction behavior
- Improve reasoning on complex tasks

### 5. **Configuration Options**

```python
# In ArangoDB config
qa_generation_config = {
    "student_teacher": {
        "enable": True,
        "student_model": "unsloth/Phi-3.5-mini-instruct",
        "teacher_model": "gpt-4o-mini",
        "max_iterations": 3,
        "use_local_student": True,  # Run student locally
        "use_api_teacher": True,     # Teacher via API
        "thinking_format": "iterative"  # or "consolidated"
    }
}
```

## Implementation Priority

1. **Phase 1**: Add StudentTeacherConfig to ArangoDB
2. **Phase 2**: Implement iterative thinking generation
3. **Phase 3**: Add async teacher hint manager
4. **Phase 4**: Create thinking chain formatter
5. **Phase 5**: Add metrics and monitoring

## Expected Outcomes

- **50-100% richer thinking data** with error correction patterns
- **Better generalization** in fine-tuned models
- **Improved reasoning** on edge cases
- **Natural self-correction** behavior in deployed models

## Next Steps

1. Update ArangoDB Q&A generator with student-teacher support
2. Configure appropriate student/teacher model pairs
3. Test on sample documents
4. Monitor thinking quality metrics
5. Fine-tune unsloth models on enhanced data