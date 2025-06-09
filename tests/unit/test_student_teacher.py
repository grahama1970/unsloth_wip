import pytest
import sys
from pathlib import Path
#!/usr/bin/env python3
"""Test script to verify student-teacher thinking enhancement with Claude."""

import asyncio
import sys
from pathlib import Path

# Add necessary paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/graham/workspace/experiments/claude_max_proxy/src')

from src.unsloth.data.thinking_enhancer import ThinkingEnhancer, StudentTeacherConfig
from loguru import logger


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_student_teacher():
    """Test the student-teacher enhancement with a simple example."""
    
    # Configure student-teacher setup
    config = StudentTeacherConfig(
        student_model="huggingface/microsoft/Phi-3.5-mini-instruct",  # Use proper HF provider
        teacher_model="anthropic/claude-3-opus-20240229",  # Use specific Claude model
        judge_model="openai/gpt-4o-mini",  # Use proper OpenAI provider
        max_iterations=3,
        use_local_student=False,  # Use API for testing
        student_temperature=0.7,
        teacher_temperature=0.8,
        thinking_format="iterative"
    )
    
    # Create enhancer
    enhancer = ThinkingEnhancer(config)
    
    # Test example
    test_example = {
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant that solves problems step by step."
            },
            {
                "role": "user",
                "content": "What is the result of 15 * 13?"
            },
            {
                "role": "assistant",
                "content": "The result of 15 * 13 is 195."
            }
        ],
        "metadata": {
            "thinking": "To multiply 15 * 13, I can use the distributive property: 15 * 13 = 15 * (10 + 3) = 150 + 45 = 195"
        }
    }
    
    logger.info("Testing student-teacher enhancement...")
    
    # Enhance the example
    enhanced = await enhancer._enhance_single(test_example)
    
    # Print results
    logger.info("\n=== Original Thinking ===")
    logger.info(enhanced["metadata"]["original_thinking"])
    
    logger.info("\n=== Enhanced Thinking ===")
    logger.info(enhanced["metadata"]["thinking"])
    
    logger.info(f"\n=== Statistics ===")
    logger.info(f"Iterations: {enhanced['metadata']['thinking_iterations']}")
    logger.info(f"Converged: {enhanced['metadata']['thinking_converged']}")
    
    if config.save_iterations and "thinking_iterations_detail" in enhanced["metadata"]:
        logger.info("\n=== Iteration Details ===")
        for i, iter_detail in enumerate(enhanced["metadata"]["thinking_iterations_detail"]):
            logger.info(f"\nIteration {i+1}:")
            logger.info(f"Answer: {iter_detail['answer']}")
            logger.info(f"Correct: {iter_detail['is_correct']}")
            if iter_detail.get('hint_received'):
                logger.info(f"Hint: {iter_detail['hint_received']}")


if __name__ == "__main__":
    try:
        asyncio.run(test_student_teacher())
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise