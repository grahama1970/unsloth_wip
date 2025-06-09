#!/usr/bin/env python3
"""Example: Test your fine-tuned model with comprehensive inference tests."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import asyncio
import json
from pathlib import Path
from unsloth.inference import InferenceEngine, GenerationConfig, InferenceTestSuite, TestCase


async def main():
    """Demonstrate comprehensive model testing after fine-tuning."""
    
    # Path to your fine-tuned model/adapter
    model_path = "./outputs/adapter"  # or "./lora_model" depending on where you saved it
    
    print(" Fine-Tuned Model Testing Examples\n")
    
    # 1. Basic Inference
    print("1️⃣ Basic Inference Test")
    print("-" * 50)
    
    engine = InferenceEngine(model_path, load_in_4bit=True)
    engine.load_model()
    
    # Test with a simple prompt
    prompt = "What is machine learning?"
    response = engine.generate(
        prompt,
        GenerationConfig(
            max_new_tokens=150,
            temperature=0.7,
            stream=False
        )
    )
    
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
    
    # 2. Test with different temperatures
    print("2️⃣ Temperature Comparison")
    print("-" * 50)
    
    prompt = "Write a creative description of a sunset."
    temperatures = [0.3, 0.7, 1.0]
    
    for temp in temperatures:
        response = engine.generate(
            prompt,
            GenerationConfig(
                max_new_tokens=100,
                temperature=temp,
                stream=False
            )
        )
        print(f"Temperature {temp}: {response[:100]}...\n")
    
    # 3. Batch inference
    print("3️⃣ Batch Inference")
    print("-" * 50)
    
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How do I make a perfect cup of coffee?"
    ]
    
    responses = engine.generate_batch(
        questions,
        GenerationConfig(max_new_tokens=100, temperature=0.7)
    )
    
    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r[:100]}...\n")
    
    # 4. Comprehensive Test Suite
    print("4️⃣ Running Comprehensive Test Suite")
    print("-" * 50)
    
    # Create custom test cases for your domain
    custom_tests = [
        TestCase(
            category="Domain Specific",
            question="What are the key benefits of using LoRA for fine-tuning?",
            expected_keywords=["parameter", "efficient", "memory", "adaptation"],
            expected_format="technical_explanation"
        ),
        TestCase(
            category="Domain Specific", 
            question="How does student-teacher learning improve model performance?",
            expected_keywords=["iterative", "feedback", "hints", "improvement"],
            expected_format="technical_explanation"
        ),
        TestCase(
            category="Code Understanding",
            question="Write a Python function to merge two sorted lists.",
            expected_keywords=["def", "merge", "while", "return", "sorted"],
            expected_format="python_code"
        )
    ]
    
    # Run comprehensive tests
    test_suite = InferenceTestSuite(
        model_path=str(model_path),
        output_dir="./model_test_results",
        use_judge=True,  # Use GPT-4 to evaluate responses
        judge_model="gpt-4"
    )
    
    # Add our custom tests
    test_suite.DEFAULT_TEST_CASES.extend(custom_tests)
    
    # Run specific categories
    results = await test_suite.run_tests(
        categories=["Factual Knowledge", "Reasoning", "Domain Specific"]
    )
    
    print("\n Test suite completed! Check ./model_test_results/ for detailed report.")
    
    # 5. Compare with base model (if you have both)
    print("\n5️⃣ Base vs Fine-tuned Comparison")
    print("-" * 50)
    
    # Example prompt that should show improvement after fine-tuning
    test_prompt = "Explain how to create a LoRA adapter for language models."
    
    # If you have the base model available
    try:
        base_engine = InferenceEngine("unsloth/Phi-3.5-mini-instruct", load_in_4bit=True)
        base_engine.load_model()
        
        base_response = base_engine.generate(test_prompt, GenerationConfig(max_new_tokens=150))
        finetuned_response = engine.generate(test_prompt, GenerationConfig(max_new_tokens=150))
        
        print("Base Model Response:")
        print(base_response[:200] + "...\n")
        
        print("Fine-tuned Model Response:")
        print(finetuned_response[:200] + "...")
        
    except Exception as e:
        print(f"Could not load base model for comparison: {e}")


def create_custom_test_file():
    """Create a template for custom test cases."""
    
    custom_tests = [
        {
            "category": "Your Domain",
            "question": "Your specific question here",
            "expected_keywords": ["keyword1", "keyword2"],
            "expected_format": "technical_explanation"
        },
        {
            "category": "Your Domain",
            "question": "Another domain-specific question",
            "expected_keywords": ["expected", "terms"],
            "expected_format": "comparison"
        }
    ]
    
    with open("custom_tests.json", "w") as f:
        json.dump(custom_tests, f, indent=2)
    
    print("Created custom_tests.json - edit this file to add your own test cases!")


if __name__ == "__main__":
    print("Unsloth Fine-tuned Model Testing")
    print("=" * 50 + "\n")
    
    print("CLI Usage Examples:")
    print("-" * 30)
    
    print("1. Basic inference:")
    print("   unsloth infer --model ./outputs/adapter --prompt 'What is AI?'")
    
    print("\n2. Interactive chat:")
    print("   unsloth infer --model ./outputs/adapter --interactive")
    
    print("\n3. Run test suite:")
    print("   unsloth test-inference --model ./outputs/adapter")
    
    print("\n4. Test specific categories:")
    print("   unsloth test-inference --model ./outputs/adapter \\")
    print("                          --category 'Math' --category 'Reasoning'")
    
    print("\n5. Interactive testing:")
    print("   unsloth test-inference --model ./outputs/adapter --interactive")
    
    print("\n6. Custom test file:")
    print("   unsloth test-inference --model ./outputs/adapter \\")
    print("                          --custom-tests custom_tests.json")
    
    print("\n" + "=" * 50)
    print("Running example tests...\n")
    
    # Create custom test file template
    create_custom_test_file()
    
    # Run the main testing
    asyncio.run(main())