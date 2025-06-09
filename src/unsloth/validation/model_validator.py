"""Model validation utilities for trained LoRA adapters."""
Module: model_validator.py
Description: Data models and schemas for model validator

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from unsloth import FastLanguageModel


class ModelValidator:
    """Validate trained LoRA adapters with various tests."""

    def __init__(self):
        """Initialize the validator."""
        self.model = None
        self.tokenizer = None

    async def validate_adapter(
        self,
        adapter_path: Path,
        base_model: str,
        test_prompts: list[str] | None = None,
        validation_dataset: Path | None = None,
        compare_base: bool = True
    ) -> dict[str, Any]:
        """
        Validate a trained LoRA adapter.
        
        Args:
            adapter_path: Path to the adapter files
            base_model: Base model name
            test_prompts: List of prompts to test
            validation_dataset: Optional validation dataset
            compare_base: Whether to compare with base model
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating adapter at: {adapter_path}")

        results = {
            "adapter_path": str(adapter_path),
            "base_model": base_model,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        try:
            # Load model with adapter
            logger.info("Loading model with adapter...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(self.model)

            # Basic inference test
            results["tests"]["basic_inference"] = await self._test_basic_inference()

            # Test with provided prompts
            if test_prompts:
                results["tests"]["prompt_responses"] = await self._test_prompts(test_prompts)

            # Compare with base model if requested
            if compare_base:
                results["tests"]["base_comparison"] = await self._compare_with_base(
                    base_model, test_prompts or self._get_default_prompts()
                )

            # Validate on dataset if provided
            if validation_dataset and validation_dataset.exists():
                results["tests"]["dataset_validation"] = await self._validate_on_dataset(
                    validation_dataset
                )

            # Memory and performance tests
            results["tests"]["performance"] = await self._test_performance()

            # Check adapter files
            results["tests"]["file_integrity"] = self._check_adapter_files(adapter_path)

            # Overall status
            all_passed = all(
                test.get("passed", False)
                for test in results["tests"].values()
                if isinstance(test, dict)
            )
            results["status"] = "passed" if all_passed else "failed"

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results["status"] = "error"
            results["error"] = str(e)

        finally:
            # Cleanup
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()

        return results

    async def _test_basic_inference(self) -> dict[str, Any]:
        """Test basic model inference."""
        logger.info("Testing basic inference...")

        try:
            prompt = "Hello, how are you?"
            inputs = self.tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "passed": len(response) > len(prompt),
                "prompt": prompt,
                "response_length": len(response),
                "sample_response": response[:100]
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    async def _test_prompts(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Test model with specific prompts."""
        logger.info(f"Testing {len(prompts)} prompts...")

        results = []
        for prompt in prompts:
            try:
                # Apply chat template if available
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted = prompt

                inputs = self.tokenizer(formatted, return_tensors="pt")

                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1]:],
                    skip_special_tokens=True
                )

                results.append({
                    "prompt": prompt,
                    "response": response,
                    "response_length": len(response),
                    "passed": len(response) > 10
                })

            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "error": str(e),
                    "passed": False
                })

        return results

    async def _compare_with_base(
        self,
        base_model: str,
        prompts: list[str]
    ) -> dict[str, Any]:
        """Compare adapter responses with base model."""
        logger.info("Comparing with base model...")

        try:
            # Load base model
            base_model_obj, base_tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(base_model_obj)

            comparisons = []

            for prompt in prompts[:3]:  # Limit to 3 for speed
                # Get adapter response
                adapter_response = await self._generate_response(
                    prompt, self.model, self.tokenizer
                )

                # Get base response
                base_response = await self._generate_response(
                    prompt, base_model_obj, base_tokenizer
                )

                # Compare
                comparisons.append({
                    "prompt": prompt,
                    "adapter_length": len(adapter_response),
                    "base_length": len(base_response),
                    "length_diff": len(adapter_response) - len(base_response),
                    "adapter_sample": adapter_response[:100],
                    "base_sample": base_response[:100]
                })

            # Cleanup base model
            del base_model_obj
            del base_tokenizer
            torch.cuda.empty_cache()

            # Analyze differences
            avg_length_diff = np.mean([c["length_diff"] for c in comparisons])

            return {
                "passed": True,
                "comparisons": comparisons,
                "avg_length_difference": avg_length_diff,
                "adapter_tends_to": "longer" if avg_length_diff > 0 else "shorter"
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    async def _generate_response(
        self,
        prompt: str,
        model: Any,
        tokenizer: Any
    ) -> str:
        """Generate a response from a model."""
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )

        return tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )

    async def _validate_on_dataset(self, dataset_path: Path) -> dict[str, Any]:
        """Validate on a dataset."""
        logger.info(f"Validating on dataset: {dataset_path}")

        try:
            # Load validation examples
            examples = []
            with open(dataset_path) as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Limit to 10 examples
                        break
                    examples.append(json.loads(line))

            # Test each example
            results = []
            for example in examples:
                if "messages" in example:
                    # Extract question
                    question = None
                    for msg in example["messages"]:
                        if msg["role"] == "user":
                            question = msg["content"]
                            break

                    if question:
                        response = await self._generate_response(
                            question, self.model, self.tokenizer
                        )
                        results.append({
                            "question": question[:50] + "...",
                            "response_length": len(response),
                            "passed": len(response) > 10
                        })

            passed_count = sum(1 for r in results if r.get("passed", False))

            return {
                "passed": passed_count == len(results),
                "total_examples": len(results),
                "passed_examples": passed_count,
                "success_rate": passed_count / len(results) if results else 0
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    async def _test_performance(self) -> dict[str, Any]:
        """Test model performance metrics."""
        logger.info("Testing performance...")

        try:
            prompt = "Write a short story about a robot."

            # Measure inference time
            import time
            start_time = time.time()

            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7
                )

            inference_time = time.time() - start_time
            tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]

            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
            else:
                memory_used = 0
                memory_reserved = 0

            return {
                "passed": True,
                "inference_time_seconds": round(inference_time, 3),
                "tokens_generated": tokens_generated,
                "tokens_per_second": round(tokens_generated / inference_time, 2),
                "gpu_memory_used_gb": round(memory_used, 2),
                "gpu_memory_reserved_gb": round(memory_reserved, 2)
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def _check_adapter_files(self, adapter_path: Path) -> dict[str, Any]:
        """Check adapter file integrity."""
        logger.info("Checking adapter files...")

        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]

        optional_files = [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "README.md"
        ]

        results = {
            "passed": True,
            "required_files": {},
            "optional_files": {}
        }

        # Check required files
        for file_name in required_files:
            file_path = adapter_path / file_name
            exists = file_path.exists()
            results["required_files"][file_name] = exists
            if not exists:
                results["passed"] = False

        # Check optional files
        for file_name in optional_files:
            file_path = adapter_path / file_name
            results["optional_files"][file_name] = file_path.exists()

        # Check file sizes
        results["file_sizes"] = {}
        for file_path in adapter_path.glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024 / 1024
                results["file_sizes"][file_path.name] = f"{size_mb:.2f} MB"

        return results

    def _get_default_prompts(self) -> list[str]:
        """Get default test prompts."""
        return [
            "What is machine learning?",
            "Write a Python function to reverse a string.",
            "Explain the difference between a list and a tuple in Python.",
            "What are the benefits of using version control?",
            "How does gradient descent work?"
        ]


async def main():
    """Example usage."""
    validator = ModelValidator()

    results = await validator.validate_adapter(
        adapter_path=Path("./outputs/adapter"),
        base_model="unsloth/Phi-3.5-mini-instruct",
        test_prompts=[
            "What is the capital of France?",
            "Write a haiku about coding.",
            "Explain recursion in simple terms."
        ],
        compare_base=True
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
