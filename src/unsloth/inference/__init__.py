"""Inference module for running and testing fine-tuned models."""
Module: __init__.py
Description: Package initialization and exports

from .generate import GenerationConfig, InferenceEngine
from .merge_adapter import merge_adapter_for_unsloth, merge_lora_adapter
from .test_suite import InferenceTestSuite, TestCase, TestResult, interactive_test_session

__all__ = [
    "InferenceEngine",
    "GenerationConfig",
    "InferenceTestSuite",
    "TestCase",
    "TestResult",
    "interactive_test_session",
    "merge_lora_adapter",
    "merge_adapter_for_unsloth"
]
