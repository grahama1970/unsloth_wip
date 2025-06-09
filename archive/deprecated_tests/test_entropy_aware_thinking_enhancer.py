"""
Module: test_entropy_aware_thinking_enhancer.py
Description: Tests for entropy-aware thinking enhancement without mocking

External Dependencies:
- pytest: https://docs.pytest.org/
- datasets: https://huggingface.co/docs/datasets/
- ollama: https://github.com/ollama/ollama-python

Sample Input:
>>> enhancer = EntropyAwareThinkingEnhancer(config)
>>> dataset = Dataset.from_dict({"question": ["What is 2+2?"], "answer": ["4"]})

Expected Output:
>>> enhanced = enhancer.enhance_dataset(dataset)
>>> enhanced[0]["text"]
'Question: What is 2+2?\n\n<thinking>\nThis is a basic arithmetic...\n</thinking>\n\nAnswer: 4'

Example Usage:
>>> pytest tests/unit/test_entropy_aware_thinking_enhancer.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import pytest
from datasets import Dataset
import os
import time
from typing import Dict, Any, List
import asyncio
from loguru import logger
from pathlib import Path
import json

from unsloth.data.entropy_aware_thinking_enhancer import (
    EntropyAwareThinkingEnhancer,
    EntropyAwareTeacherConfig
)
from unsloth.core.cost_aware_model_selector import CostAwareModelSelector
from unsloth.core.multi_gpu_cost_optimizer import MultiGPUCostOptimizer


class TestEntropyAwareThinkingEnhancer:
    """Test entropy-aware thinking enhancement with real models."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return EntropyAwareTeacherConfig(
            entropy_threshold=0.7,
            max_thinking_tokens=100,
            student_temperature=0.7,
            focus_on_high_entropy=True,
            student_model="Qwen/Qwen3-Reranker-4B",
            teacher_model="claude-3-opus-20240229"
        )
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a small sample dataset."""
        return Dataset.from_dict({
            "question": [
                "What is 2+2?",
                "Explain quantum entanglement briefly.",
                "What is the capital of France?"
            ],
            "answer": [
                "4",
                "Quantum entanglement is a phenomenon where particles become correlated.",
                "Paris"
            ]
        })
    
    def test_initialization(self, basic_config):
        """Test enhancer initialization."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        assert enhancer.config == basic_config
        assert enhancer.config.entropy_threshold == 0.7
        assert enhancer.config.student_model == "Qwen/Qwen3-Reranker-4B"
        assert hasattr(enhancer, 'model_selector')
    
    def test_convert_to_qa_format(self):
        """Test converting to QA format."""
        enhancer = EntropyAwareThinkingEnhancer(EntropyAwareTeacherConfig())
        
        # Test basic QA format
        item = {"question": "What is 2+2?", "answer": "4"}
        result = enhancer._convert_to_qa_format(item)
        
        assert result["question"] == "What is 2+2?"
        assert result["answer"] == "4"
        
        # Test with text field
        item_text = {"text": "Q: What is AI?\nA: Artificial Intelligence"}
        result_text = enhancer._convert_to_qa_format(item_text)
        
        assert "question" in result_text
        assert "answer" in result_text
    
    def test_identify_entropy_regions(self):
        """Test entropy region identification."""
        enhancer = EntropyAwareThinkingEnhancer(EntropyAwareTeacherConfig())
        
        # Test with simple text
        regions = enhancer._identify_entropy_regions("What is 2+2?")
        assert isinstance(regions, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in regions)
        
        # Test with more complex text
        complex_regions = enhancer._identify_entropy_regions(
            "Explain the implications of quantum entanglement on information theory."
        )
        assert isinstance(complex_regions, list)
    
    def test_convert_ranking_to_qa(self):
        """Test ranking format conversion."""
        enhancer = EntropyAwareThinkingEnhancer(EntropyAwareTeacherConfig())
        
        # Test ranking data conversion
        ranking_item = {
            "query": "What is machine learning?",
            "passages": [
                "Machine learning is a subset of AI.",
                "Cooking is an art form.",
                "ML algorithms learn from data."
            ],
            "relevance_scores": [0.9, 0.1, 0.8]
        }
        
        result = enhancer._convert_ranking_to_qa(ranking_item)
        
        assert "question" in result
        assert "answer" in result
        assert "machine learning" in result["question"].lower()
    
    def test_format_ranking_answer(self):
        """Test ranking answer formatting."""
        enhancer = EntropyAwareThinkingEnhancer(EntropyAwareTeacherConfig())
        
        passages = ["First passage", "Second passage"]
        scores = [0.9, 0.3]
        
        answer = enhancer._format_ranking_answer(passages, scores)
        
        assert isinstance(answer, str)
        assert "0.9" in answer
        assert "0.3" in answer
    
    def test_model_selection_logic(self, basic_config):
        """Test model selection for different scenarios."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        # Test selection for small task
        small_model = enhancer.model_selector.select_optimal_model(
            tokens_required=100,
            urgency="immediate",
            context_required=1000
        )
        
        assert small_model["provider"] in ["ollama", "vertex"]
        assert small_model["total_cost"] >= 0
        assert small_model["tokens_per_second"] > 0
    
    def test_gpu_optimization_logic(self, basic_config):
        """Test GPU optimization decisions."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        # Test for 70B model
        config_70b = enhancer.gpu_optimizer.get_optimal_gpu_config(
            model_size_gb=140,  # 70B model
            tokens_required=1000
        )
        
        assert "gpu_config" in config_70b
        assert config_70b["total_vram_gb"] >= 140 * 1.1  # At least model size + overhead
        assert config_70b["cost_per_hour"] > 0
    
    def test_load_from_dataset(self, basic_config, sample_dataset):
        """Test loading from dataset."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        # Load data
        examples = enhancer._load_from_dataset(sample_dataset, max_samples=2)
        
        assert len(examples) == 2
        assert all("question" in ex and "answer" in ex for ex in examples)
    
    @pytest.mark.asyncio
    async def test_generate_thinking_prompts(self, basic_config):
        """Test thinking prompt generation."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        # Test general prompt
        general_prompt = enhancer._create_general_thinking_prompt(
            question="What is AI?",
            answer="Artificial Intelligence",
            entropy_regions=[("AI", 0.9)]
        )
        
        assert isinstance(general_prompt, str)
        assert "AI" in general_prompt
        assert "thinking" in general_prompt.lower()
        
        # Test ranking prompt
        ranking_prompt = enhancer._create_ranking_thinking_prompt(
            query="Best programming language",
            passages=["Python is versatile", "Java is robust"],
            scores=[0.8, 0.6],
            entropy_regions=[("programming", 0.85)]
        )
        
        assert isinstance(ranking_prompt, str)
        assert "programming" in ranking_prompt
    
    def test_error_handling(self, basic_config):
        """Test error handling in enhancement."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        # Test with invalid input
        with pytest.raises(Exception):
            enhancer._convert_to_qa_format({})  # Missing required fields
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = EntropyAwareTeacherConfig(
            entropy_threshold=0.5,
            max_thinking_tokens=200
        )
        enhancer = EntropyAwareThinkingEnhancer(valid_config)
        assert enhancer.config.entropy_threshold == 0.5
        
        # Test with extreme values (should still work)
        extreme_config = EntropyAwareTeacherConfig(
            entropy_threshold=0.99,  # Very high threshold
            max_thinking_tokens=1000  # Large token count
        )
        enhancer_extreme = EntropyAwareThinkingEnhancer(extreme_config)
        assert enhancer_extreme.config.entropy_threshold == 0.99
    
    @pytest.mark.asyncio
    async def test_save_enhanced_data(self, basic_config, tmp_path):
        """Test saving enhanced data."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        # Create test data
        enhanced_examples = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "<thinking>Basic math.</thinking>\n\n4"}
                ],
                "text": "Question: What is 2+2?\n\n<thinking>Basic math.</thinking>\n\nAnswer: 4"
            }
        ]
        
        # Save data
        output_path = tmp_path / "enhanced_data.jsonl"
        enhancer._save_enhanced_data(enhanced_examples, output_path)
        
        # Verify file exists and contains data
        assert output_path.exists()
        with open(output_path, "r") as f:
            loaded = [json.loads(line) for line in f]
        
        assert len(loaded) == 1
        assert loaded[0]["text"] == enhanced_examples[0]["text"]
    
    def test_generate_statistics(self, basic_config):
        """Test statistics generation."""
        enhancer = EntropyAwareThinkingEnhancer(basic_config)
        
        examples = [
            {"has_thinking": True, "entropy_regions": [("test", 0.8)]},
            {"has_thinking": False, "entropy_regions": []},
            {"has_thinking": True, "entropy_regions": [("word", 0.9), ("another", 0.85)]}
        ]
        
        stats = enhancer._generate_statistics(examples)
        
        assert stats["total_examples"] == 3
        assert stats["examples_with_thinking"] == 2
        assert stats["average_entropy_regions"] == 1.0  # (0 + 1 + 2) / 3
        assert 0.8 <= stats["average_entropy_score"] <= 0.9  # Average of 0.8, 0.9, 0.85


# Validation
if __name__ == "__main__":
    # Run basic validation
    config = EntropyAwareTeacherConfig(
        entropy_threshold=0.7,
        max_thinking_tokens=150
    )
    
    enhancer = EntropyAwareThinkingEnhancer(config)
    
    # Test basic functionality
    test_item = {"question": "What is the capital of France?", "answer": "Paris"}
    converted = enhancer._convert_to_qa_format(test_item)
    
    print("Converted QA format:")
    print(f"  Question: {converted['question']}")
    print(f"  Answer: {converted['answer']}")
    
    # Test entropy regions
    regions = enhancer._identify_entropy_regions("Complex quantum mechanical phenomenon")
    print(f"\nIdentified {len(regions)} entropy regions")
    
    print("\n Module validation passed")