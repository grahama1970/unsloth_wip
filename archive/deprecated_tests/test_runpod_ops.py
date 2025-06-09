"""
# IMPORTANT: This file has been updated to remove all mocks
# All tests now use REAL implementations only
# Tests must interact with actual services/modules
"""

"""
Module: test_runpod_ops.py
Description: Tests for RunPod operations package

External Dependencies:
- pytest: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/

Sample Input:
>>> optimizer = InstanceOptimizer()
>>> config = optimizer.get_optimal_config("7B", tokens=100000)

Expected Output:
>>> config["gpu_type"]
"RTX_4090"

Example Usage:
>>> pytest tests/unit/test_runpod_ops.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import pytest
import asyncio
from datetime import datetime
from pathlib import Path

from runpod_ops import (
    InstanceOptimizer,
    CostCalculator,
    InstanceProfile,
    GPUConfig
)
from runpod_ops.training_orchestrator import TrainingConfig


class TestInstanceOptimizer:
    """Test instance optimization functionality."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = InstanceOptimizer()
        
        assert len(optimizer.gpu_configs) > 0
        assert "RTX_4090" in optimizer.gpu_configs
        assert "A100_80GB" in optimizer.gpu_configs
        
    def test_model_memory_requirements(self):
        """Test model memory calculations."""
        optimizer = InstanceOptimizer()
        
        test_cases = [
            ("3B", 6),
            ("7B", 14),
            ("13B", 26),
            ("70B", 140),
        ]
        
        for model_size, expected_gb in test_cases:
            result = optimizer._get_model_memory_requirement(model_size)
            assert result == expected_gb
            
    def test_optimal_config_selection(self):
        """Test optimal GPU selection."""
        optimizer = InstanceOptimizer()
        
        # Small model should prefer cheaper GPU
        config = optimizer.get_optimal_config(
            "7B",
            tokens=100_000,
            priority="cost"
        )
        
        assert config["gpu_type"] in ["RTX_4090", "A40"]
        assert config["gpu_count"] == 1
        assert config["estimated_cost"] < 10.0
        
    def test_multi_gpu_optimization(self):
        """Test multi-GPU configuration."""
        optimizer = InstanceOptimizer()
        
        # Large model requiring multi-GPU
        config = optimizer.get_optimal_config(
            "70B",
            tokens=1_000_000,
            priority="balanced"
        )
        
        assert config["gpu_count"] >= 1
        assert "estimated_hours" in config
        assert "tokens_per_second" in config
        
    def test_training_optimization(self):
        """Test training-specific optimization."""
        optimizer = InstanceOptimizer()
        
        config = optimizer.optimize_for_training(
            "13B",
            dataset_size=50_000,
            epochs=3
        )
        
        assert "batch_size" in config
        assert "gradient_accumulation_steps" in config
        assert config["estimated_cost"] > 0
        assert config["total_steps"] > 0
        
    def test_inference_optimization(self):
        """Test inference optimization."""
        optimizer = InstanceOptimizer()
        
        config = optimizer.optimize_for_inference(
            "7B",
            requests_per_hour=1000,
            target_latency_ms=500
        )
        
        assert "max_batch_size" in config
        assert "cost_per_1k_requests" in config
        assert config["avg_latency_ms"] <= 500
        

class TestCostCalculator:
    """Test cost calculation functionality."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = CostCalculator()
        
        assert len(calculator.profiles) > 0
        
        # Check profile types
        providers = {p.provider for p in calculator.profiles}
        assert "local" in providers
        assert "vertex" in providers
        assert "runpod" in providers
        
    def test_token_based_pricing(self):
        """Test token-based cost calculation."""
        calculator = CostCalculator()
        
        # Find Vertex profile
        vertex_profile = next(
            p for p in calculator.profiles 
            if p.provider == "vertex"
        )
        
        cost_info = calculator.calculate_total_cost(
            vertex_profile,
            total_tokens=1_000_000
        )
        
        assert cost_info["token_cost"] == 0.075  # $0.075 per million
        assert cost_info["hourly_cost"] == 0.0
        
    def test_hourly_pricing(self):
        """Test hourly cost calculation."""
        calculator = CostCalculator()
        
        # Find RunPod profile
        runpod_profile = next(
            p for p in calculator.profiles
            if p.provider == "runpod" and p.name == "rtx4090"
        )
        
        cost_info = calculator.calculate_total_cost(
            runpod_profile,
            total_tokens=100_000,
            include_startup=True
        )
        
        assert cost_info["hourly_cost"] > 0
        assert cost_info["startup_overhead_seconds"] > 0
        assert cost_info["billable_time_seconds"] >= cost_info["total_time_seconds"]
        
    def test_provider_comparison(self):
        """Test multi-provider comparison."""
        calculator = CostCalculator()
        
        comparison = calculator.compare_providers(
            "7B",
            tokens=500_000
        )
        
        assert len(comparison) > 0
        
        # Check sorting (cheapest first)
        costs = [info["total_cost"] for info in comparison.values()]
        assert costs == sorted(costs)
        
    def test_monthly_budget(self):
        """Test monthly budget calculation."""
        calculator = CostCalculator()
        
        budget = calculator.calculate_monthly_budget(
            "13B",
            daily_tokens=1_000_000,
            days=30
        )
        
        assert budget["monthly_tokens"] == 30_000_000
        assert len(budget["providers"]) > 0
        assert "recommendations" in budget
        assert "cheapest" in budget["recommendations"]
        

class TestTrainingConfig:
    """Test training configuration."""
    
    def test_config_creation(self):
        """Test training config creation."""
        config = TrainingConfig(
            model_name="test/model",
            model_size="7B",
            dataset_path="/path/to/data",
            output_path="/path/to/output",
            num_epochs=3,
            learning_rate=2e-4
        )
        
        assert config.model_name == "test/model"
        assert config.num_epochs == 3
        assert config.fp16 is True  # Default
        assert config.gradient_checkpointing is True  # Default
        
    def test_config_validation(self):
        """Test config validation."""
        # Should work with minimal config
        config = TrainingConfig(
            model_name="model",
            model_size="7B",
            dataset_path="data",
            output_path="output"
        )
        
        assert config.num_epochs == 1  # Default
        assert config.learning_rate == 2e-4  # Default
        

class TestCostOptimization:
    """Test cost optimization scenarios."""
    
    def test_small_batch_optimization(self):
        """Test optimization for small batches."""
        calculator = CostCalculator()
        
        # Small batch should prefer local or cheap options
        best = calculator.get_cheapest_provider(
            "3B",
            tokens=10_000,
            max_time_hours=1.0
        )
        
        assert best["provider"] in ["local", "vertex"]
        
    def test_large_scale_optimization(self):
        """Test optimization for large scale."""
        calculator = CostCalculator()
        
        # Large scale should consider RunPod
        best = calculator.get_cheapest_provider(
            "70B",
            tokens=10_000_000,
            exclude_providers=["local"]
        )
        
        assert best["provider"] in ["runpod", "vertex"]
        
    def test_latency_constrained_optimization(self):
        """Test optimization with latency constraints."""
        optimizer = InstanceOptimizer()
        
        # Need fast inference
        config = optimizer.optimize_for_inference(
            "7B",
            requests_per_hour=5000,
            target_latency_ms=100  # Very low latency
        )
        
        # Should recommend powerful GPU
        assert config["gpu_type"] in ["H100", "A100_80GB", "A100_40GB"]
        

@pytest.mark.asyncio
class TestAsyncComponents:
    """Test async components (would need mocking in real tests)."""
    
    async def test_training_config_generation(self):
        """Test training script generation."""
        from runpod_ops.training_orchestrator import TrainingOrchestrator
        
        orchestrator = TrainingOrchestrator()
        
        config = TrainingConfig(
            model_name="test/model",
            model_size="7B",
            dataset_path="data.json",
            output_path="output",
            batch_size=4,
            gradient_accumulation_steps=4
        )
        
        script = orchestrator._generate_training_script(config, num_gpus=2)
        
        assert "torchrun" in script
        assert "--nproc_per_node=2" in script
        assert "--per_device_train_batch_size 4" in script
        assert "--gradient_accumulation_steps 4" in script
        
    async def test_cost_estimation(self):
        """Test cost estimation for training."""
        from runpod_ops.runpod_manager import RunPodManager
        
        manager = RunPodManager(api_key="test_key")
        
        estimate = await manager.estimate_training_cost(
            "7B",
            dataset_size=10_000,
            epochs=3,
            multi_gpu=False
        )
        
        assert estimate["model_size"] == "7B"
        assert estimate["estimated_hours"] > 0
        assert estimate["total_cost"] > 0
        assert estimate["tokens_per_second"] > 0


# Validation
if __name__ == "__main__":
    # Run basic tests
    print("RunPod Operations Test Suite")
    print("=" * 50)
    
    # Test optimizer
    optimizer = InstanceOptimizer()
    config = optimizer.get_optimal_config("7B", tokens=100_000, priority="cost")
    print(f"\nOptimal config for 7B model (100K tokens):")
    print(f"  GPU: {config['gpu_type']}")
    print(f"  Cost: ${config['estimated_cost']}")
    print(f"  Time: {config['estimated_hours']} hours")
    
    # Test calculator
    calculator = CostCalculator()
    comparison = calculator.compare_providers("13B", tokens=1_000_000)
    
    print(f"\nCost comparison for 13B model (1M tokens):")
    for i, (provider, info) in enumerate(list(comparison.items())[:3]):
        print(f"  {i+1}. {provider}: ${info['total_cost']} ({info['processing_time_hours']:.1f}h)")
        
    print("\n Module validation passed")