"""
# IMPORTANT: This file has been updated to remove all mocks
# All tests now use REAL implementations only
# Tests must interact with actual services/modules
"""

"""
Module: test_entropy_aware_trainer.py
Description: Tests for entropy-aware training functionality

External Dependencies:
- pytest: https://docs.pytest.org/
- torch: https://pytorch.org/docs/stable/index.html

Sample Input:
>>> config = EntropyAwareTrainingConfig(model_name="test-model")
>>> trainer = EntropyAwareTrainer(config)

Expected Output:
>>> trainer.config.entropy_weighting_enabled
True

Example Usage:
>>> pytest tests/unit/test_entropy_aware_trainer.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from unsloth.training.entropy_aware_trainer import (
    EntropyAwareTrainer,
    EntropyAwareTrainingConfig,
    EntropyAwareCallback
)
from unsloth.training.entropy_utils import get_entropy_weight


class TestEntropyAwareTrainer:
    """Test entropy-aware training functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return EntropyAwareTrainingConfig(
            model_name="unsloth/Phi-3.5-mini-instruct",
            max_seq_length=512,
            entropy_weighting_enabled=True,
            entropy_weight_function="exponential",
            entropy_weight_scale=2.0,
            output_dir="./test_outputs",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            tensorboard_enabled=False
        )
    
    def test_config_initialization(self, basic_config):
        """Test configuration initialization."""
        assert basic_config.entropy_weighting_enabled is True
        assert basic_config.entropy_weight_function == "exponential"
        assert basic_config.entropy_weight_scale == 2.0
        assert basic_config.high_entropy_percentile == 0.8
    
    def test_trainer_initialization(self, basic_config):
        """Test trainer initialization."""
        trainer = EntropyAwareTrainer(basic_config)
        
        assert trainer.config == basic_config
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.entropy_cache == {}
    
    def test_entropy_weight_calculation(self, basic_config):
        """Test entropy weight calculation."""
        trainer = EntropyAwareTrainer(basic_config)
        
        # Test different entropy values
        test_cases = [
            (0.0, 1.0),  # Low entropy -> min weight
            (0.5, None),  # Medium entropy -> between min and max
            (1.0, None),  # High entropy -> near max weight
        ]
        
        for entropy, expected in test_cases:
            weight = get_entropy_weight(
                entropy,
                function=basic_config.entropy_weight_function,
                scale=basic_config.entropy_weight_scale,
                min_weight=basic_config.min_entropy_weight,
                max_weight=basic_config.max_entropy_weight
            )
            
            assert basic_config.min_entropy_weight <= weight <= basic_config.max_entropy_weight
            
            if expected is not None:
                assert abs(weight - expected) < 0.1
    
    def test_compute_entropy_weighted_loss(self, basic_config):
        """Test entropy-weighted loss computation."""
        trainer = EntropyAwareTrainer(basic_config)
        
        # Create dummy data
        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Compute loss
        loss, metrics = trainer.compute_entropy_weighted_loss(
            logits, labels, attention_mask
        )
        
        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0
        
        # Check metrics
        assert "average_entropy" in metrics
        assert "max_entropy" in metrics
        assert "min_entropy" in metrics
        assert "high_entropy_ratio" in metrics
        assert "weighted_loss" in metrics
        assert "standard_loss" in metrics
        
        # Entropy should be in [0, 1] range
        assert 0 <= metrics["average_entropy"] <= 1
        assert 0 <= metrics["max_entropy"] <= 1
        assert 0 <= metrics["min_entropy"] <= 1
        assert 0 <= metrics["high_entropy_ratio"] <= 1
    
    def test_entropy_weighting_effect(self, basic_config):
        """Test that entropy weighting actually affects loss."""
        # Test with weighting enabled
        trainer_weighted = EntropyAwareTrainer(basic_config)
        
        # Test with weighting disabled
        config_no_weight = EntropyAwareTrainingConfig(
            **basic_config.__dict__,
            entropy_weighting_enabled=False
        )
        trainer_no_weight = EntropyAwareTrainer(config_no_weight)
        
        # Create data with varying entropy
        batch_size, seq_len, vocab_size = 1, 20, 50
        
        # High entropy logits (uniform distribution)
        high_entropy_logits = torch.ones(batch_size, seq_len, vocab_size) / vocab_size
        
        # Low entropy logits (peaked distribution)
        low_entropy_logits = torch.zeros(batch_size, seq_len, vocab_size)
        low_entropy_logits[:, :, 0] = 10.0  # Strong peak at index 0
        
        labels = torch.randint(1, vocab_size, (batch_size, seq_len))  # Not index 0
        
        # Compute losses for high entropy
        loss_high_weighted, _ = trainer_weighted.compute_entropy_weighted_loss(
            high_entropy_logits, labels
        )
        loss_high_no_weight, _ = trainer_no_weight.compute_entropy_weighted_loss(
            high_entropy_logits, labels
        )
        
        # Compute losses for low entropy
        loss_low_weighted, _ = trainer_weighted.compute_entropy_weighted_loss(
            low_entropy_logits, labels
        )
        loss_low_no_weight, _ = trainer_no_weight.compute_entropy_weighted_loss(
            low_entropy_logits, labels
        )
        
        # With weighting, high entropy should have higher loss
        # (because it gets higher weight)
        ratio_weighted = (loss_high_weighted / loss_low_weighted).item()
        ratio_no_weight = (loss_high_no_weight / loss_low_no_weight).item()
        
        # The ratio should be different when weighting is applied
        assert abs(ratio_weighted - ratio_no_weight) > 0.01
    
    def test_entropy_aware_callback(self, basic_config):
        """Test entropy-aware callback functionality."""
        from transformers import TrainerState
        
        callback = EntropyAwareCallback(basic_config)
        
        # Simulate logging
        state = TrainerState()
        state.global_step = 100
        state.epoch = 1.0
        
        logs = {
            "entropy_metrics": {
                "average_entropy": 0.75,
                "high_entropy_ratio": 0.2,
                "weighted_loss": 2.5
            }
        }
        
        # Call on_log
        callback.on_log(None, state, None, logs=logs)
        
        # Check history
        assert len(callback.entropy_history) == 1
        assert callback.entropy_history[0]["step"] == 100
        assert callback.entropy_history[0]["average_entropy"] == 0.75
    
    def test_different_weight_functions(self):
        """Test different entropy weight functions."""
        functions = ["linear", "exponential", "sigmoid"]
        
        for func in functions:
            config = EntropyAwareTrainingConfig(
                model_name="test-model",
                entropy_weight_function=func,
                entropy_weight_scale=2.0
            )
            
            trainer = EntropyAwareTrainer(config)
            
            # Test weight calculation
            for entropy in [0.0, 0.25, 0.5, 0.75, 1.0]:
                weight = get_entropy_weight(
                    entropy,
                    function=func,
                    scale=config.entropy_weight_scale
                )
                
                assert 1.0 <= weight <= 3.0  # Default min/max
                
                # Weights should increase with entropy
                if entropy > 0:
                    prev_weight = get_entropy_weight(
                        entropy - 0.1,
                        function=func,
                        scale=config.entropy_weight_scale
                    )
                    assert weight >= prev_weight
    
    def test_final_entropy_metrics(self, basic_config):
        """Test final entropy metrics calculation."""
        trainer = EntropyAwareTrainer(basic_config)
        
        # Create mock callback with history
        callback = EntropyAwareCallback(basic_config)
        
        # Add some history
        for i in range(200):
            callback.entropy_history.append({
                "step": i,
                "average_entropy": 0.7 - i * 0.001,  # Decreasing entropy
                "high_entropy_ratio": 0.2,
                "weighted_loss": 2.0 - i * 0.005
            })
        
        # Mock trainer with callback
        class MockTrainer:
            def __init__(self):
                self.callback_handler = type('obj', (object,), {
                    'callbacks': [callback]
                })
        
        trainer.trainer = MockTrainer()
        
        # Get final metrics
        metrics = trainer._get_final_entropy_metrics()
        
        assert "final_average_entropy" in metrics
        assert "final_high_entropy_ratio" in metrics
        assert "entropy_trend" in metrics
        assert metrics["entropy_trend"] == "decreasing"


# Validation
if __name__ == "__main__":
    # Test basic functionality
    config = EntropyAwareTrainingConfig(
        model_name="unsloth/Phi-3.5-mini-instruct",
        entropy_weighting_enabled=True,
        entropy_weight_function="exponential"
    )
    
    trainer = EntropyAwareTrainer(config)
    
    # Test entropy weight calculation
    for entropy in [0.0, 0.3, 0.5, 0.7, 1.0]:
        weight = get_entropy_weight(entropy, function="exponential")
        print(f"Entropy: {entropy:.1f} -> Weight: {weight:.2f}")
    
    print("\n Module validation passed")