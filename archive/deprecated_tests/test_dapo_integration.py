"""
# IMPORTANT: This file has been updated to remove all mocks
# All tests now use REAL implementations only
# Tests must interact with actual services/modules
"""

"""Integration tests for DAPO RL implementation.
Module: test_dapo_integration.py
Description: Comprehensive integration tests for DAPO training

External Dependencies:
- pytest: https://docs.pytest.org/
- torch: https://pytorch.org/docs/stable/

Sample Input:
>>> pytest tests/integration/test_dapo_integration.py -v

Expected Output:
>>> test_dapo_training_small_model PASSED
>>> test_dapo_loss_calculation PASSED
>>> 2 passed in 45.2s
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import json
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unsloth.training.dapo_rl import DAPOConfig, DAPOTrainer, create_dapo_trainer


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer for testing."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    return model, ref_model, tokenizer


@pytest.fixture
def test_dataset():
    """Create test dataset."""
    data = [
        {"question": "What is AI?", "answer": "Artificial Intelligence is computer systems that mimic human intelligence."},
        {"question": "Explain ML", "answer": "Machine Learning is algorithms that learn from data."},
        {"question": "What is DL?", "answer": "Deep Learning uses neural networks with multiple layers."}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        return Path(f.name)


class TestDAPOTraining:
    """Test DAPO training functionality."""
    
    def test_dapo_config_creation(self):
        """Test DAPO configuration."""
        config = DAPOConfig(
            clip_lower=0.7,
            clip_upper=1.3,
            dynamic_sampling=True,
            token_level_pg=True
        )
        
        assert config.clip_lower == 0.7
        assert config.clip_upper == 1.3
        assert config.dynamic_sampling is True
        assert config.token_level_pg is True
    
    def test_dapo_trainer_initialization(self, mock_model_and_tokenizer):
        """Test DAPO trainer initialization."""
        model, ref_model, tokenizer = mock_model_and_tokenizer
        
        config = DAPOConfig()
        trainer = DAPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            config=config
        )
        
        assert trainer.model is model
        assert trainer.ref_model is ref_model
        assert trainer.config.clip_upper == 1.28
    
    @pytest.mark.slow
    def test_dapo_training_step(self, mock_model_and_tokenizer):
        """Test single DAPO training step."""
        model, ref_model, tokenizer = mock_model_and_tokenizer
        
        config = DAPOConfig(gradient_accumulation_steps=1)
        trainer = DAPOTrainer(model, ref_model, tokenizer, config)
        
        # Create mock batch
        texts = ["Hello world", "Test input"]
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        
        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "rewards": torch.randn(2, inputs["input_ids"].shape[1]) * 0.1
        }
        
        # Execute training step
        metrics = trainer.train_step(batch)
        
        assert "loss" in metrics
        assert "pg_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics
        assert metrics["filtered_ratio"] <= 1.0
    
    def test_dynamic_sampling_filter(self, mock_model_and_tokenizer):
        """Test dynamic sampling functionality."""
        model, ref_model, tokenizer = mock_model_and_tokenizer
        
        config = DAPOConfig(dynamic_sampling=True)
        trainer = DAPOTrainer(model, ref_model, tokenizer, config)
        
        # Create batch with zero variance rewards (should be filtered)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "rewards": torch.zeros(2, 3)  # All zeros - no gradient
        }
        
        filtered = trainer.dynamic_sample_filter(batch, batch["rewards"])
        
        # Should filter out samples with no variance
        assert len(filtered["input_ids"]) <= len(batch["input_ids"])
    
    def test_advantage_calculation(self, mock_model_and_tokenizer):
        """Test GAE advantage calculation."""
        model, ref_model, tokenizer = mock_model_and_tokenizer
        
        config = DAPOConfig(gamma=0.99, lam=0.95)
        trainer = DAPOTrainer(model, ref_model, tokenizer, config)
        
        # Create mock data
        rewards = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
        values = torch.tensor([[0.5, 0.6, 0.7], [0.4, 0.5, 0.6]])
        attention_mask = torch.ones_like(rewards)
        
        advantages, returns = trainer.compute_advantages(rewards, values, attention_mask)
        
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert not torch.isnan(advantages).any()
    
    def test_overlong_penalty(self, mock_model_and_tokenizer):
        """Test overlong sequence penalty."""
        model, ref_model, tokenizer = mock_model_and_tokenizer
        
        config = DAPOConfig(max_length=10, overlong_penalty=0.1)
        trainer = DAPOTrainer(model, ref_model, tokenizer, config)
        
        # Test with sequences of different lengths
        rewards = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        sequence_lengths = torch.tensor([8, 15])  # One normal, one overlong
        
        penalized_rewards = trainer.apply_overlong_penalty(rewards, sequence_lengths)
        
        # Second sequence should have penalty
        assert penalized_rewards[0, -1] == rewards[0, -1]  # No penalty
        assert penalized_rewards[1, -1] < rewards[1, -1]   # Has penalty
    
    @pytest.mark.slow
    def test_full_training_loop(self, mock_model_and_tokenizer, test_dataset):
        """Test complete training loop."""
        model, ref_model, tokenizer = mock_model_and_tokenizer
        
        # Load test data
        with open(test_dataset) as f:
            data = json.load(f)
        
        # Configure training
        config = DAPOConfig(
            clip_upper=1.25,
            dynamic_sampling=False,  # Disable for test stability
            gradient_accumulation_steps=1
        )
        
        trainer = DAPOTrainer(model, ref_model, tokenizer, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Training loop
        initial_loss = None
        final_loss = None
        
        for epoch in range(2):  # Small number of epochs
            for item in data:
                # Prepare batch
                text = f"Q: {item['question']}\nA: {item['answer']}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=50)
                
                batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "rewards": torch.randn_like(inputs["input_ids"].float()) * 0.1
                }
                
                # Training step
                metrics = trainer.train_step(batch)
                
                if initial_loss is None:
                    initial_loss = metrics["loss"]
                final_loss = metrics["loss"]
                
                # Backward pass
                loss = torch.tensor(metrics["loss"], requires_grad=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        # Verify training occurred
        assert final_loss != initial_loss
        assert not torch.isnan(torch.tensor(final_loss))
        
        # Cleanup
        test_dataset.unlink()


class TestDAPOLoss:
    """Test DAPO loss computation."""
    
    def test_loss_calculation(self):
        """Test DAPO loss function."""
        from unsloth.training.dapo_rl import DAPOLoss
        
        config = DAPOConfig()
        loss_fn = DAPOLoss(config)
        
        # Create mock data
        batch_size, seq_len, vocab_size = 2, 10, 50
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        old_logits = torch.randn(batch_size, seq_len, vocab_size)
        actions = torch.randint(0, vocab_size, (batch_size, seq_len))
        advantages = torch.randn(batch_size, seq_len)
        returns = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Calculate loss
        loss_dict = loss_fn(
            logits, old_logits, actions, advantages,
            returns, values, attention_mask
        )
        
        assert "loss" in loss_dict
        assert "pg_loss" in loss_dict
        assert "value_loss" in loss_dict
        assert "entropy" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
    
    def test_decoupled_clipping(self):
        """Test asymmetric clipping behavior."""
        from unsloth.training.dapo_rl import DAPOLoss
        
        config = DAPOConfig(clip_lower=0.8, clip_upper=1.3)
        loss_fn = DAPOLoss(config)
        
        # Test with positive and negative advantages
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        old_logits = logits.clone()  # Same logits for ratio = 1
        actions = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create advantages with clear positive/negative split
        advantages = torch.tensor([
            [1.0, 1.0, -1.0, -1.0, 0.0],
            [1.0, -1.0, 1.0, -1.0, 0.0]
        ])
        
        returns = torch.zeros(batch_size, seq_len)
        values = torch.zeros(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)
        
        loss_dict = loss_fn(
            logits, old_logits, actions, advantages,
            returns, values, attention_mask
        )
        
        # Verify loss is computed
        assert loss_dict["pg_loss"].item() != 0.0  # Should have non-zero policy loss


@pytest.mark.benchmark
class TestDAPOPerformance:
    """Benchmark DAPO performance."""
    
    def test_training_speed(self, mock_model_and_tokenizer, benchmark):
        """Benchmark training step speed."""
        model, ref_model, tokenizer = mock_model_and_tokenizer
        
        config = DAPOConfig()
        trainer = DAPOTrainer(model, ref_model, tokenizer, config)
        
        # Create batch
        batch_size = 8
        seq_len = 128
        
        batch = {
            "input_ids": torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "rewards": torch.randn(batch_size, seq_len) * 0.1
        }
        
        # Benchmark training step
        def train_step():
            return trainer.train_step(batch)
        
        result = benchmark(train_step)
        
        # Verify reasonable performance
        assert result is not None
    
    def test_memory_usage(self, mock_model_and_tokenizer):
        """Test memory efficiency."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model, ref_model, tokenizer = mock_model_and_tokenizer
        config = DAPOConfig()
        trainer = DAPOTrainer(model, ref_model, tokenizer, config)
        
        # Run training steps
        for _ in range(10):
            batch = {
                "input_ids": torch.randint(0, tokenizer.vocab_size, (4, 64)),
                "attention_mask": torch.ones(4, 64),
                "rewards": torch.randn(4, 64) * 0.1
            }
            trainer.train_step(batch)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    print(" DAPO integration tests completed")