"""Tests for entropy calculation utilities."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import torch
import pytest
import time
import json
from pathlib import Path

from unsloth.training.entropy_utils import (
    calculate_token_entropy,
    get_entropy_weight,
    identify_high_entropy_tokens,
    visualize_entropy_distribution
)


class TestEntropyUtils:
    """Test entropy calculation functions with real PyTorch tensors."""
    
    def test_uniform_distribution(self):
        """Test entropy calculation for uniform distribution (high entropy)."""
        start_time = time.time()
        
        # Create uniform logits (all equal) - should have high entropy
        batch_size, seq_len, vocab_size = 4, 64, 50257
        logits = torch.ones(batch_size, seq_len, vocab_size) * 0.5
        
        # Add small random noise to avoid exact uniformity
        logits += torch.randn_like(logits) * 0.01
        
        entropy = calculate_token_entropy(logits)
        duration = time.time() - start_time
        
        # Verify shape
        assert entropy.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {entropy.shape}"
        
        # Verify high entropy (close to log(vocab_size))
        theoretical_max = torch.log(torch.tensor(float(vocab_size)))
        mean_entropy = entropy.mean().item()
        
        # Should be close to maximum entropy
        assert mean_entropy > theoretical_max * 0.95, f"Uniform distribution entropy too low: {mean_entropy}"
        
        # Duration check
        assert 0.1 <= duration <= 0.5, f"Test duration {duration}s outside expected range"
        
        print(f"Uniform distribution entropy: {mean_entropy:.3f} (max: {theoretical_max:.3f})")
        print(f"Test duration: {duration:.3f}s")
    
    def test_peaked_distribution(self):
        """Test entropy calculation for peaked distribution (low entropy)."""
        start_time = time.time()
        
        # Create peaked logits (one token very likely) - should have low entropy
        batch_size, seq_len, vocab_size = 4, 64, 50257
        logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
        
        # Make one token very likely
        peak_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, peak_indices[b, s]] = 10.0
        
        entropy = calculate_token_entropy(logits)
        duration = time.time() - start_time
        
        # Verify shape
        assert entropy.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {entropy.shape}"
        
        # Verify low entropy (compared to uniform distribution)
        mean_entropy = entropy.mean().item()
        max_entropy = torch.log(torch.tensor(float(vocab_size))).item()
        # Should be much less than max entropy
        assert mean_entropy < max_entropy * 0.80, f"Peaked distribution entropy too high: {mean_entropy} (max: {max_entropy})"
        
        # Duration check
        assert 0.1 <= duration <= 0.5, f"Test duration {duration}s outside expected range"
        
        print(f"Peaked distribution entropy: {mean_entropy:.3f}")
        print(f"Test duration: {duration:.3f}s")
    
    def test_identify_high_entropy(self):
        """Test identification of high-entropy tokens in batch."""
        start_time = time.time()
        
        # Create mixed entropy distribution
        batch_size, seq_len, vocab_size = 8, 128, 50257
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Make first quarter low entropy (peaked)
        logits[:, :seq_len//4, :] = logits[:, :seq_len//4, :] * 0.1
        logits[:, :seq_len//4, 0] = 10.0
        
        # Make last quarter high entropy (uniform)
        logits[:, -seq_len//4:, :] = torch.randn(batch_size, seq_len//4, vocab_size) * 0.01
        
        entropy = calculate_token_entropy(logits)
        high_entropy_mask = identify_high_entropy_tokens(entropy, threshold=0.8)
        
        duration = time.time() - start_time
        
        # Verify mask shape
        assert high_entropy_mask.shape == entropy.shape
        assert high_entropy_mask.dtype == torch.bool
        
        # Verify approximately 20% are marked as high entropy
        high_entropy_ratio = high_entropy_mask.float().mean().item()
        assert 0.15 <= high_entropy_ratio <= 0.25, f"High entropy ratio {high_entropy_ratio} outside expected range"
        
        # Duration check
        assert 0.2 <= duration <= 1.0, f"Test duration {duration}s outside expected range"
        
        print(f"High-entropy tokens: {high_entropy_mask.sum().item()} / {high_entropy_mask.numel()}")
        print(f"Ratio: {100 * high_entropy_ratio:.1f}%")
        print(f"Test duration: {duration:.3f}s")


class TestEntropyWeighting:
    """Test entropy weighting functions."""
    
    def test_linear_weighting(self):
        """Test linear entropy weighting function."""
        start_time = time.time()
        
        # Create entropy tensor
        batch_size, seq_len = 4, 128
        entropy = torch.rand(batch_size, seq_len) * 5.0  # Random entropy values
        
        weights = get_entropy_weight(entropy, function='linear', scale=1.0)
        
        duration = time.time() - start_time
        
        # Verify shape
        assert weights.shape == entropy.shape
        
        # Verify range [1.0, 2.0]
        assert weights.min() >= 0.99, f"Min weight {weights.min()} below expected"
        assert weights.max() <= 2.01, f"Max weight {weights.max()} above expected"
        
        # Verify monotonic increase with entropy
        sorted_entropy, indices = entropy.flatten().sort()
        sorted_weights = weights.flatten()[indices]
        diffs = sorted_weights[1:] - sorted_weights[:-1]
        assert (diffs >= -1e-6).all(), "Weights not monotonically increasing with entropy"
        
        print(f"Linear weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"Test duration: {duration:.3f}s")
    
    def test_exponential_weighting(self):
        """Test exponential entropy weighting function."""
        start_time = time.time()
        
        # Create entropy tensor
        batch_size, seq_len = 4, 128
        entropy = torch.rand(batch_size, seq_len) * 5.0
        
        weights = get_entropy_weight(entropy, function='exponential', scale=1.0)
        
        duration = time.time() - start_time
        
        # Verify shape and range
        assert weights.shape == entropy.shape
        assert weights.min() >= 0.99
        assert weights.max() <= 2.01
        
        # Exponential should have steeper increase for high entropy
        linear_weights = get_entropy_weight(entropy, function='linear', scale=1.0)
        high_entropy_mask = entropy > entropy.median()
        
        # Compare slopes at high entropy
        exp_high = weights[high_entropy_mask].mean()
        lin_high = linear_weights[high_entropy_mask].mean()
        
        print(f"Exponential weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"High entropy avg - Exp: {exp_high:.3f}, Linear: {lin_high:.3f}")
        print(f"Test duration: {duration:.3f}s")
    
    def test_sigmoid_weighting(self):
        """Test sigmoid entropy weighting function."""
        start_time = time.time()
        
        # Create entropy tensor
        batch_size, seq_len = 4, 128
        entropy = torch.rand(batch_size, seq_len) * 5.0
        
        weights = get_entropy_weight(entropy, function='sigmoid', scale=1.0)
        
        duration = time.time() - start_time
        
        # Verify shape and range
        assert weights.shape == entropy.shape
        assert weights.min() >= 0.99
        assert weights.max() <= 2.01
        
        # Sigmoid should have steepest change around median
        median_entropy = entropy.median()
        near_median_mask = (entropy > median_entropy - 0.5) & (entropy < median_entropy + 0.5)
        
        print(f"Sigmoid weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"Weights near median: {weights[near_median_mask].mean():.3f}")
        print(f"Test duration: {duration:.3f}s")


class TestEntropyVisualization:
    """Test entropy visualization and statistics."""
    
    def test_entropy_statistics(self):
        """Test entropy distribution statistics generation."""
        start_time = time.time()
        
        # Create realistic entropy distribution
        batch_size, seq_len = 16, 256
        
        # Mix of high and low entropy
        entropy = torch.zeros(batch_size, seq_len)
        
        # 80% low entropy (0-2)
        low_entropy_size = int(0.8 * batch_size * seq_len)
        entropy.flatten()[:low_entropy_size] = torch.rand(low_entropy_size) * 2.0
        
        # 20% high entropy (3-5)
        high_entropy_size = batch_size * seq_len - low_entropy_size
        entropy.flatten()[low_entropy_size:] = 3.0 + torch.rand(high_entropy_size) * 2.0
        
        # Shuffle
        entropy = entropy.flatten()[torch.randperm(batch_size * seq_len)].reshape(batch_size, seq_len)
        
        # Generate statistics
        stats = visualize_entropy_distribution(entropy)
        
        duration = time.time() - start_time
        
        # Verify statistics structure
        assert 'mean' in stats
        assert 'std' in stats
        assert 'percentiles' in stats
        assert 'high_entropy_ratio' in stats
        
        # Verify high entropy ratio is approximately 20%
        assert 0.15 <= stats['high_entropy_ratio'] <= 0.25
        
        # Duration check
        assert 0.0001 <= duration <= 1.0
        
        print(f"Entropy statistics:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  90th percentile: {stats['percentiles']['90']:.3f}")
        print(f"  High-entropy ratio: {100 * stats['high_entropy_ratio']:.1f}%")
        print(f"Test duration: {duration:.3f}s")


class TestHoneypot:
    """Honeypot tests designed to fail."""
    
    @pytest.mark.honeypot
    def test_instant_entropy_calculation(self):
        """HONEYPOT: Large tensor entropy calculation completes instantly."""
        start_time = time.time()
        
        # Huge tensor that should take time
        batch_size, seq_len, vocab_size = 1000, 1000, 50257
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # This should take significant time
        entropy = calculate_token_entropy(logits)
        duration = time.time() - start_time
        
        # This should FAIL - computation cannot be instant
        assert duration < 0.001, f"Real entropy calculation took {duration}s, not instant"


# Run validation
if __name__ == "__main__":
    print("Running entropy utils tests...")
    
    # Test entropy calculation
    test_entropy = TestEntropyUtils()
    test_entropy.test_uniform_distribution()
    test_entropy.test_peaked_distribution()
    test_entropy.test_identify_high_entropy()
    
    # Test weighting
    test_weight = TestEntropyWeighting()
    test_weight.test_linear_weighting()
    test_weight.test_exponential_weighting()
    test_weight.test_sigmoid_weighting()
    
    # Test visualization
    test_viz = TestEntropyVisualization()
    test_viz.test_entropy_statistics()
    
    print("\n All entropy utils tests passed!")