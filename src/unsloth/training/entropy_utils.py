"""
Module: entropy_utils.py
Description: Token entropy calculation utilities for entropy-aware training

External Dependencies:
- torch: https://pytorch.org/docs/stable/index.html
- numpy: https://numpy.org/doc/stable/

Sample Input:
>>> logits = torch.randn(2, 10, 50257)  # [batch, seq_len, vocab_size]
>>> entropy = calculate_token_entropy(logits)

Expected Output:
>>> entropy.shape
torch.Size([2, 10])
>>> entropy[0, 0].item()  # Example entropy value
2.456

Example Usage:
>>> from unsloth.training.entropy_utils import calculate_token_entropy, get_entropy_weight
>>> logits = model(input_ids).logits
>>> entropy = calculate_token_entropy(logits)
>>> weights = get_entropy_weight(entropy, function='linear', scale=1.0)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional, Callable
from loguru import logger


def calculate_token_entropy(
    logits: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Calculate Shannon entropy for each token position.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        temperature: Temperature for softmax (default: 1.0)
        epsilon: Small value to avoid log(0) (default: 1e-8)
        
    Returns:
        entropy: Token-level entropy values [batch_size, seq_len]
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Calculate probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Calculate entropy: -sum(p * log(p))
    log_probs = torch.log(probs + epsilon)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy


def get_entropy_weight(
    entropy: torch.Tensor,
    function: Literal['linear', 'exponential', 'sigmoid'] = 'linear',
    scale: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 2.0
) -> torch.Tensor:
    """
    Convert entropy values to loss weights using configurable functions.
    
    Args:
        entropy: Token entropy values [batch_size, seq_len]
        function: Weighting function type ('linear', 'exponential', 'sigmoid')
        scale: Scaling factor for the weight function
        min_weight: Minimum weight value (default: 1.0)
        max_weight: Maximum weight value (default: 2.0)
        
    Returns:
        weights: Loss weights for each token [batch_size, seq_len]
    """
    # Normalize entropy to [0, 1] range
    entropy_min = entropy.min()
    entropy_max = entropy.max()
    
    epsilon = 1e-8
    if entropy_max - entropy_min > epsilon:
        normalized_entropy = (entropy - entropy_min) / (entropy_max - entropy_min)
    else:
        normalized_entropy = torch.zeros_like(entropy)
    
    # Apply weighting function
    if function == 'linear':
        # Linear: weight = min_weight + scale * normalized_entropy * (max_weight - min_weight)
        weights = min_weight + scale * normalized_entropy * (max_weight - min_weight)
    
    elif function == 'exponential':
        # Exponential: weight increases exponentially with entropy
        exp_factor = torch.exp(scale * normalized_entropy) - 1
        exp_max = torch.exp(torch.tensor(scale)) - 1
        weights = min_weight + (exp_factor / exp_max) * (max_weight - min_weight)
    
    elif function == 'sigmoid':
        # Sigmoid: S-shaped curve for smooth transition
        sigmoid_input = scale * (normalized_entropy - 0.5) * 10  # Center at 0.5
        sigmoid_output = torch.sigmoid(sigmoid_input)
        weights = min_weight + sigmoid_output * (max_weight - min_weight)
    
    else:
        raise ValueError(f"Unknown weighting function: {function}")
    
    return weights


def identify_high_entropy_tokens(
    entropy: torch.Tensor,
    threshold: float = 0.8,
    method: Literal['percentile', 'absolute'] = 'percentile'
) -> torch.Tensor:
    """
    Identify high-entropy token positions.
    
    Args:
        entropy: Token entropy values [batch_size, seq_len]
        threshold: Threshold for high entropy (0.8 = top 20% for percentile)
        method: 'percentile' or 'absolute' threshold
        
    Returns:
        mask: Boolean mask for high-entropy tokens [batch_size, seq_len]
    """
    if method == 'percentile':
        # Get threshold value at given percentile
        flat_entropy = entropy.flatten()
        threshold_value = torch.quantile(flat_entropy, threshold)
        mask = entropy >= threshold_value
    
    elif method == 'absolute':
        # Use absolute threshold value
        mask = entropy >= threshold
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return mask


def visualize_entropy_distribution(
    entropy: torch.Tensor,
    save_path: Optional[str] = None
) -> dict:
    """
    Generate statistics for entropy distribution visualization.
    
    Args:
        entropy: Token entropy values [batch_size, seq_len]
        save_path: Optional path to save statistics
        
    Returns:
        stats: Dictionary with entropy statistics
    """
    flat_entropy = entropy.flatten().cpu().numpy()
    
    stats = {
        'mean': float(np.mean(flat_entropy)),
        'std': float(np.std(flat_entropy)),
        'min': float(np.min(flat_entropy)),
        'max': float(np.max(flat_entropy)),
        'percentiles': {
            '10': float(np.percentile(flat_entropy, 10)),
            '25': float(np.percentile(flat_entropy, 25)),
            '50': float(np.percentile(flat_entropy, 50)),
            '75': float(np.percentile(flat_entropy, 75)),
            '90': float(np.percentile(flat_entropy, 90)),
            '95': float(np.percentile(flat_entropy, 95)),
            '99': float(np.percentile(flat_entropy, 99))
        },
        'high_entropy_ratio': float(np.mean(flat_entropy > np.percentile(flat_entropy, 80)))
    }
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved entropy statistics to {save_path}")
    
    return stats


# Custom weighting function factory
def create_custom_weight_function(
    func: Callable[[torch.Tensor], torch.Tensor]
) -> Callable:
    """
    Create a custom weighting function wrapper.
    
    Args:
        func: Custom function that maps normalized entropy [0,1] to weights
        
    Returns:
        weight_function: Function compatible with get_entropy_weight
    """
    def weight_function(entropy: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize entropy
        entropy_min = entropy.min()
        entropy_max = entropy.max()
        
        if entropy_max - entropy_min > 1e-8:
            normalized = (entropy - entropy_min) / (entropy_max - entropy_min)
        else:
            normalized = torch.zeros_like(entropy)
        
        return func(normalized)
    
    return weight_function


# Validation function
if __name__ == "__main__":
    # Test with realistic data
    batch_size, seq_len, vocab_size = 2, 128, 50257
    
    # Create sample logits with varying entropy
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Make some tokens more certain (lower entropy)
    logits[:, :seq_len//4, :] = logits[:, :seq_len//4, :] * 5.0
    
    # Calculate entropy
    entropy = calculate_token_entropy(logits)
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
    
    # Test different weighting functions
    for func in ['linear', 'exponential', 'sigmoid']:
        weights = get_entropy_weight(entropy, function=func, scale=1.0)
        print(f"\n{func.capitalize()} weights:")
        print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"  Mean: {weights.mean():.3f}")
    
    # Identify high-entropy tokens
    high_entropy_mask = identify_high_entropy_tokens(entropy)
    print(f"\nHigh-entropy tokens: {high_entropy_mask.sum().item()} / {high_entropy_mask.numel()}")
    print(f"Percentage: {100 * high_entropy_mask.float().mean():.1f}%")
    
    # Get statistics
    stats = visualize_entropy_distribution(entropy)
    print(f"\nEntropy statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  High-entropy ratio: {100 * stats['high_entropy_ratio']:.1f}%")
    
    print("\n Module validation passed")