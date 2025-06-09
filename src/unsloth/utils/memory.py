"""Memory management utilities."""
Module: memory.py
Description: Functions for memory operations

import gc

import torch
from loguru import logger


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.debug("Memory cleared")


def get_memory_stats() -> dict:
    """Get current memory statistics."""
    stats = {
        "cuda_available": torch.cuda.is_available()
    }

    if torch.cuda.is_available():
        stats.update({
            "cuda_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cuda_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            "cuda_max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_device_count": torch.cuda.device_count()
        })

    return stats


def log_memory_usage(prefix: str = ""):
    """Log current memory usage."""
    stats = get_memory_stats()

    if prefix:
        prefix = f"{prefix} - "

    if stats["cuda_available"]:
        logger.info(
            f"{prefix}GPU Memory: "
            f"{stats['cuda_allocated']:.2f}/{stats['cuda_reserved']:.2f} GB allocated/reserved"
        )
    else:
        logger.info(f"{prefix}No GPU available")
