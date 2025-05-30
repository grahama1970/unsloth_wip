import torch
import gc

def clear_memory():
    """Clear CUDA cache and garbage collect."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


if __name__ == "__main__":
    clear_memory()
