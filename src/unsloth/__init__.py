"""
Unsloth: Fast and memory-efficient fine-tuning

Module: __init__.py
Description: Package initialization and exports
"""

__version__ = "0.1.0"

# Mock FastLanguageModel for testing
class FastLanguageModel:
    """Mock FastLanguageModel for testing purposes"""

    @staticmethod
    def from_pretrained(*args, **kwargs):
        """Mock from_pretrained method"""
        class MockModel:
            def __init__(self):
                self.config = type('Config', (), {'max_seq_length': 2048})()

        class MockTokenizer:
            def __init__(self):
                self.pad_token = "[PAD]"
                self.padding_side = "left"

        return MockModel(), MockTokenizer()

    @staticmethod
    def get_peft_model(*args, **kwargs):
        """Mock get_peft_model method"""
        return args[0] if args else None


# Re-export commonly used items
__all__ = ["FastLanguageModel", "__version__"]
