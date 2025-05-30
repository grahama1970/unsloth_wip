import torch
from compressed_tensors import FloatQuantizationCompressor, QuantizationConfig

def test_fp8_support():
    try:
        # Step 1: Check for CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. FP8 requires CUDA-enabled GPUs.")
        
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU Detected: {gpu_name}")

        # Step 2: Check GPU compatibility
        gpu_properties = torch.cuda.get_device_properties(0)
        if gpu_properties.major < 8:
            raise RuntimeError("FP8 support requires a GPU with compute capability >= 8.0 (e.g., Hopper architecture).")
        
        # Step 3: Check PyTorch version compatibility
        torch_version = torch.__version__
        if not torch_version >= "2.5.0":
            raise RuntimeError(f"PyTorch version {torch_version} does not support FP8. Upgrade to >= 2.5.0.")
        
        print(f"PyTorch version: {torch_version}")

        # Step 4: Create an FP32 tensor as input
        tensor = torch.randn(10, 10, device="cuda", dtype=torch.float32)
        print("Original Tensor (float32):")
        print(tensor)

        # Step 5: Configure FP8 quantization
        config = QuantizationConfig(
            method="float8",
            format="e4m3fn",
            config_groups={
                "bit_precision": 8,
                "min_value": -4.0,
                "max_value": 4.0
            }
        )
        compressor = FloatQuantizationCompressor(config=config)

        # Step 6: Compress the tensor using FP8
        compressed_tensor = compressor.compress(tensor)
        print("Compressed Tensor (FP8):")
        print(compressed_tensor)

        # Step 7: Decompress the tensor back to FP32
        decompressed_tensor = compressor.decompress(compressed_tensor)
        print("Decompressed Tensor:")
        print(decompressed_tensor)

        # Step 8: Verify accuracy of decompressed tensor
        if torch.allclose(tensor, decompressed_tensor, atol=1e-2):
            print("FP8 compression and decompression are functional.")
        else:
            print("FP8 decompressed tensor differs from the original tensor.")

    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_fp8_support()
