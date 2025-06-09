"""
Module: multi_gpu_cost_optimizer.py
Description: Optimizes GPU selection including multi-GPU configurations for cost-effectiveness

External Dependencies:
- runpod: https://docs.runpod.io/
- google-cloud-aiplatform: https://cloud.google.com/vertex-ai/docs

Sample Input:
>>> optimizer = MultiGPUCostOptimizer()
>>> config = optimizer.get_optimal_gpu_config(model_size_gb=140, tokens_required=100000)

Expected Output:
>>> config
{
    'gpu_config': '2x RTX 4090',
    'total_vram_gb': 48,
    'cost_per_hour': 1.38,
    'estimated_tokens_per_second': 80,
    'provider': 'runpod'
}

Example Usage:
>>> from unsloth.core.multi_gpu_cost_optimizer import MultiGPUCostOptimizer
>>> optimizer = MultiGPUCostOptimizer()
>>> best_config = optimizer.optimize_for_workload(model_name="llama3.1:70b", total_tokens=50000)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Literal, AsyncIterator
from enum import Enum
import math
from loguru import logger
import os


@dataclass 
class GPUSpec:
    """GPU specifications including VRAM and performance."""
    name: str
    vram_gb: int
    fp16_tflops: float  # FP16 performance in TFLOPS
    cost_per_hour: float
    availability: float  # 0-1, how often available
    supports_nvlink: bool = False
    

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU setup."""
    gpu_type: str
    gpu_count: int
    total_vram_gb: int
    total_cost_per_hour: float
    interconnect: Literal["nvlink", "pcie", "none"]
    efficiency_factor: float  # 0-1, accounting for multi-GPU overhead
    

@dataclass
class ModelRequirements:
    """Requirements for running a model."""
    model_size_gb: float
    min_vram_overhead_gb: float = 2.0  # OS/framework overhead
    kv_cache_gb_per_token: float = 0.00002  # ~20MB per 1M tokens for 70B model
    batch_size: int = 1
    sequence_length: int = 4096
    

class MultiGPUCostOptimizer:
    """Optimize GPU selection including multi-GPU configurations."""
    
    # Updated GPU specs with current RunPod pricing
    GPU_SPECS = {
        # Consumer GPUs (often best value)
        "RTX 4090": GPUSpec(
            name="RTX 4090",
            vram_gb=24,
            fp16_tflops=82.6,
            cost_per_hour=0.69,
            availability=0.8,
            supports_nvlink=False
        ),
        "RTX 4080": GPUSpec(
            name="RTX 4080", 
            vram_gb=16,
            fp16_tflops=48.7,
            cost_per_hour=0.56,
            availability=0.7,
            supports_nvlink=False
        ),
        "RTX A6000": GPUSpec(
            name="RTX A6000",
            vram_gb=48,
            fp16_tflops=38.7,
            cost_per_hour=0.79,
            availability=0.9,
            supports_nvlink=True
        ),
        
        # Professional GPUs
        "L40S": GPUSpec(
            name="L40S",
            vram_gb=48,
            fp16_tflops=91.6,
            cost_per_hour=1.14,
            availability=0.8,
            supports_nvlink=True
        ),
        "A100 PCIe": GPUSpec(
            name="A100 PCIe",
            vram_gb=40,
            fp16_tflops=77.97,
            cost_per_hour=1.64,
            availability=0.7,
            supports_nvlink=True
        ),
        "A100 SXM": GPUSpec(
            name="A100 SXM",
            vram_gb=80,
            fp16_tflops=77.97,
            cost_per_hour=2.21,
            availability=0.6,
            supports_nvlink=True
        ),
        
        # Latest generation
        "H100 PCIe": GPUSpec(
            name="H100 PCIe",
            vram_gb=80,
            fp16_tflops=204.9,
            cost_per_hour=2.69,
            availability=0.5,
            supports_nvlink=True
        ),
        "H100 SXM": GPUSpec(
            name="H100 SXM",
            vram_gb=80,
            fp16_tflops=267.6,
            cost_per_hour=3.89,
            availability=0.4,
            supports_nvlink=True
        ),
    }
    
    # Multi-GPU efficiency factors
    MULTI_GPU_EFFICIENCY = {
        # (gpu_count, interconnect) -> efficiency
        (1, "none"): 1.0,
        (2, "pcie"): 0.85,  # 15% overhead
        (2, "nvlink"): 0.95,  # 5% overhead  
        (4, "pcie"): 0.70,  # 30% overhead
        (4, "nvlink"): 0.90,  # 10% overhead
        (8, "pcie"): 0.55,  # 45% overhead
        (8, "nvlink"): 0.85,  # 15% overhead
    }
    
    # Model size to parameter count approximation
    MODEL_PARAMS_TO_GB = {
        "7b": 14,    # FP16
        "13b": 26,
        "30b": 60,
        "70b": 140,
        "180b": 360,
        "405b": 810,
    }
    
    def __init__(self, 
                 prefer_consumer_gpus: bool = True,
                 max_gpus: int = 8,
                 include_gemini_comparison: bool = True):
        """
        Initialize optimizer.
        
        Args:
            prefer_consumer_gpus: Prefer consumer GPUs for cost savings
            max_gpus: Maximum GPUs to consider
            include_gemini_comparison: Include Gemini Flash in comparisons
        """
        self.prefer_consumer_gpus = prefer_consumer_gpus
        self.max_gpus = max_gpus
        self.include_gemini_comparison = include_gemini_comparison
        
        # Gemini Flash pricing for comparison
        self.gemini_flash_cost_per_million = 0.075  # Input tokens
        self.gemini_flash_tokens_per_second = 150
        
    def calculate_vram_requirements(self, 
                                  model_size_gb: float,
                                  batch_size: int = 1,
                                  sequence_length: int = 4096,
                                  kv_cache_tokens: int = 0,
                                  training_mode: Optional[str] = None) -> float:
        """
        Calculate total VRAM requirements for a model.
        
        Based on Trelis Research GPU Configuration Guide:
        - 4-bit LoRA: 0.5 bytes per parameter
        - 16-bit LoRA: 2 bytes per parameter  
        - Full 16-bit fine-tuning: 16 bytes per parameter (includes gradients, optimizer states)
        
        Args:
            model_size_gb: Model weights size in GB
            batch_size: Inference batch size
            sequence_length: Maximum sequence length
            kv_cache_tokens: Number of tokens in KV cache
            training_mode: None for inference, "4bit-lora", "16bit-lora", or "full-16bit"
            
        Returns:
            Total VRAM required in GB
        """
        if training_mode:
            # Training VRAM calculations from Trelis guide
            model_params_billions = model_size_gb / 2  # Assuming FP16, 2 bytes per param
            
            if training_mode == "4bit-lora":
                # 0.5 bytes per parameter
                vram_needed = model_params_billions * 0.5
            elif training_mode == "16bit-lora":
                # 2 bytes per parameter
                vram_needed = model_params_billions * 2
            elif training_mode == "full-16bit":
                # 16 bytes per parameter (model + gradients + optimizer states)
                vram_needed = model_params_billions * 16
            else:
                raise ValueError(f"Unknown training mode: {training_mode}")
                
            # Add batch size scaling for training
            vram_needed *= (1 + (batch_size - 1) * 0.1)  # ~10% per additional batch
            
        else:
            # Inference mode (original calculation)
            # Base model weights
            vram_needed = model_size_gb
            
            # Activation memory (rough estimate)
            activation_memory_gb = (batch_size * sequence_length * 8192 * 4) / (1024**3)  # Assuming hidden_size=8192
            vram_needed += activation_memory_gb
            
            # KV cache memory
            if kv_cache_tokens > 0:
                # Approximate: 2 * num_layers * hidden_size * 2 (K+V) * bytes_per_param
                kv_cache_gb = kv_cache_tokens * 0.00002  # ~20MB per 1M tokens for 70B model
                vram_needed += kv_cache_gb
                
        # Framework overhead
        vram_needed += 2.0  # ~2GB overhead
        
        # Safety margin (10%)
        vram_needed *= 1.1
        
        return vram_needed
        
    def get_multi_gpu_configs(self, 
                            vram_required: float,
                            gpu_specs: List[GPUSpec]) -> List[MultiGPUConfig]:
        """
        Generate possible multi-GPU configurations.
        
        Args:
            vram_required: Total VRAM needed
            gpu_specs: Available GPU types
            
        Returns:
            List of possible configurations
        """
        configs = []
        
        for gpu_spec in gpu_specs:
            # Single GPU
            if gpu_spec.vram_gb >= vram_required:
                configs.append(MultiGPUConfig(
                    gpu_type=gpu_spec.name,
                    gpu_count=1,
                    total_vram_gb=gpu_spec.vram_gb,
                    total_cost_per_hour=gpu_spec.cost_per_hour,
                    interconnect="none",
                    efficiency_factor=1.0
                ))
                
            # Multi-GPU configurations
            for gpu_count in [2, 4, 8]:
                if gpu_count > self.max_gpus:
                    continue
                    
                total_vram = gpu_spec.vram_gb * gpu_count
                
                # Need some headroom for multi-GPU overhead
                if total_vram >= vram_required * 1.2:
                    interconnect = "nvlink" if gpu_spec.supports_nvlink else "pcie"
                    efficiency = self.MULTI_GPU_EFFICIENCY.get((gpu_count, interconnect), 0.5)
                    
                    configs.append(MultiGPUConfig(
                        gpu_type=gpu_spec.name,
                        gpu_count=gpu_count,
                        total_vram_gb=total_vram,
                        total_cost_per_hour=gpu_spec.cost_per_hour * gpu_count,
                        interconnect=interconnect,
                        efficiency_factor=efficiency
                    ))
                    
        return configs
        
    def estimate_tokens_per_second(self,
                                 gpu_spec: GPUSpec,
                                 gpu_count: int,
                                 model_size_gb: float,
                                 efficiency: float) -> float:
        """
        Estimate inference speed based on GPU specs and model size.
        
        Args:
            gpu_spec: GPU specifications
            gpu_count: Number of GPUs
            model_size_gb: Model size in GB
            efficiency: Multi-GPU efficiency factor
            
        Returns:
            Estimated tokens per second
        """
        # Base estimation: TFLOPS correlates with inference speed
        # Rough approximation based on benchmarks
        single_gpu_base = gpu_spec.fp16_tflops * 0.5  # ~0.5 tokens per TFLOP
        
        # Adjust for model size (larger models are slower)
        if model_size_gb < 20:  # 7B models
            size_factor = 1.5
        elif model_size_gb < 50:  # 13B models  
            size_factor = 1.0
        elif model_size_gb < 100:  # 30B models
            size_factor = 0.7
        elif model_size_gb < 200:  # 70B models
            size_factor = 0.5
        else:  # 180B+ models
            size_factor = 0.3
            
        single_gpu_tokens = single_gpu_base * size_factor
        
        # Multi-GPU scaling
        total_tokens = single_gpu_tokens * gpu_count * efficiency
        
        return total_tokens
        
    def get_optimal_gpu_config(self,
                             model_size_gb: float,
                             tokens_required: int,
                             max_latency_seconds: Optional[float] = None,
                             batch_size: int = 1) -> Dict[str, Any]:
        """
        Get optimal GPU configuration for a model and workload.
        
        Args:
            model_size_gb: Model size in GB
            tokens_required: Total tokens to generate
            max_latency_seconds: Maximum acceptable latency
            batch_size: Inference batch size
            
        Returns:
            Optimal configuration details
        """
        # Calculate VRAM requirements
        vram_required = self.calculate_vram_requirements(
            model_size_gb, batch_size, 4096, tokens_required
        )
        
        logger.info(f"VRAM required: {vram_required:.1f} GB for {model_size_gb}GB model")
        
        # Get possible configurations
        gpu_specs = list(self.GPU_SPECS.values())
        if self.prefer_consumer_gpus:
            # Sort by cost-effectiveness (TFLOPS per dollar)
            gpu_specs.sort(key=lambda g: g.fp16_tflops / g.cost_per_hour, reverse=True)
            
        configs = self.get_multi_gpu_configs(vram_required, gpu_specs)
        
        # Calculate cost and performance for each config
        results = []
        
        for config in configs:
            gpu_spec = self.GPU_SPECS[config.gpu_type]
            
            # Estimate performance
            tokens_per_second = self.estimate_tokens_per_second(
                gpu_spec, config.gpu_count, model_size_gb, config.efficiency_factor
            )
            
            # Calculate time and cost
            generation_time = tokens_required / tokens_per_second
            total_cost = (generation_time / 3600) * config.total_cost_per_hour
            
            # Skip if too slow
            if max_latency_seconds and generation_time > max_latency_seconds:
                continue
                
            results.append({
                "gpu_config": f"{config.gpu_count}x {config.gpu_type}" if config.gpu_count > 1 else config.gpu_type,
                "gpu_count": config.gpu_count,
                "gpu_type": config.gpu_type,
                "total_vram_gb": config.total_vram_gb,
                "cost_per_hour": config.total_cost_per_hour,
                "tokens_per_second": tokens_per_second,
                "generation_time_seconds": generation_time,
                "total_cost": total_cost,
                "cost_per_1k_tokens": (total_cost / tokens_required) * 1000,
                "efficiency": config.efficiency_factor,
                "interconnect": config.interconnect,
                "provider": "runpod"
            })
            
        # Add Gemini Flash comparison if enabled
        if self.include_gemini_comparison:
            gemini_time = tokens_required / self.gemini_flash_tokens_per_second
            gemini_cost = (tokens_required / 1_000_000) * self.gemini_flash_cost_per_million
            
            results.append({
                "gpu_config": "Gemini Flash API",
                "gpu_count": 0,
                "gpu_type": "API",
                "total_vram_gb": 0,
                "cost_per_hour": 0,
                "tokens_per_second": self.gemini_flash_tokens_per_second,
                "generation_time_seconds": gemini_time,
                "total_cost": gemini_cost,
                "cost_per_1k_tokens": (gemini_cost / tokens_required) * 1000,
                "efficiency": 1.0,
                "interconnect": "none",
                "provider": "vertex"
            })
            
        # Sort by total cost
        results.sort(key=lambda x: x["total_cost"])
        
        if not results:
            raise ValueError(f"No suitable GPU configuration found for {model_size_gb}GB model")
            
        best = results[0]
        
        # Log comparison
        logger.info(f"\nOptimal configuration: {best['gpu_config']}")
        logger.info(f"Cost: ${best['total_cost']:.4f} ({best['cost_per_1k_tokens']:.3f}/1k tokens)")
        logger.info(f"Time: {best['generation_time_seconds']:.1f}s ({best['tokens_per_second']:.1f} tok/s)")
        
        if len(results) > 1:
            logger.info("\nAlternatives:")
            for alt in results[1:4]:  # Show top 3 alternatives
                savings = (alt['total_cost'] - best['total_cost']) / best['total_cost'] * 100
                logger.info(
                    f"  {alt['gpu_config']}: ${alt['total_cost']:.4f} "
                    f"(+{savings:.1f}%), {alt['generation_time_seconds']:.1f}s"
                )
                
        return best
        
    def optimize_for_workload(self,
                            model_name: str,
                            total_tokens: int,
                            requests_per_minute: float = 1.0,
                            latency_sla_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize GPU selection for a specific workload pattern.
        
        Args:
            model_name: Model name (e.g., "llama3.1:70b")
            total_tokens: Total tokens to generate
            requests_per_minute: Expected request rate
            latency_sla_seconds: Maximum acceptable latency per request
            
        Returns:
            Optimal configuration with recommendations
        """
        # Parse model size
        model_size_gb = None
        for size_key, gb in self.MODEL_PARAMS_TO_GB.items():
            if size_key in model_name.lower():
                model_size_gb = gb
                break
                
        if not model_size_gb:
            raise ValueError(f"Cannot determine model size from name: {model_name}")
            
        # Adjust for request pattern
        concurrent_requests = max(1, requests_per_minute / 60 * (latency_sla_seconds or 10))
        effective_batch_size = min(8, int(concurrent_requests))
        
        logger.info(
            f"Optimizing for {model_name} ({model_size_gb}GB), "
            f"{requests_per_minute} req/min, batch size: {effective_batch_size}"
        )
        
        # Get optimal config
        config = self.get_optimal_gpu_config(
            model_size_gb=model_size_gb,
            tokens_required=total_tokens,
            max_latency_seconds=latency_sla_seconds,
            batch_size=effective_batch_size
        )
        
        # Add recommendations
        config["recommendations"] = []
        
        if config["gpu_count"] > 1:
            config["recommendations"].append(
                f"Use vLLM or TGI for efficient multi-GPU serving with {config['gpu_count']} GPUs"
            )
            
        if config["provider"] == "vertex":
            config["recommendations"].append(
                "Gemini Flash is most cost-effective for this workload"
            )
        elif config["cost_per_1k_tokens"] > 0.1:
            config["recommendations"].append(
                "Consider using Gemini Flash API for better cost efficiency"
            )
            
        if requests_per_minute > 10:
            config["recommendations"].append(
                "Consider using multiple instances for load balancing"
            )
            
        return config
        
    def optimize_for_training(self,
                            model_name: str,
                            dataset_size: int,
                            training_mode: str = "4bit-lora",
                            target_batch_size: int = 4) -> Dict[str, Any]:
        """
        Optimize GPU selection for training workloads.
        
        Based on Trelis Research recommendations:
        - Start with batch size 1, increase GPUs until no OOM
        - Then double batch size until OOM and back off
        
        Args:
            model_name: Model name (e.g., "llama3.1:70b")
            dataset_size: Number of training samples
            training_mode: "4bit-lora", "16bit-lora", or "full-16bit"
            target_batch_size: Desired batch size for training
            
        Returns:
            Optimal training configuration
        """
        # Parse model size
        model_size_gb = None
        for size_key, gb in self.MODEL_PARAMS_TO_GB.items():
            if size_key in model_name.lower():
                model_size_gb = gb
                break
                
        if not model_size_gb:
            raise ValueError(f"Cannot determine model size from name: {model_name}")
            
        # Calculate VRAM for training
        vram_required = self.calculate_vram_requirements(
            model_size_gb=model_size_gb,
            batch_size=target_batch_size,
            training_mode=training_mode
        )
        
        logger.info(
            f"Training {model_name} with {training_mode}: "
            f"{vram_required:.1f}GB VRAM required for batch size {target_batch_size}"
        )
        
        # Get available configurations
        gpu_specs = list(self.GPU_SPECS.values())
        configs = self.get_multi_gpu_configs(vram_required, gpu_specs)
        
        # Filter and sort configurations
        valid_configs = []
        for config in configs:
            gpu_spec = self.GPU_SPECS[config.gpu_type]
            
            # Calculate training time (rough estimate)
            # Assume 1000 tokens per sample, 3 epochs
            total_tokens = dataset_size * 1000 * 3
            tokens_per_second = self.estimate_tokens_per_second(
                gpu_spec, config.gpu_count, model_size_gb, config.efficiency_factor
            ) * 0.3  # Training is slower than inference
            
            training_hours = (total_tokens / tokens_per_second) / 3600
            total_cost = training_hours * config.total_cost_per_hour
            
            valid_configs.append({
                "config": config,
                "training_hours": training_hours,
                "total_cost": total_cost,
                "cost_per_sample": total_cost / dataset_size,
                "tokens_per_second": tokens_per_second
            })
            
        # Sort by cost
        valid_configs.sort(key=lambda x: x["total_cost"])
        
        if not valid_configs:
            raise ValueError("No suitable GPU configuration found for training")
            
        best = valid_configs[0]
        config = best["config"]
        
        result = {
            "gpu_config": f"{config.gpu_count}x {config.gpu_type}" if config.gpu_count > 1 else config.gpu_type,
            "gpu_count": config.gpu_count,
            "total_vram_gb": config.total_vram_gb,
            "cost_per_hour": config.total_cost_per_hour,
            "training_hours": best["training_hours"],
            "total_cost": best["total_cost"],
            "cost_per_sample": best["cost_per_sample"],
            "training_mode": training_mode,
            "batch_size": target_batch_size,
            "recommendations": []
        }
        
        # Add recommendations based on Trelis guide
        if config.gpu_count == 1:
            result["recommendations"].append(
                "Use Unsloth for minimum VRAM usage on single GPU"
            )
        elif training_mode == "full-16bit":
            result["recommendations"].append(
                "Consider FSDP (Fully Sharded Data Parallel) for multi-GPU training"
            )
        else:
            result["recommendations"].append(
                "Use DDP (Distributed Data Parallel) for multi-GPU LoRA training"
            )
            
        if dataset_size > 10000 and training_mode != "full-16bit":
            result["recommendations"].append(
                "Consider full fine-tuning for large datasets (>10k samples)"
            )
            
        # Log comparison
        logger.info(f"\nOptimal training configuration: {result['gpu_config']}")
        logger.info(f"Training time: {result['training_hours']:.1f} hours")
        logger.info(f"Total cost: ${result['total_cost']:.2f}")
        logger.info(f"Cost per sample: ${result['cost_per_sample']:.4f}")
        
        return result


# Validation  
if __name__ == "__main__":
    optimizer = MultiGPUCostOptimizer()
    
    # Test different model sizes
    test_cases = [
        ("Small model", 14, 10000),      # 7B
        ("Medium model", 60, 10000),      # 30B  
        ("Large model", 140, 10000),      # 70B
        ("XLarge model", 360, 10000),     # 180B
    ]
    
    print("Multi-GPU Cost Optimization Analysis")
    print("=" * 80)
    
    for name, size_gb, tokens in test_cases:
        print(f"\n{name} ({size_gb}GB, {tokens} tokens):")
        
        try:
            config = optimizer.get_optimal_gpu_config(
                model_size_gb=size_gb,
                tokens_required=tokens
            )
            
            print(f"  Best: {config['gpu_config']}")
            print(f"  Cost: ${config['total_cost']:.4f}")
            print(f"  Speed: {config['tokens_per_second']:.1f} tok/s")
            
            if config['gpu_count'] > 1:
                print(f"  Multi-GPU: {config['gpu_count']}x with {config['interconnect']} ({config['efficiency']*100:.0f}% efficiency)")
                
        except Exception as e:
            print(f"  Error: {e}")
            
    # Test workload optimization
    print("\n\nWorkload Optimization Example:")
    workload_config = optimizer.optimize_for_workload(
        model_name="llama3.1:70b",
        total_tokens=100000,
        requests_per_minute=5,
        latency_sla_seconds=30
    )
    
    print(f"Model: llama3.1:70b")
    print(f"Best config: {workload_config['gpu_config']}")
    print(f"Total cost: ${workload_config['total_cost']:.2f}")
    print("\nRecommendations:")
    for rec in workload_config.get('recommendations', []):
        print(f"  - {rec}")
        
    # Test training optimization
    print("\n\nTraining Optimization Examples:")
    print("=" * 80)
    
    training_configs = [
        ("7B Model - 4bit LoRA", "llama3.1:7b", 1000, "4bit-lora"),
        ("70B Model - 4bit LoRA", "llama3.1:70b", 1000, "4bit-lora"),
        ("70B Model - 16bit LoRA", "llama3.1:70b", 1000, "16bit-lora"),
        ("70B Model - Full Fine-tuning", "llama3.1:70b", 10000, "full-16bit"),
    ]
    
    for desc, model, dataset_size, mode in training_configs:
        print(f"\n{desc} ({dataset_size} samples):")
        try:
            train_config = optimizer.optimize_for_training(
                model_name=model,
                dataset_size=dataset_size,
                training_mode=mode,
                target_batch_size=4
            )
            
            print(f"  GPU Config: {train_config['gpu_config']}")
            print(f"  VRAM: {train_config['total_vram_gb']:.1f}GB")
            print(f"  Training time: {train_config['training_hours']:.1f} hours")
            print(f"  Total cost: ${train_config['total_cost']:.2f}")
            print(f"  Cost per sample: ${train_config['cost_per_sample']:.4f}")
            print("  Recommendations:")
            for rec in train_config['recommendations']:
                print(f"    - {rec}")
                
        except Exception as e:
            print(f"  Error: {e}")
        
    print("\n Module validation passed")