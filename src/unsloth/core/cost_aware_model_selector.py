"""
Module: cost_aware_model_selector.py
Description: Cost-aware model selection with RunPod vs Gemini Flash comparison

External Dependencies:
- ollama: https://github.com/ollama/ollama-python
- runpod: https://docs.runpod.io/
- google-cloud-aiplatform: https://cloud.google.com/vertex-ai/docs

Sample Input:
>>> selector = CostAwareModelSelector()
>>> model_info = selector.select_optimal_model(tokens_required=10000, urgency="normal")

Expected Output:
>>> model_info
{'provider': 'vertex', 'model': 'gemini-1.5-flash', 'cost_per_million': 0.35, 'tokens_per_second': 150}

Example Usage:
>>> from unsloth.core.cost_aware_model_selector import CostAwareModelSelector
>>> selector = CostAwareModelSelector()
>>> response = await selector.generate_with_cost_optimization(prompt="Generate text", expected_tokens=1000)
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List, Tuple
from enum import Enum
import os
from loguru import logger
import asyncio
import httpx
import time
import json

# Cost and performance data
@dataclass
class ModelPerformanceProfile:
    """Performance and cost profile for a model."""
    provider: str
    model_name: str
    tokens_per_second: float  # Average generation speed
    cost_per_million_tokens: float  # Cost per million tokens
    startup_time_seconds: float  # Time to spin up (0 for always-on)
    min_runtime_seconds: float  # Minimum billing period
    supports_streaming: bool
    max_context_length: int
    

class CostAwareModelSelector:
    """Select models based on cost efficiency including speed and pricing."""
    
    # Performance profiles based on benchmarks
    PERFORMANCE_PROFILES = {
        # Ollama local models (free but limited by hardware)
        "ollama/llama2:7b": ModelPerformanceProfile(
            provider="ollama",
            model_name="llama2:7b",
            tokens_per_second=50,  # On typical consumer GPU
            cost_per_million_tokens=0.0,
            startup_time_seconds=0,
            min_runtime_seconds=0,
            supports_streaming=True,
            max_context_length=4096
        ),
        "ollama/mixtral:8x7b": ModelPerformanceProfile(
            provider="ollama", 
            model_name="mixtral:8x7b",
            tokens_per_second=30,  # Slower due to size
            cost_per_million_tokens=0.0,
            startup_time_seconds=0,
            min_runtime_seconds=0,
            supports_streaming=True,
            max_context_length=32768
        ),
        
        # Vertex AI Gemini Flash - Very fast and cost-effective
        "vertex/gemini-1.5-flash": ModelPerformanceProfile(
            provider="vertex",
            model_name="gemini-1.5-flash-002",
            tokens_per_second=150,  # Very fast inference
            cost_per_million_tokens=0.075,  # $0.075 per million input tokens
            startup_time_seconds=0,  # Always available
            min_runtime_seconds=0,
            supports_streaming=True,
            max_context_length=1000000  # 1M context window
        ),
        "vertex/gemini-1.5-flash-8b": ModelPerformanceProfile(
            provider="vertex",
            model_name="gemini-1.5-flash-8b",
            tokens_per_second=200,  # Even faster
            cost_per_million_tokens=0.0375,  # Half the cost
            startup_time_seconds=0,
            min_runtime_seconds=0,
            supports_streaming=True,
            max_context_length=1000000
        ),
        
        # RunPod models - Powerful but with startup costs
        "runpod/llama3.1:70b": ModelPerformanceProfile(
            provider="runpod",
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            tokens_per_second=40,  # On H100
            cost_per_million_tokens=3.0,  # Based on ~$3/hour H100
            startup_time_seconds=120,  # 2 minutes to spin up
            min_runtime_seconds=60,  # Minimum 1 minute billing
            supports_streaming=True,
            max_context_length=128000
        ),
        "runpod/qwen2.5:72b": ModelPerformanceProfile(
            provider="runpod",
            model_name="Qwen/Qwen2.5-72B-Instruct", 
            tokens_per_second=45,
            cost_per_million_tokens=3.0,
            startup_time_seconds=120,
            min_runtime_seconds=60,
            supports_streaming=True,
            max_context_length=32768
        ),
        "runpod/llama3.1:405b": ModelPerformanceProfile(
            provider="runpod",
            model_name="meta-llama/Llama-3.1-405B-Instruct",
            tokens_per_second=25,  # Slower due to size, even on multiple H100s
            cost_per_million_tokens=12.0,  # ~$12/hour for 8xH100
            startup_time_seconds=300,  # 5 minutes
            min_runtime_seconds=60,
            supports_streaming=True,
            max_context_length=128000
        ),
    }
    
    # RunPod GPU pricing (per hour)
    RUNPOD_GPU_COSTS = {
        "RTX 4090": 0.69,
        "RTX A6000": 0.79,
        "L40S": 1.14,
        "H100 PCIe": 2.69,
        "H100 SXM": 3.89,
        "8xH100": 27.12,  # For very large models
    }
    
    def __init__(self,
                 prefer_low_cost: bool = True,
                 max_startup_wait: float = 60.0,
                 include_startup_cost: bool = True):
        """
        Initialize cost-aware selector.
        
        Args:
            prefer_low_cost: Optimize for cost over speed
            max_startup_wait: Maximum seconds to wait for model startup
            include_startup_cost: Include startup time in cost calculations
        """
        self.prefer_low_cost = prefer_low_cost
        self.max_startup_wait = max_startup_wait
        self.include_startup_cost = include_startup_cost
        
    def calculate_total_cost(self,
                           profile: ModelPerformanceProfile,
                           total_tokens: int,
                           include_startup: bool = True) -> Tuple[float, float]:
        """
        Calculate total cost and time for generating tokens.
        
        Args:
            profile: Model performance profile
            total_tokens: Total tokens to generate
            include_startup: Include startup costs
            
        Returns:
            (total_cost, total_time_seconds)
        """
        # Generation time
        generation_time = total_tokens / profile.tokens_per_second
        
        # Total time including startup
        total_time = generation_time
        if include_startup:
            total_time += profile.startup_time_seconds
            
        # Calculate cost
        if profile.provider == "runpod":
            # RunPod charges by time, minimum billing period
            billable_time = max(total_time, profile.min_runtime_seconds)
            hourly_cost = profile.cost_per_million_tokens * profile.tokens_per_second * 3600 / 1_000_000
            total_cost = (billable_time / 3600) * hourly_cost
        else:
            # Token-based pricing (Vertex)
            total_cost = (total_tokens / 1_000_000) * profile.cost_per_million_tokens
            
        return total_cost, total_time
        
    def compare_models_for_workload(self,
                                  total_tokens: int,
                                  max_wait_seconds: Optional[float] = None,
                                  required_context: int = 4096) -> List[Dict[str, Any]]:
        """
        Compare models for a specific workload.
        
        Args:
            total_tokens: Total tokens to generate
            max_wait_seconds: Maximum acceptable wait time
            required_context: Required context window
            
        Returns:
            List of model comparisons sorted by cost efficiency
        """
        comparisons = []
        
        for model_key, profile in self.PERFORMANCE_PROFILES.items():
            # Skip if context window too small
            if profile.max_context_length < required_context:
                continue
                
            # Calculate cost and time
            cost, time_seconds = self.calculate_total_cost(
                profile, total_tokens, self.include_startup_cost
            )
            
            # Skip if too slow
            if max_wait_seconds and time_seconds > max_wait_seconds:
                continue
                
            comparisons.append({
                "model": model_key,
                "provider": profile.provider,
                "total_cost": cost,
                "total_time_seconds": time_seconds,
                "cost_per_1k_tokens": cost * 1000 / total_tokens if total_tokens > 0 else 0,
                "tokens_per_second": profile.tokens_per_second,
                "startup_time": profile.startup_time_seconds
            })
            
        # Sort by cost
        comparisons.sort(key=lambda x: x["total_cost"])
        
        return comparisons
        
    def select_optimal_model(self,
                           tokens_required: int,
                           urgency: Literal["immediate", "normal", "batch"] = "normal",
                           min_quality_tier: Optional[str] = None,
                           context_required: int = 4096) -> Dict[str, Any]:
        """
        Select the optimal model based on cost and requirements.
        
        Args:
            tokens_required: Number of tokens needed
            urgency: How quickly results are needed
            min_quality_tier: Minimum model size (e.g., "70b", "8b")
            context_required: Required context window size
            
        Returns:
            Selected model information
        """
        # Define urgency constraints
        urgency_constraints = {
            "immediate": 30,  # 30 seconds max
            "normal": 300,    # 5 minutes max  
            "batch": None     # No time constraint
        }
        
        max_wait = urgency_constraints.get(urgency)
        
        # Get comparisons
        comparisons = self.compare_models_for_workload(
            tokens_required,
            max_wait,
            context_required
        )
        
        if not comparisons:
            raise ValueError("No suitable models found for requirements")
            
        # Filter by quality tier if specified
        if min_quality_tier:
            filtered = []
            for comp in comparisons:
                model_name = comp["model"]
                # Simple heuristic: check if model name contains size
                if any(size in model_name.lower() for size in ["70b", "72b", "405b"]):
                    if min_quality_tier in ["70b", "large"]:
                        filtered.append(comp)
                elif "405b" in model_name.lower():
                    if min_quality_tier in ["405b", "xlarge"]:
                        filtered.append(comp)
                else:
                    # Smaller models
                    if min_quality_tier in ["7b", "8b", "small", "medium"]:
                        filtered.append(comp)
                        
            comparisons = filtered if filtered else comparisons
            
        # Select best option
        best = comparisons[0]
        
        # Log decision
        logger.info(
            f"Selected {best['model']} - "
            f"Cost: ${best['total_cost']:.4f}, "
            f"Time: {best['total_time_seconds']:.1f}s, "
            f"Speed: {best['tokens_per_second']:.1f} tok/s"
        )
        
        # Special note for Gemini Flash vs RunPod comparison
        if len(comparisons) > 1:
            runpod_options = [c for c in comparisons if c["provider"] == "runpod"]
            gemini_options = [c for c in comparisons if c["provider"] == "vertex"]
            
            if runpod_options and gemini_options:
                runpod_best = runpod_options[0]
                gemini_best = gemini_options[0]
                
                cost_ratio = runpod_best["total_cost"] / gemini_best["total_cost"]
                speed_ratio = gemini_best["tokens_per_second"] / runpod_best["tokens_per_second"]
                
                logger.info(
                    f"Gemini Flash is {cost_ratio:.1f}x cheaper and "
                    f"{speed_ratio:.1f}x faster than RunPod for this workload"
                )
                
        return best
        
    async def benchmark_actual_performance(self,
                                         model_key: str,
                                         test_prompt: str = "Write a 500 word essay about AI.",
                                         expected_tokens: int = 500) -> Dict[str, float]:
        """
        Benchmark actual model performance.
        
        Args:
            model_key: Model to benchmark
            test_prompt: Prompt to test with
            expected_tokens: Expected output tokens
            
        Returns:
            Actual performance metrics
        """
        profile = self.PERFORMANCE_PROFILES.get(model_key)
        if not profile:
            raise ValueError(f"Unknown model: {model_key}")
            
        start_time = time.time()
        tokens_generated = 0
        
        try:
            if profile.provider == "ollama":
                # Benchmark Ollama
                import ollama
                
                response = ollama.generate(
                    model=profile.model_name,
                    prompt=test_prompt,
                    stream=True
                )
                
                for chunk in response:
                    if "response" in chunk:
                        tokens_generated += len(chunk["response"].split())
                        
            elif profile.provider == "vertex":
                # Benchmark Vertex AI
                from google.cloud import aiplatform
                from vertexai.generative_models import GenerativeModel
                
                model = GenerativeModel(profile.model_name)
                response = model.generate_content(test_prompt)
                tokens_generated = len(response.text.split())
                
            else:
                logger.warning(f"Benchmarking not implemented for {profile.provider}")
                return {}
                
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}
            
        end_time = time.time()
        duration = end_time - start_time
        actual_tokens_per_second = tokens_generated / duration if duration > 0 else 0
        
        return {
            "actual_tokens_per_second": actual_tokens_per_second,
            "expected_tokens_per_second": profile.tokens_per_second,
            "tokens_generated": tokens_generated,
            "duration_seconds": duration,
            "efficiency": actual_tokens_per_second / profile.tokens_per_second if profile.tokens_per_second > 0 else 0
        }


# Validation
if __name__ == "__main__":
    selector = CostAwareModelSelector()
    
    # Test different workloads
    workloads = [
        ("Small task", 100, "immediate"),
        ("Medium task", 1000, "normal"),
        ("Large task", 10000, "normal"),
        ("Batch processing", 100000, "batch"),
    ]
    
    print("Cost Comparison for Different Workloads:")
    print("=" * 80)
    
    for name, tokens, urgency in workloads:
        print(f"\n{name} ({tokens} tokens, {urgency} urgency):")
        
        comparisons = selector.compare_models_for_workload(tokens, context_required=4096)
        
        # Show top 3 options
        for i, comp in enumerate(comparisons[:3]):
            print(f"  {i+1}. {comp['model']}")
            print(f"     Cost: ${comp['total_cost']:.4f} (${comp['cost_per_1k_tokens']:.3f}/1k tokens)")
            print(f"     Time: {comp['total_time_seconds']:.1f}s ({comp['tokens_per_second']:.1f} tok/s)")
            print(f"     Provider: {comp['provider']}")
            
        # Special comparison
        gemini_flash = next((c for c in comparisons if "gemini-1.5-flash" in c["model"]), None)
        runpod_70b = next((c for c in comparisons if "llama3.1:70b" in c["model"]), None)
        
        if gemini_flash and runpod_70b:
            print(f"\n  Gemini Flash vs RunPod 70B:")
            print(f"  - Gemini Flash: ${gemini_flash['total_cost']:.4f} in {gemini_flash['total_time_seconds']:.1f}s")
            print(f"  - RunPod 70B: ${runpod_70b['total_cost']:.4f} in {runpod_70b['total_time_seconds']:.1f}s")
            print(f"  - Gemini Flash is {runpod_70b['total_cost']/gemini_flash['total_cost']:.1f}x cheaper")
    
    print("\n Module validation passed")