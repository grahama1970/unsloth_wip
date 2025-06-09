"""
Module: model_selector.py
Description: Intelligent model selection for different inference needs

External Dependencies:
- ollama: https://github.com/ollama/ollama-python
- runpod: https://docs.runpod.io/
- google-cloud-aiplatform: https://cloud.google.com/vertex-ai/docs

Sample Input:
>>> selector = ModelSelector()
>>> model_info = selector.select_model(task="hint_generation", context_length=1000)

Expected Output:
>>> model_info
{'provider': 'ollama', 'model': 'llama2:7b', 'endpoint': 'http://localhost:11434', 'estimated_cost': 0.0}

Example Usage:
>>> from unsloth.core.model_selector import ModelSelector
>>> selector = ModelSelector()
>>> response = await selector.generate(prompt="Give a hint", task="teacher_hint")
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List
from enum import Enum
import os
from loguru import logger
import asyncio
import httpx
import json

# Import provider clients
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available - install with: pip install ollama")

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    logger.warning("RunPod not available - install with: pip install runpod")

try:
    from google.cloud import aiplatform
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    logger.warning("Vertex AI not available - install with: pip install google-cloud-aiplatform")


class ModelProvider(Enum):
    """Available model providers."""
    OLLAMA = "ollama"
    RUNPOD = "runpod"
    VERTEX = "vertex"
    LITELLM = "litellm"  # Fallback for API models


@dataclass
class ModelConfig:
    """Configuration for a model."""
    provider: ModelProvider
    model_name: str
    size_category: Literal["small", "medium", "large", "xlarge"]  # <7B, 7-30B, 30-70B, 70B+
    context_window: int
    cost_per_1k_tokens: float
    latency_ms: int  # Expected latency
    capabilities: List[str]  # ["chat", "instruct", "code", "reasoning"]
    

@dataclass
class ModelSelection:
    """Selected model information."""
    provider: str
    model: str
    endpoint: Optional[str]
    estimated_cost: float
    estimated_latency_ms: int
    reason: str


class ModelSelector:
    """Intelligent model selection based on task requirements."""
    
    # Model registry
    MODELS = {
        # Ollama models (local)
        "ollama/llama2:7b": ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama2:7b",
            size_category="small",
            context_window=4096,
            cost_per_1k_tokens=0.0,  # Free locally
            latency_ms=200,
            capabilities=["chat", "instruct", "reasoning"]
        ),
        "ollama/mixtral:8x7b": ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="mixtral:8x7b",
            size_category="medium",
            context_window=32768,
            cost_per_1k_tokens=0.0,
            latency_ms=500,
            capabilities=["chat", "instruct", "code", "reasoning"]
        ),
        "ollama/llama3:70b": ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama3:70b",
            size_category="large",
            context_window=8192,
            cost_per_1k_tokens=0.0,
            latency_ms=2000,
            capabilities=["chat", "instruct", "code", "reasoning"]
        ),
        
        # Vertex models
        "vertex/gemini-flash": ModelConfig(
            provider=ModelProvider.VERTEX,
            model_name="gemini-1.5-flash",
            size_category="medium",
            context_window=1000000,  # 1M context
            cost_per_1k_tokens=0.00035,
            latency_ms=100,
            capabilities=["chat", "instruct", "code", "reasoning"]
        ),
        
        # RunPod models (on-demand GPUs)
        "runpod/llama3.1:405b": ModelConfig(
            provider=ModelProvider.RUNPOD,
            model_name="meta-llama/Llama-3.1-405B-Instruct",
            size_category="xlarge",
            context_window=128000,
            cost_per_1k_tokens=0.005,  # Estimated RunPod cost
            latency_ms=5000,
            capabilities=["chat", "instruct", "code", "reasoning"]
        ),
        "runpod/qwen2.5:72b": ModelConfig(
            provider=ModelProvider.RUNPOD,
            model_name="Qwen/Qwen2.5-72B-Instruct",
            size_category="large",
            context_window=32768,
            cost_per_1k_tokens=0.003,
            latency_ms=3000,
            capabilities=["chat", "instruct", "code", "reasoning"]
        ),
    }
    
    def __init__(self, 
                 prefer_local: bool = True,
                 max_cost_per_1k: float = 0.01,
                 max_latency_ms: int = 5000):
        """
        Initialize model selector.
        
        Args:
            prefer_local: Prefer local models when possible
            max_cost_per_1k: Maximum cost per 1k tokens
            max_latency_ms: Maximum acceptable latency
        """
        self.prefer_local = prefer_local
        self.max_cost_per_1k = max_cost_per_1k
        self.max_latency_ms = max_latency_ms
        
        # Check available providers
        self._check_providers()
        
    def _check_providers(self):
        """Check which providers are available."""
        self.available_providers = []
        
        if OLLAMA_AVAILABLE and self._check_ollama_running():
            self.available_providers.append(ModelProvider.OLLAMA)
            logger.info("Ollama is available")
            
        if VERTEX_AVAILABLE and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            self.available_providers.append(ModelProvider.VERTEX)
            logger.info("Vertex AI is available")
            
        if RUNPOD_AVAILABLE and os.getenv("RUNPOD_API_KEY"):
            self.available_providers.append(ModelProvider.RUNPOD)
            logger.info("RunPod is available")
            
        # LiteLLM is always available as fallback
        self.available_providers.append(ModelProvider.LITELLM)
        
    def _check_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            return response.status_code == 200
        except:
            return False
            
    def select_model(self,
                    task: Literal["teacher_hint", "student_thinking", "evaluation", "complex_reasoning"],
                    context_length: int = 1000,
                    require_capabilities: List[str] = None,
                    min_model_size: Optional[str] = None) -> ModelSelection:
        """
        Select the best model for a task.
        
        Args:
            task: Type of task to perform
            context_length: Required context window size
            require_capabilities: Required model capabilities
            min_model_size: Minimum model size category
            
        Returns:
            ModelSelection with chosen model details
        """
        require_capabilities = require_capabilities or ["chat", "reasoning"]
        
        # Define task requirements
        task_requirements = {
            "teacher_hint": {
                "min_size": "small",
                "prefer_size": "medium",
                "max_latency": 1000,
                "capabilities": ["reasoning", "instruct"]
            },
            "student_thinking": {
                "min_size": "small",
                "prefer_size": "small",  # Fast iteration
                "max_latency": 500,
                "capabilities": ["reasoning"]
            },
            "evaluation": {
                "min_size": "medium",
                "prefer_size": "large",
                "max_latency": 3000,
                "capabilities": ["reasoning", "instruct"]
            },
            "complex_reasoning": {
                "min_size": "large",
                "prefer_size": "xlarge",
                "max_latency": 10000,
                "capabilities": ["reasoning", "code", "instruct"]
            }
        }
        
        req = task_requirements.get(task, task_requirements["teacher_hint"])
        
        # Override with user requirements
        if min_model_size:
            req["min_size"] = min_model_size
            
        # Find suitable models
        candidates = []
        
        for model_key, config in self.MODELS.items():
            # Check if provider is available
            if config.provider not in self.available_providers:
                continue
                
            # Check size requirements
            size_order = ["small", "medium", "large", "xlarge"]
            if size_order.index(config.size_category) < size_order.index(req["min_size"]):
                continue
                
            # Check context window
            if config.context_window < context_length:
                continue
                
            # Check capabilities
            if not all(cap in config.capabilities for cap in require_capabilities):
                continue
                
            # Check constraints
            if config.cost_per_1k_tokens > self.max_cost_per_1k:
                continue
            if config.latency_ms > self.max_latency_ms:
                continue
                
            candidates.append((model_key, config))
            
        if not candidates:
            # Fallback to LiteLLM with GPT-3.5
            return ModelSelection(
                provider="litellm",
                model="gpt-3.5-turbo",
                endpoint=None,
                estimated_cost=0.001,
                estimated_latency_ms=500,
                reason="No suitable models found, using API fallback"
            )
            
        # Sort by preference
        def score_model(item):
            model_key, config = item
            score = 0
            
            # Prefer local if configured
            if self.prefer_local and config.provider == ModelProvider.OLLAMA:
                score += 1000
                
            # Prefer appropriate size
            if config.size_category == req["prefer_size"]:
                score += 500
                
            # Lower cost is better
            score -= config.cost_per_1k_tokens * 100
            
            # Lower latency is better
            score -= config.latency_ms / 10
            
            return score
            
        candidates.sort(key=score_model, reverse=True)
        best_model_key, best_config = candidates[0]
        
        # Build response
        endpoint = None
        if best_config.provider == ModelProvider.OLLAMA:
            endpoint = "http://localhost:11434"
        elif best_config.provider == ModelProvider.RUNPOD:
            endpoint = f"runpod://{best_config.model_name}"
            
        reason = f"Selected {best_config.size_category} model for {task} task"
        if best_config.provider == ModelProvider.OLLAMA:
            reason += " (local, no cost)"
        elif best_config.provider == ModelProvider.VERTEX:
            reason += " (fast, low cost)"
        elif best_config.provider == ModelProvider.RUNPOD:
            reason += " (powerful, on-demand GPU)"
            
        return ModelSelection(
            provider=best_config.provider.value,
            model=best_config.model_name,
            endpoint=endpoint,
            estimated_cost=best_config.cost_per_1k_tokens,
            estimated_latency_ms=best_config.latency_ms,
            reason=reason
        )
        
    async def generate(self,
                      prompt: str,
                      task: str = "teacher_hint",
                      temperature: float = 0.7,
                      max_tokens: int = 500,
                      **kwargs) -> str:
        """
        Generate text using the best model for the task.
        
        Args:
            prompt: Input prompt
            task: Task type for model selection
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for model selection
            
        Returns:
            Generated text
        """
        # Select model
        selection = self.select_model(task, len(prompt), **kwargs)
        logger.info(f"Using {selection.model} via {selection.provider}: {selection.reason}")
        
        # Generate based on provider
        if selection.provider == "ollama":
            return await self._generate_ollama(
                prompt, selection.model, temperature, max_tokens
            )
        elif selection.provider == "vertex":
            return await self._generate_vertex(
                prompt, selection.model, temperature, max_tokens
            )
        elif selection.provider == "runpod":
            return await self._generate_runpod(
                prompt, selection.model, temperature, max_tokens
            )
        else:
            # Fallback to LiteLLM
            return await self._generate_litellm(
                prompt, selection.model, temperature, max_tokens
            )
            
    async def _generate_ollama(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate using Ollama."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available")
            
        response = await ollama.AsyncClient().generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        
        return response["response"]
        
    async def _generate_vertex(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate using Vertex AI."""
        if not VERTEX_AVAILABLE:
            raise RuntimeError("Vertex AI not available")
            
        # Initialize Vertex AI
        aiplatform.init()
        
        # Use generative AI
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel(model)
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        
        return response.text
        
    async def _generate_runpod(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate using RunPod."""
        if not RUNPOD_AVAILABLE:
            raise RuntimeError("RunPod not available")
            
        # This would spin up a RunPod instance
        # For now, return a placeholder
        logger.warning(f"RunPod generation not fully implemented for {model}")
        return await self._generate_litellm(prompt, "gpt-3.5-turbo", temperature, max_tokens)
        
    async def _generate_litellm(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate using LiteLLM."""
        import litellm
        
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


# Validation
if __name__ == "__main__":
    selector = ModelSelector()
    
    # Test model selection for different tasks
    tasks = ["teacher_hint", "student_thinking", "evaluation", "complex_reasoning"]
    
    for task in tasks:
        selection = selector.select_model(task, context_length=2000)
        print(f"\n{task}:")
        print(f"  Model: {selection.model}")
        print(f"  Provider: {selection.provider}")
        print(f"  Cost: ${selection.estimated_cost:.4f}/1k tokens")
        print(f"  Latency: ~{selection.estimated_latency_ms}ms")
        print(f"  Reason: {selection.reason}")
        
    # Test generation
    async def test_generation():
        try:
            response = await selector.generate(
                "What is 2+2?",
                task="teacher_hint"
            )
            print(f"\nGenerated response: {response[:100]}...")
        except Exception as e:
            print(f"\nGeneration failed: {e}")
            
    asyncio.run(test_generation())
    
    print("\n Module validation passed")