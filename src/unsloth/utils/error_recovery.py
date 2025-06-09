"""Comprehensive error recovery and resilience utilities.
Module: error_recovery.py
Description: Robust error handling, retry logic, and recovery mechanisms

External Dependencies:
- tenacity: https://tenacity.readthedocs.io/
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> @with_recovery(max_retries=3)
>>> async def risky_operation():
...     return await api_call()

Expected Output:
>>> result = await risky_operation()
>>> # Automatically retries on failure

Example Usage:
>>> from unsloth.utils.error_recovery import with_recovery, RecoveryManager
>>> recovery = RecoveryManager()
>>> result = await recovery.execute_with_fallback(main_func, fallback_func)
"""

import asyncio
import functools
import traceback
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

T = TypeVar("T")


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different failure types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CHECKPOINT = "checkpoint"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"


class ErrorCategory(str, Enum):
    """Categories of errors for targeted recovery."""
    OOM = "out_of_memory"
    NETWORK = "network_error"
    TIMEOUT = "timeout"
    PERMISSION = "permission_denied"
    RESOURCE = "resource_unavailable"
    INVALID_INPUT = "invalid_input"
    UNKNOWN = "unknown"


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=5),
        half_open_requests: int = 3
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time before attempting recovery
            half_open_requests: Number of test requests in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_successes = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_successes = 0
            else:
                raise RuntimeError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == "half_open":
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed")
        elif self.state == "closed":
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RecoveryManager:
    """Comprehensive recovery management system."""
    
    def __init__(self):
        """Initialize recovery manager."""
        self.checkpoints: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_handlers: Dict[ErrorCategory, Callable] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default error handlers."""
        self.error_handlers[ErrorCategory.OOM] = self._handle_oom
        self.error_handlers[ErrorCategory.NETWORK] = self._handle_network_error
        self.error_handlers[ErrorCategory.TIMEOUT] = self._handle_timeout
        self.error_handlers[ErrorCategory.RESOURCE] = self._handle_resource_error
    
    async def execute_with_recovery(
        self,
        func: Callable[..., T],
        *args,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        fallback: Optional[Callable[..., T]] = None,
        checkpoint_id: Optional[str] = None,
        **kwargs
    ) -> T:
        """Execute function with comprehensive recovery.
        
        Args:
            func: Function to execute
            strategy: Recovery strategy to use
            fallback: Fallback function if main fails
            checkpoint_id: ID for checkpoint recovery
            
        Returns:
            Function result or fallback result
        """
        try:
            # Try checkpoint recovery first
            if checkpoint_id and checkpoint_id in self.checkpoints:
                logger.info(f"Recovering from checkpoint: {checkpoint_id}")
                return self.checkpoints[checkpoint_id]
            
            # Execute with circuit breaker if available
            if func.__name__ in self.circuit_breakers:
                breaker = self.circuit_breakers[func.__name__]
                result = await breaker.call(func, *args, **kwargs)
            else:
                result = await func(*args, **kwargs)
            
            # Save checkpoint if requested
            if checkpoint_id:
                self.checkpoints[checkpoint_id] = result
            
            return result
            
        except Exception as e:
            error_category = self._categorize_error(e)
            logger.error(f"Error in {func.__name__}: {error_category} - {str(e)}")
            
            # Try error-specific handler
            if error_category in self.error_handlers:
                try:
                    return await self.error_handlers[error_category](
                        func, e, *args, **kwargs
                    )
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
            
            # Try recovery strategy
            if strategy == RecoveryStrategy.FALLBACK and fallback:
                logger.info("Attempting fallback function")
                return await fallback(*args, **kwargs)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._graceful_degradation(func, e)
            else:
                raise
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for appropriate handling."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if "out of memory" in error_str or "oom" in error_str:
            return ErrorCategory.OOM
        elif any(net in error_str for net in ["connection", "network", "timeout"]):
            return ErrorCategory.NETWORK
        elif "timeout" in error_type:
            return ErrorCategory.TIMEOUT
        elif "permission" in error_str or "access denied" in error_str:
            return ErrorCategory.PERMISSION
        elif "resource" in error_str or "unavailable" in error_str:
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.UNKNOWN
    
    async def _handle_oom(self, func: Callable, error: Exception, *args, **kwargs):
        """Handle out-of-memory errors."""
        logger.warning("OOM detected, attempting recovery...")
        
        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")
        except ImportError:
            pass
        
        # Try with reduced batch size
        if "batch_size" in kwargs:
            new_batch_size = max(1, kwargs["batch_size"] // 2)
            logger.info(f"Retrying with reduced batch size: {new_batch_size}")
            kwargs["batch_size"] = new_batch_size
            return await func(*args, **kwargs)
        
        raise error
    
    async def _handle_network_error(self, func: Callable, error: Exception, *args, **kwargs):
        """Handle network errors with exponential backoff."""
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=4, max=60),
            retry=retry_if_exception_type((ConnectionError, TimeoutError))
        )
        async def retry_with_backoff():
            return await func(*args, **kwargs)
        
        try:
            return await retry_with_backoff()
        except RetryError:
            logger.error("Network error persists after retries")
            raise error
    
    async def _handle_timeout(self, func: Callable, error: Exception, *args, **kwargs):
        """Handle timeout errors."""
        # Increase timeout if possible
        if "timeout" in kwargs:
            new_timeout = kwargs["timeout"] * 2
            logger.info(f"Retrying with increased timeout: {new_timeout}s")
            kwargs["timeout"] = new_timeout
            return await func(*args, **kwargs)
        
        raise error
    
    async def _handle_resource_error(self, func: Callable, error: Exception, *args, **kwargs):
        """Handle resource unavailability."""
        logger.info("Resource unavailable, waiting before retry...")
        await asyncio.sleep(30)  # Wait for resource
        
        return await func(*args, **kwargs)
    
    def _graceful_degradation(self, func: Callable, error: Exception) -> Any:
        """Provide degraded functionality when full feature fails."""
        logger.warning(f"Graceful degradation for {func.__name__}")
        
        # Return sensible defaults based on function name
        if "evaluate" in func.__name__:
            return {"status": "degraded", "error": str(error), "metrics": {}}
        elif "train" in func.__name__:
            return {"status": "failed", "error": str(error), "checkpoint": None}
        else:
            return None
    
    def create_checkpoint(self, checkpoint_id: str, data: Any):
        """Create recovery checkpoint."""
        self.checkpoints[checkpoint_id] = data
        logger.debug(f"Created checkpoint: {checkpoint_id}")
    
    def register_circuit_breaker(
        self,
        func_name: str,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=5)
    ):
        """Register circuit breaker for function."""
        self.circuit_breakers[func_name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )


# Decorator for easy recovery application
def with_recovery(
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    max_retries: int = 3,
    fallback: Optional[Callable] = None,
    checkpoint: bool = False
):
    """Decorator to add recovery to functions.
    
    Args:
        strategy: Recovery strategy
        max_retries: Maximum retry attempts
        fallback: Fallback function
        checkpoint: Enable checkpointing
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = RecoveryManager()
            
            checkpoint_id = None
            if checkpoint:
                checkpoint_id = f"{func.__name__}_{datetime.now().isoformat()}"
            
            if strategy == RecoveryStrategy.RETRY:
                @retry(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential(multiplier=1, min=4, max=10)
                )
                async def retry_func():
                    return await func(*args, **kwargs)
                
                try:
                    result = await retry_func()
                    if checkpoint_id:
                        manager.create_checkpoint(checkpoint_id, result)
                    return result
                except RetryError as e:
                    if fallback:
                        return await fallback(*args, **kwargs)
                    raise e.last_attempt.exception()
            else:
                return await manager.execute_with_recovery(
                    func,
                    *args,
                    strategy=strategy,
                    fallback=fallback,
                    checkpoint_id=checkpoint_id,
                    **kwargs
                )
        
        return wrapper
    return decorator


# Utility functions for common recovery scenarios
async def safe_file_operation(
    operation: Callable[[Path], T],
    file_path: Path,
    create_if_missing: bool = False
) -> Optional[T]:
    """Safely execute file operations with recovery."""
    try:
        return operation(file_path)
    except FileNotFoundError:
        if create_if_missing:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            return operation(file_path)
        logger.error(f"File not found: {file_path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        return None
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return None


def create_recovery_report(errors: List[Exception]) -> Dict[str, Any]:
    """Create comprehensive error recovery report."""
    manager = RecoveryManager()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_errors": len(errors),
        "error_categories": {},
        "recovery_suggestions": []
    }
    
    for error in errors:
        category = manager._categorize_error(error)
        if category not in report["error_categories"]:
            report["error_categories"][category] = []
        
        report["error_categories"][category].append({
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        })
    
    # Add recovery suggestions
    if ErrorCategory.OOM in report["error_categories"]:
        report["recovery_suggestions"].append(
            "Reduce batch size or enable gradient checkpointing"
        )
    
    if ErrorCategory.NETWORK in report["error_categories"]:
        report["recovery_suggestions"].append(
            "Check network connectivity and API endpoints"
        )
    
    return report


if __name__ == "__main__":
    # Validation
    async def validate():
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_func():
            raise RuntimeError("Test failure")
        
        # Should fail twice then open circuit
        try:
            breaker.call(failing_func)
        except:
            pass
        
        try:
            breaker.call(failing_func)
        except:
            pass
        
        assert breaker.state == "open"
        
        # Test recovery decorator
        @with_recovery(max_retries=2)
        async def retrying_func(should_fail: bool = True):
            if should_fail:
                raise ValueError("Retry test")
            return "success"
        
        try:
            await retrying_func(should_fail=True)
        except ValueError:
            pass  # Expected
        
        # Test recovery manager
        manager = RecoveryManager()
        
        async def main_func():
            raise RuntimeError("Main failed")
        
        async def fallback_func():
            return "fallback_success"
        
        result = await manager.execute_with_recovery(
            main_func,
            strategy=RecoveryStrategy.FALLBACK,
            fallback=fallback_func
        )
        
        assert result == "fallback_success"
        
        print(" Error recovery validation passed")
    
    asyncio.run(validate())