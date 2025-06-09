
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
"""Test runpod_ops import."""

try:
    from runpod_ops import (
        RunPodManager,
        InstanceOptimizer,
        CostCalculator,
        TrainingOrchestrator,
        InferenceServer
    )
    print(" Successfully imported runpod_ops components")
    print(f"  - RunPodManager: {RunPodManager}")
    print(f"  - InstanceOptimizer: {InstanceOptimizer}")
    print(f"  - CostCalculator: {CostCalculator}")
    print(f"  - TrainingOrchestrator: {TrainingOrchestrator}")
    print(f"  - InferenceServer: {InferenceServer}")
except ImportError as e:
    print(f" Import error: {e}")
    print("\nTo install runpod_ops:")
    print("  uv pip install git+https://github.com/grahama1970/runpod_ops.git")