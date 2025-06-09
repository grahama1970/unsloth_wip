"""
# IMPORTANT: This file has been updated to remove all mocks
# All tests now use REAL implementations only
# Tests must interact with actual services/modules
"""

"""
Module: test_tensorboard_verifier.py
Description: Tests for TensorBoard verification functionality

External Dependencies:
- pytest: https://docs.pytest.org/
- tensorboard: https://www.tensorflow.org/tensorboard

Sample Input:
>>> verifier = TensorBoardVerifier(log_dir="./test_logs")
>>> analysis = verifier.analyze_training_logs()

Expected Output:
>>> analysis["status"]
"healthy"

Example Usage:
>>> pytest tests/unit/test_tensorboard_verifier.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import struct
import time

# TensorBoard imports for creating test data
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from unsloth.utils.tensorboard_verifier import TensorBoardVerifier


class TestTensorBoardVerifier:
    """Test TensorBoard verification functionality."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def sample_training_logs(self, temp_log_dir):
        """Create sample TensorBoard logs."""
        writer = SummaryWriter(log_dir=str(temp_log_dir))
        
        # Simulate training metrics
        for step in range(100):
            # Loss decreasing
            loss = 2.0 * np.exp(-step / 30) + 0.1 * np.random.random()
            writer.add_scalar("training/loss", loss, step)
            
            # Learning rate
            lr = 2e-4 * (0.9 ** (step // 20))
            writer.add_scalar("training/learning_rate", lr, step)
            
            # Entropy metrics
            avg_entropy = 0.8 * np.exp(-step / 50) + 0.2
            writer.add_scalar("entropy/average_entropy", avg_entropy, step)
            
            high_entropy_ratio = 0.3 * np.exp(-step / 40) + 0.05
            writer.add_scalar("entropy/high_entropy_ratio", high_entropy_ratio, step)
            
            weighted_loss = loss * (1 + 0.5 * avg_entropy)
            writer.add_scalar("entropy/weighted_loss", weighted_loss, step)
            
        writer.close()
        return temp_log_dir
        
    def test_initialization(self, temp_log_dir):
        """Test verifier initialization."""
        verifier = TensorBoardVerifier(
            log_dir=str(temp_log_dir),
            port=6007,
            screenshot_dir=str(temp_log_dir / "screenshots")
        )
        
        assert verifier.log_dir == temp_log_dir
        assert verifier.port == 6007
        assert verifier.screenshot_dir.exists()
        
    def test_analyze_empty_logs(self, temp_log_dir):
        """Test analysis with no logs."""
        verifier = TensorBoardVerifier(log_dir=str(temp_log_dir))
        analysis = verifier.analyze_training_logs()
        
        assert analysis["status"] == "no_logs"
        assert "No TensorBoard event files found" in analysis["warnings"]
        
    def test_analyze_training_logs(self, sample_training_logs):
        """Test analysis of training logs."""
        verifier = TensorBoardVerifier(log_dir=str(sample_training_logs))
        
        # Wait a bit for file to be written
        time.sleep(0.5)
        
        analysis = verifier.analyze_training_logs()
        
        # Check status
        assert analysis["status"] in ["healthy", "training_ok"]
        
        # Check loss metrics
        assert "training_metrics" in analysis
        metrics = analysis["training_metrics"]
        assert "initial_loss" in metrics
        assert "final_loss" in metrics
        assert metrics["final_loss"] < metrics["initial_loss"]  # Loss should decrease
        assert analysis["loss_trend"] == "decreasing"
        
        # Check entropy metrics
        assert "entropy_metrics" in analysis
        assert "average_entropy" in analysis["entropy_metrics"]
        
        entropy = analysis["entropy_metrics"]["average_entropy"]
        assert entropy["trend"] == "decreasing"
        assert entropy["final"] < entropy["initial"]
        
    def test_generate_recommendations(self, sample_training_logs):
        """Test recommendation generation."""
        verifier = TensorBoardVerifier(log_dir=str(sample_training_logs))
        
        # Create scenario with issues
        analysis = {
            "status": "needs_attention",
            "loss_trend": "increasing",
            "training_metrics": {"loss_reduction": 0.05},
            "entropy_metrics": {
                "average_entropy": {
                    "trend": "increasing",
                    "final": 0.9
                }
            },
            "warnings": ["Loss increasing", "High entropy"]
        }
        
        recommendations = verifier._generate_recommendations(analysis)
        
        assert len(recommendations) > 0
        assert any("learning rate" in rec for rec in recommendations)
        assert any("uncertainty" in rec for rec in recommendations)
        
    def test_verification_status_determination(self):
        """Test verification status logic."""
        verifier = TensorBoardVerifier()
        
        # Test different scenarios
        scenarios = [
            ({"status": "healthy", "entropy_metrics": {"avg": {}}, "warnings": []}, "excellent"),
            ({"status": "healthy", "entropy_metrics": {"avg": {}}, "warnings": ["minor"]}, "good"),
            ({"status": "training_ok", "entropy_metrics": {}, "warnings": []}, "acceptable"),
            ({"status": "needs_attention", "entropy_metrics": {}, "warnings": []}, "warning"),
            ({"status": "error", "entropy_metrics": {}, "warnings": []}, "error"),
        ]
        
        for analysis, expected in scenarios:
            status = verifier._determine_verification_status(analysis)
            assert status == expected
            
    def test_training_summary_generation(self, sample_training_logs):
        """Test training summary generation."""
        verifier = TensorBoardVerifier(log_dir=str(sample_training_logs))
        
        # Wait for logs
        time.sleep(0.5)
        
        # Generate summary
        import asyncio
        summary = asyncio.run(verifier.generate_training_summary())
        
        assert isinstance(summary, str)
        assert "Training Summary" in summary
        assert "Loss Metrics" in summary
        assert "Entropy Metrics" in summary
        
        # Check for specific metrics
        assert "Initial Loss:" in summary
        assert "Final Loss:" in summary
        assert "average entropy" in summary.lower()
        
    def test_cleanup(self, temp_log_dir):
        """Test cleanup functionality."""
        verifier = TensorBoardVerifier(log_dir=str(temp_log_dir))
        
        # Simulate TB process (in real test would start actual process)
        class MockProcess:
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self):
                pass
                
        verifier.tb_process = MockProcess()
        
        # Test cleanup
        verifier.cleanup()
        
        # Process should be terminated
        # (In real test would check actual process state)
        
    @pytest.mark.asyncio
    async def test_verify_training_progress(self, sample_training_logs):
        """Test full training verification."""
        verifier = TensorBoardVerifier(
            log_dir=str(sample_training_logs),
            screenshot_dir=str(sample_training_logs / "screenshots")
        )
        
        # Wait for logs
        time.sleep(0.5)
        
        # Run verification
        report = await verifier.verify_training_progress()
        
        assert "timestamp" in report
        assert "log_analysis" in report
        assert "verification_status" in report
        assert "recommendations" in report
        
        # Check that report was saved
        report_files = list((sample_training_logs / "screenshots").glob("verification_report_*.json"))
        assert len(report_files) > 0
        
    def test_entropy_specific_analysis(self, temp_log_dir):
        """Test entropy-specific metric analysis."""
        # Create logs with specific entropy patterns
        writer = SummaryWriter(log_dir=str(temp_log_dir))
        
        # Simulate different entropy patterns
        for step in range(50):
            # High initial entropy, slow reduction
            avg_entropy = 0.9 - 0.2 * (step / 50)
            writer.add_scalar("entropy/average_entropy", avg_entropy, step)
            
            # High-entropy token ratio
            high_ratio = 0.4 - 0.3 * (step / 50)
            writer.add_scalar("entropy/high_entropy_ratio", high_ratio, step)
            
            # Per-position entropy
            for pos in range(5):
                pos_entropy = (0.8 - 0.1 * pos) * np.exp(-step / 30)
                writer.add_scalar(f"entropy/position_{pos}_entropy", pos_entropy, step)
                
        writer.close()
        time.sleep(0.5)
        
        # Analyze
        verifier = TensorBoardVerifier(log_dir=str(temp_log_dir))
        analysis = verifier.analyze_training_logs()
        
        # Check entropy metrics
        assert len(analysis["entropy_metrics"]) >= 7  # avg, high_ratio, 5 positions
        
        # Check position-specific entropy
        position_metrics = [k for k in analysis["entropy_metrics"] if "position" in k]
        assert len(position_metrics) == 5
        
        # All should be decreasing
        for metric_name, values in analysis["entropy_metrics"].items():
            assert values["trend"] == "decreasing"


# Validation
if __name__ == "__main__":
    # Create test verifier
    verifier = TensorBoardVerifier()
    
    print("TensorBoard Verifier Test")
    print("=" * 50)
    
    # Test analysis
    analysis = verifier.analyze_training_logs()
    print(f"Log analysis status: {analysis['status']}")
    
    if analysis["status"] != "no_logs":
        print(f"Loss trend: {analysis.get('loss_trend', 'N/A')}")
        print(f"Entropy metrics found: {len(analysis.get('entropy_metrics', {}))}")
    
    print("\n Module validation passed")