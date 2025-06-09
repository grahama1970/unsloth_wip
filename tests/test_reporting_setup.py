"""
Test to verify the test reporting setup is working correctly.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import pytest


class TestReportingSetup:
    """Test class to verify pytest reporting is configured properly."""
    
    def test_simple_pass(self):
        """A simple test that should pass."""
        assert 1 + 1 == 2
        
    def test_with_duration(self):
        """A test to show duration in reports."""
        import time
        time.sleep(0.1)
        assert True
        
    @pytest.mark.unit
    def test_marked_unit(self):
        """A unit test with marker."""
        assert "test" in "testing"
        
    @pytest.mark.integration
    def test_marked_integration(self):
        """An integration test with marker."""
        assert len([1, 2, 3]) == 3
        
    @pytest.mark.skip(reason="Testing skip functionality")
    def test_skipped(self):
        """A test that should be skipped."""
        assert False  # This should never run
        
    @pytest.mark.xfail(reason="Testing expected failure")
    def test_expected_failure(self):
        """A test that is expected to fail."""
        assert False  # This is expected to fail