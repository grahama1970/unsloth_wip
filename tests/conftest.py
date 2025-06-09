"""
Pytest configuration and shared fixtures for the unsloth test suite.

This file is automatically loaded by pytest and provides:
- Common test fixtures
- Pytest markers for test categorization
- Test configuration settings
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Mark test as a unit test (fast, isolated)")
    config.addinivalue_line("markers", "integration: Mark test as an integration test (may use external services)")
    config.addinivalue_line("markers", "validation: Mark test as a validation test (checks output quality)")
    config.addinivalue_line("markers", "e2e: Mark test as an end-to-end test (full workflow)")
    config.addinivalue_line("markers", "smoke: Mark test as a smoke test (basic sanity check)")
    config.addinivalue_line("markers", "performance: Mark test as a performance test (benchmarking)")
    config.addinivalue_line("markers", "slow: Mark test as slow (takes > 5 seconds)")
    config.addinivalue_line("markers", "gpu: Mark test as requiring GPU")


# Common fixtures
@pytest.fixture
def test_data_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def project_root():
    """Return path to project root directory."""
    return Path(__file__).parent.parent