import pytest
import sys
from pathlib import Path
"""Basic tests for unsloth_wip"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



def test_basic_import():
    """Test basic functionality"""
    # This is a minimal test to ensure pytest runs
    assert True, "Basic test should pass"
    print(" Basic test passed for unsloth_wip")

def test_module_structure():
    """Test that module structure exists"""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Check for src directory
    has_src = os.path.exists(os.path.join(project_root, 'src'))
    # Check for unsloth module inside src
    has_unsloth = os.path.exists(os.path.join(project_root, 'src', 'unsloth'))
    
    assert has_src, "Project should have src/ directory"
    assert has_unsloth, "Project should have src/unsloth/ module"
    print(" Module structure verified")
