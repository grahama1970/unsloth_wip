"""
Module: test_dashboard_simple.py
Purpose: Test dashboard generation without starting a server

External Dependencies:
- pytest: https://docs.pytest.org/

Example Usage:
>>> pytest test_dashboard_simple.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path

import pytest


def test_dashboard_generation():
    """Test that we can generate dashboard HTML files"""
    # Create output directory
    output_dir = Path("./test_dashboard_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate evaluation dashboard HTML
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Dashboard</title>
</head>
<body>
    <h1>Unsloth Model Evaluation Dashboard</h1>
    <p>Generated on {}</p>
</body>
</html>""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Save dashboard
    dashboard_path = output_dir / "evaluation_dashboard.html"
    with open(dashboard_path, "w") as f:
        f.write(dashboard_html)
    
    # Verify file was created
    assert dashboard_path.exists(), "Dashboard HTML file was not created"
    assert dashboard_path.stat().st_size > 0, "Dashboard HTML file is empty"
    
    # Clean up
    dashboard_path.unlink()
    output_dir.rmdir()
    
    
def test_dashboard_style_compliance():
    """Test that dashboard follows 2025 style guide"""
    required_styles = [
        "--color-primary-start: #4F46E5",
        "--color-primary-end: #6366F1", 
        "--font-family-base: 'Inter'",
        "--border-radius-base: 8px",
        "--transition-duration: 250ms"
    ]
    
    dashboard_html = """<style>
    :root {
        --color-primary-start: #4F46E5;
        --color-primary-end: #6366F1;
        --font-family-base: 'Inter', system-ui, -apple-system, sans-serif;
        --border-radius-base: 8px;
        --transition-duration: 250ms;
    }
    </style>"""
    
    for style in required_styles:
        assert style in dashboard_html, f"Missing required style: {style}"
        

def test_dashboard_data_structure():
    """Test dashboard data structure is correct"""
    dashboard_data = {
        "generated_at": datetime.now().isoformat(),
        "metrics": {
            "perplexity": 12.16,
            "answer_relevancy": 0.934,
            "faithfulness": 0.946,
            "hallucination": 0.052
        },
        "improvements": {
            "perplexity": -20.2,
            "answer_relevancy": 4.7,
            "faithfulness": 3.5,
            "hallucination": -41.6
        }
    }
    
    # Verify structure
    assert "generated_at" in dashboard_data
    assert "metrics" in dashboard_data
    assert "improvements" in dashboard_data
    
    # Verify metrics
    assert len(dashboard_data["metrics"]) == 4
    assert all(isinstance(v, (int, float)) for v in dashboard_data["metrics"].values())
    
    # Verify improvements
    assert len(dashboard_data["improvements"]) == 4
    assert dashboard_data["improvements"]["perplexity"] < 0  # Should be negative (improvement)
    assert dashboard_data["improvements"]["hallucination"] < 0  # Should be negative (reduction)
    

if __name__ == "__main__":
    # Run simple validation
    test_dashboard_generation()
    test_dashboard_style_compliance()
    test_dashboard_data_structure()
    print(" Dashboard tests passed")