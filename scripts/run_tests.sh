#!/bin/bash
# Test runner script with proper reporting for unsloth_wip project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure we're in the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Unsloth WIP Test Runner ===${NC}"
echo -e "${BLUE}Project Root: $PROJECT_ROOT${NC}"
echo ""

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${RED}Warning: No virtual environment found at .venv${NC}"
fi

# Create reports directory if it doesn't exist
mkdir -p docs/reports

# Function to run tests with a specific marker
run_test_category() {
    local category=$1
    local marker=$2
    
    echo -e "\n${BLUE}Running $category tests...${NC}"
    
    if [ "$marker" = "all" ]; then
        uv run pytest || true
    else
        uv run pytest -m "$marker" || true
    fi
}

# Parse command line arguments
TEST_TYPE="${1:-all}"

case $TEST_TYPE in
    unit)
        run_test_category "Unit" "unit"
        ;;
    integration)
        run_test_category "Integration" "integration"
        ;;
    e2e)
        run_test_category "End-to-End" "e2e"
        ;;
    smoke)
        run_test_category "Smoke" "smoke"
        ;;
    performance)
        run_test_category "Performance" "performance"
        ;;
    all)
        echo -e "${GREEN}Running all tests...${NC}"
        run_test_category "All" "all"
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Usage: $0 [unit|integration|e2e|smoke|performance|all]"
        exit 1
        ;;
esac

# Generate timestamp for report
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Convert JSON report to Markdown if tests were run
if [ -f "docs/reports/test_report.json" ]; then
    echo -e "\n${BLUE}Generating Markdown report...${NC}"
    
    # Use claude-test-reporter if available
    if command -v claude-test-reporter &> /dev/null; then
        uv run claude-test-reporter \
            --input docs/reports/test_report.json \
            --output docs/reports/test_report_${TIMESTAMP}.md \
            --format markdown
    else
        # Fallback: Use Python to convert JSON to Markdown
        uv run python -c "
import json
from datetime import datetime
from pathlib import Path

# Read JSON report
with open('docs/reports/test_report.json', 'r') as f:
    data = json.load(f)

# Extract test results
tests = data.get('tests', [])
summary = data.get('summary', {})

# Generate Markdown report
report = f'''# Test Report - Unsloth WIP
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Tests**: {summary.get('total', 0)}
- **Passed**: {summary.get('passed', 0)}
- **Failed**: {summary.get('failed', 0)}
- **Skipped**: {summary.get('skipped', 0)}
- **Duration**: {summary.get('duration', 0):.2f}s

## Test Results

| Test Name | Result | Duration | Error |
|-----------|--------|----------|-------|
'''

# Add test results
for test in tests:
    name = test.get('nodeid', 'Unknown')
    outcome = test.get('outcome', 'unknown')
    duration = test.get('duration', 0)
    error = ''
    
    if outcome == 'failed':
        error = test.get('call', {}).get('longrepr', 'No error details')
        error = error.split('\\n')[0] if isinstance(error, str) else str(error)[:100]
    
    status = '✅' if outcome == 'passed' else '❌' if outcome == 'failed' else '⏭️'
    report += f'| {name} | {status} {outcome} | {duration:.3f}s | {error} |\\n'

# Write report
Path('docs/reports').mkdir(parents=True, exist_ok=True)
Path(f'docs/reports/test_report_{TIMESTAMP}.md').write_text(report)
print(f'Report generated: docs/reports/test_report_{TIMESTAMP}.md')
"
    fi
    
    # Copy latest reports with timestamp
    cp docs/reports/test_report.json "docs/reports/test_report_${TIMESTAMP}.json"
    cp docs/reports/test_report.html "docs/reports/test_report_${TIMESTAMP}.html"
    
    echo -e "${GREEN}Test reports generated:${NC}"
    echo -e "  - docs/reports/test_report_${TIMESTAMP}.json"
    echo -e "  - docs/reports/test_report_${TIMESTAMP}.html"
    echo -e "  - docs/reports/test_report_${TIMESTAMP}.md"
fi

# Show coverage report location if it exists
if [ -d "docs/reports/coverage" ]; then
    echo -e "\n${GREEN}Coverage report available at:${NC}"
    echo -e "  - docs/reports/coverage/index.html"
fi

# Display test summary
echo -e "\n${BLUE}=== Test Summary ===${NC}"
if [ -f "docs/reports/test_report.json" ]; then
    uv run python -c "
import json
with open('docs/reports/test_report.json', 'r') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    print(f\"Total: {summary.get('total', 0)}\")
    print(f\"Passed: {summary.get('passed', 0)}\")
    print(f\"Failed: {summary.get('failed', 0)}\")
    print(f\"Skipped: {summary.get('skipped', 0)}\")
    print(f\"Duration: {summary.get('duration', 0):.2f}s\")
    
    # Exit with error if tests failed
    if summary.get('failed', 0) > 0:
        exit(1)
"
else
    echo -e "${RED}No test report found!${NC}"
    exit 1
fi