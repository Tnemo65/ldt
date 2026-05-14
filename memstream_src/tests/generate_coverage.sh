#!/bin/bash

# =============================================================================
# Test Coverage Report Generator for CA-DQStream + MemStream v5
# =============================================================================
#
# This script runs pytest with coverage reporting for the complete pipeline.
#
# Coverage targets:
#   - operators/: Layer functions
#   - core/: MemStream core + IEC
#   - Minimum: 70% line coverage
#
# Usage:
#   ./tests/generate_coverage.sh
#   ./tests/generate_coverage.sh --html      # Generate HTML report
#   ./tests/generate_coverage.sh --xml       # Generate XML report (CI)
#   ./tests/generate_coverage.sh --no-fail   # Don't fail on low coverage
#
# Reference: original_flow.md Section 8.8 (Test Matrix)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
GENERATE_HTML=false
GENERATE_XML=false
NO_FAIL=false
COVERAGE_MIN=70

while [[ $# -gt 0 ]]; do
    case $1 in
        --html)
            GENERATE_HTML=true
            shift
            ;;
        --xml)
            GENERATE_XML=true
            shift
            ;;
        --no-fail)
            NO_FAIL=true
            shift
            ;;
        --min)
            COVERAGE_MIN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--html] [--xml] [--no-fail] [--min COVERAGE_MIN]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/"

# Coverage options
COVERAGE_TARGETS=(
    "--cov=operators"
    "--cov=core"
    "--cov-report=term"
)

if [ "$GENERATE_HTML" = true ]; then
    COVERAGE_TARGETS+=("--cov-report=html")
    echo "HTML coverage report will be generated in: htmlcov/index.html"
fi

if [ "$GENERATE_XML" = true ]; then
    COVERAGE_TARGETS+=("--cov-report=xml")
    echo "XML coverage report will be generated in: coverage.xml"
fi

COVERAGE_TARGETS+=("--cov-fail-under=$COVERAGE_MIN")

# Verbose output
COVERAGE_TARGETS+=("-v")
COVERAGE_TARGETS+=("--tb=short")

echo "================================================================================"
echo "CA-DQStream + MemStream v5 - Test Coverage Report"
echo "================================================================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Coverage targets: operators, core"
echo "Minimum coverage: $COVERAGE_MIN%"
echo ""

# Check for pytest-cov
if ! python -c "import pytest_cov" 2>/dev/null; then
    echo "WARNING: pytest-cov not installed. Installing..."
    pip install pytest-cov
fi

# Run tests with coverage
echo "Running tests with coverage..."
echo ""

if [ "$NO_FAIL" = true ]; then
    # Run without failing on low coverage
    PYTEST_CMD="$PYTEST_CMD ${COVERAGE_TARGETS[*]/--cov-fail-under=*/}"
    echo "Command: $PYTEST_CMD"
    echo ""
    $PYTEST_CMD || true
else
    echo "Command: $PYTEST_CMD ${COVERAGE_TARGETS[*]}"
    echo ""
    $PYTEST_CMD "${COVERAGE_TARGETS[@]}"
fi

echo ""
echo "================================================================================"
echo "Coverage Report Generation Complete"
echo "================================================================================"
echo ""

# Print summary
if [ "$GENERATE_HTML" = true ]; then
    echo "HTML report: file://$PROJECT_ROOT/htmlcov/index.html"
fi

if [ "$GENERATE_XML" = true ]; then
    echo "XML report: $PROJECT_ROOT/coverage.xml"
fi

echo ""
echo "To view HTML report:"
echo "  open htmlcov/index.html"
echo ""
echo "To run specific test files:"
echo "  pytest tests/test_complete_pipeline.py -v"
echo "  pytest tests/test_integration.py -v"
echo "  pytest tests/test_memstream_core.py -v"
echo ""
