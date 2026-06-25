#!/bin/bash
set -e

TEST_VENV="${TEST_VENV:-/tmp/openptv2_test_venv}"

if [ ! -d "$TEST_VENV" ]; then
    echo "Error: Test venv not found at $TEST_VENV"
    echo "Run install_wheel.sh first"
    exit 1
fi

echo "=== Running full test suite ==="
echo "Using Python: $TEST_VENV/bin/python"

"$TEST_VENV/bin/python" -m pytest bindings/tests/ algorithms/tests/ -v -m "not slow" --tb=short
