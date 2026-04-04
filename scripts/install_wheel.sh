#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_ROOT/dist"
TEST_VENV="${TEST_VENV:-/tmp/openptv2_test_venv}"

if [ -d "$TEST_VENV" ]; then
    echo "Removing existing test venv: $TEST_VENV"
    rm -rf "$TEST_VENV"
fi

echo "=== Creating clean environment with Python 3.11 ==="
uv venv "$TEST_VENV" --python 3.11

echo ""
echo "=== Installing wheel with GUI extras ==="
WHEEL_FILE="$DIST_DIR"/openptv2-1.0.0-cp311-cp311-linux_x86_64.whl
if [ ! -f "$WHEEL_FILE" ]; then
    WHEEL_FILE=$(ls "$DIST_DIR"/openptv2-*.whl | head -1)
    echo "Using wheel: $WHEEL_FILE"
fi
uv pip install "$WHEEL_FILE[gui]" -v

echo ""
echo "=== Installing test dependencies ==="
uv pip install pytest pytest-cov numba

echo ""
echo "=== Verifying import ==="
uv run --python "$TEST_VENV/bin/python" -c "import openptv2; import optv; print('OK: Both openptv2 and optv imported successfully')"

echo ""
echo "=== Test environment ready at: $TEST_VENV ==="
echo "To activate: source $TEST_VENV/bin/activate"
echo "To run tests: uv run --python $TEST_VENV/bin/python -m pytest bindings/tests/ algorithms/tests/ -v"
