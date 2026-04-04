#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_ROOT/dist"

echo "=== Building binary wheel ==="
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

uv run --with build --with setuptools --with cython --with numpy python -m build --wheel

echo ""
echo "=== Wheel built successfully ==="
ls -lh "$DIST_DIR"/*.whl
