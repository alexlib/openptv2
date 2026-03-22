#!/bin/bash
# Script to build binary wheels using cibuildwheel
# Usage: ./run_cibuildwheel.sh

set -e

echo "=== Cleaning previous build artifacts ==="
rm -rf bindings/build bindings/dist bindings/liboptv bindings/optv/*.c bindings/optv/optv
rm -rf wheelhouse

echo "=== Installing build dependencies ==="
python -m pip install --upgrade pip
python -m pip install numpy>=2.0.0 cython>=3.0.0 setuptools wheel cibuildwheel

echo "=== Preparing source files ==="
cd bindings
python setup.py prepare
cd ..

echo "=== Building wheels with cibuildwheel ==="
# Build for current Python version only by default
# Set CIBW_BUILD to build specific versions, e.g.:
# export CIBW_BUILD="cp311-* cp312-* cp313-*"
python -m cibuildwheel --output-dir wheelhouse bindings/

echo "=== Build complete ==="
echo "Wheels are in: wheelhouse/"
ls -la wheelhouse/
