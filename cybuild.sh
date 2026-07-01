#!/usr/bin/env bash
# Build and/or annotate individual Cython pure-Python modules in algorithms/
# Usage:
#   ./cybuild.sh          # cythonize all (no compile)
#   ./cybuild.sh -a       # cythonize + annotate all
#   ./cybuild.sh -a file  # cythonize + annotate one file
#   ./cybuild.sh -c file  # cythonize + compile to .so for one file
set -euo pipefail

SRC_DIR="src/openptv2/algorithms"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

cythonize_one() {
    local f="$1"
    local annotate="${2:-0}"
    local py="$SRC_DIR/$f.py"
    if [ ! -f "$py" ]; then
        echo "ERROR: $py not found"
        exit 1
    fi
    local opts="--3str -3 --module-name openptv2.algorithms.$f"
    if [ "$annotate" = "1" ]; then
        opts="$opts -a"
    fi
    echo "--- Cythonizing $f ---"
    # shellcheck disable=SC2086
    uv run cython $opts "$py" -o "$SRC_DIR/$f.c" 2>&1
    echo "--- Done $f ---"
}

compile_one() {
    local f="$1"
    local py="$SRC_DIR/$f.py"
    local c="$SRC_DIR/$f.c"
    if [ ! -f "$c" ]; then
        cythonize_one "$f" 0
    fi
    echo "--- Compiling $f to .so ---"
    uv run python -c "
import sys, numpy
from setuptools import Extension
from setuptools.command.build_ext import build_ext

ext = Extension(
    'openptv2.algorithms.$f',
    sources=['$c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-Wno-cpp', '-Wno-unused-function'],
)
cmd = build_ext(dict(
    build_lib='build_lib',
    build_temp='build_temp',
))
cmd.build_extensions([ext])
" 2>&1
    echo "--- Compiled $f ---"
}

annotate_one() {
    local f="$1"
    cythonize_one "$f" 1
    echo "Annotation: $SRC_DIR/$f.html"
}

MODE="${1:-all}"
FILE="${2:-}"

if [ "$MODE" = "-a" ] && [ -n "$FILE" ]; then
    annotate_one "$FILE"
elif [ "$MODE" = "-c" ] && [ -n "$FILE" ]; then
    compile_one "$FILE"
elif [ "$MODE" = "-a" ]; then
    for f in "$SRC_DIR"/*.py; do
        base=$(basename "$f" .py)
        # Skip __init__
        [ "$base" = "__init__" ] && continue
        annotate_one "$base"
    done
elif [ "$MODE" = "all" ] || [ "$MODE" = "-c" ]; then
    for f in "$SRC_DIR"/*.py; do
        base=$(basename "$f" .py)
        [ "$base" = "__init__" ] && continue
        cythonize_one "$base" 0
    done
fi
