#!/usr/bin/env bash
# Measure real per-line coverage of the compiled Cython algorithms modules.
#
# Line tracing is opt-in and slow, so this does a dedicated traced build, runs
# the suite under the Cython.Coverage plugin, then restores the fast production
# build. Pass extra pytest args after the script name, e.g.:
#
#   scripts/coverage.sh -m "not slow"
#   scripts/coverage.sh tests/unit/test_trafo.py
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Delete only Cython-GENERATED artifacts. cas_shim.c is hand-written (needed by
# track_kernels_tracking) and must survive.
clean() {
    rm -rf build/   # force real recompile (build/ cache ignores macro changes)
    find src/openptv2/algorithms -maxdepth 1 -name '*.c' ! -name 'cas_shim.c' -delete
    find src/openptv2/algorithms -maxdepth 1 -name '*.so' -delete
}

# Re-cythonize is mtime-gated on the .py files, so a traced build would reuse the
# cached non-traced .c. Wipe generated artifacts to force a full linetrace build.
echo "==> Clean + traced build (OPENPTV_CYTHON_TRACE=1) — slow"
clean
OPENPTV_CYTHON_TRACE=1 uv run python setup.py build_ext --inplace

echo "==> pytest under Cython.Coverage"
# Python 3.12+/coverage 7 default to sys.monitoring (PEP 669), which does NOT
# invoke Cython's legacy line-trace hooks -> compiled modules report 0%. Force
# the C tracer so Cython linetrace events are captured.
export COVERAGE_CORE=ctrace
set +e
uv run pytest --cov --cov-config=.coveragerc.cython --cov-report=term-missing "$@"
rc=$?
set -e

echo "==> Clean + restore fast production build (no line tracing)"
clean
uv run python setup.py build_ext --inplace

exit "$rc"
