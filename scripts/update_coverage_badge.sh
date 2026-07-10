#!/usr/bin/env bash
# Run pytest coverage (non-slow tests) and patch the badge in README.md.
# Usage: scripts/update_coverage_badge.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PCT=$(uv run pytest -m "not slow" --cov=src/openptv2 --cov-report=term -q 2>&1 \
    | grep '^TOTAL' | awk '{print $NF}' | tr -d '%')

if [[ -z "$PCT" ]]; then
    echo "ERROR: could not extract coverage percentage" >&2
    exit 1
fi

if   [[ "$PCT" -ge 90 ]]; then COLOR=brightgreen
elif [[ "$PCT" -ge 80 ]]; then COLOR=green
elif [[ "$PCT" -ge 70 ]]; then COLOR=yellowgreen
elif [[ "$PCT" -ge 60 ]]; then COLOR=yellow
else                            COLOR=red
fi

BADGE="https://img.shields.io/badge/coverage-${PCT}%25-${COLOR}"

sed -i "s|https://img.shields.io/badge/coverage-[^)]*|${BADGE}|g" README.md

echo "Coverage: ${PCT}% (${COLOR}) — README.md badge updated"
