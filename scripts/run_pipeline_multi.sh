#!/usr/bin/env bash
# Run openptv2-batch (sequence + tracking) over every subfolder of a
# multi-folder experiment -- e.g. several workpieces sharing one calibration.
#
# Convention this expects (see docs/multi-folder-runs.md for the full
# writeup): the experiment root has a shared calibration directory plus one
# subfolder per run, each with its own parameters_*_sample.yaml (a small
# frame range, for a quick check) and parameters_*_batch.yaml (the full
# range). Each subfolder is its own cwd when openptv2-batch runs (it chdirs
# to the yaml's parent dir per docs/cloud-batch.md), so each gets its own
# <subfolder>/res/ -- runs don't clobber each other, sequential or parallel.
#
# Usage:
#   ./run_pipeline_multi.sh <experiment_root>                  # full batch, all subfolders
#   ./run_pipeline_multi.sh <experiment_root> --sample          # quick sample run instead
#   ./run_pipeline_multi.sh <experiment_root> --parallel        # concurrent
#   ./run_pipeline_multi.sh <experiment_root> --folder wp2      # only wp2 (repeatable)
#   ./run_pipeline_multi.sh <experiment_root> --pattern 'parameters_Run1'  # non-default yaml prefix
#
# This script lives inside the openptv2 checkout and runs `uv run
# openptv2-batch` against itself -- no separate OPENPTV2_DIR/--project needed,
# it just needs to be invoked from (or copied alongside) this repo.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 || "$1" == --* ]]; then
  echo "usage: $0 <experiment_root> [--sample|--batch] [--parallel] [--folder NAME]... [--pattern PREFIX]" >&2
  exit 1
fi
EXP_ROOT="$(cd "$1" && pwd)"
shift

VARIANT="batch"
PARALLEL=0
PATTERN="parameters_Run1"
FOLDERS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) VARIANT="sample"; shift ;;
    --batch) VARIANT="batch"; shift ;;
    --parallel) PARALLEL=1; shift ;;
    --folder) FOLDERS+=("$2"); shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  # Auto-discover: any subfolder containing <pattern>_<variant>.yaml
  while IFS= read -r -d '' yaml; do
    FOLDERS+=("$(basename "$(dirname "$yaml")")")
  done < <(find "$EXP_ROOT" -mindepth 2 -maxdepth 2 -name "${PATTERN}_${VARIANT}.yaml" -print0 | sort -z)
fi

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  echo "no ${PATTERN}_${VARIANT}.yaml found under $EXP_ROOT (one folder level deep)" >&2
  exit 1
fi

run_one() {
  local folder="$1"
  local yaml="$EXP_ROOT/$folder/${PATTERN}_${VARIANT}.yaml"
  local log="$EXP_ROOT/$folder/pipeline_$VARIANT.log"
  if [[ ! -f "$yaml" ]]; then
    echo "[$folder] SKIP -- $yaml not found" | tee -a "$log"
    return 1
  fi
  echo "[$folder] starting ($VARIANT) -> $log"
  ( cd "$REPO_DIR" && uv run openptv2-batch "$yaml" ) >"$log" 2>&1
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$folder] done"
  else
    echo "[$folder] FAILED (exit $status) -- see $log"
  fi
  return $status
}

fail=0
if [[ $PARALLEL -eq 1 ]]; then
  pids=()
  for folder in "${FOLDERS[@]}"; do
    run_one "$folder" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
else
  for folder in "${FOLDERS[@]}"; do
    run_one "$folder" || fail=1
  done
fi

if [[ $fail -eq 0 ]]; then
  echo "all requested folders completed ($VARIANT)"
else
  echo "one or more folders failed ($VARIANT) -- check pipeline_$VARIANT.log in each folder" >&2
fi
exit $fail
