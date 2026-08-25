#!/usr/bin/env bash
# Run the full ChangeSim evaluator once with isolated pair workers.
#
# The evaluator itself now starts every pair in a fresh Python process, retries
# a failed pair, freezes predictions before scoring, and resumes only from
# fingerprint-validated pair manifests. Repeatedly relaunching a shared-process
# evaluator to work around SAM3's former CUDA leak is neither needed nor safe.
#
# progress.jsonl uses status "prediction_frozen" (not the legacy "success");
# this wrapper intentionally does not parse it. The Python evaluator is the
# authority for resume validation.
#
# Usage:
#   scripts/run_eval_resilient.sh MANIFEST OUTPUT_DIR [extra evaluator args]
#
# Example:
#   scripts/run_eval_resilient.sh \
#     data/changesim/manifest-table3.jsonl \
#     out/changesim-table3-guarded-v1 \
#     --prediction-variant guarded

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MANIFEST OUTPUT_DIR [extra evaluator args]" >&2
  exit 64
fi

OCMASK_MANIFEST=$1
OCMASK_OUTPUT=$2
shift 2

OCMASK_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OCMASK_REPOSITORY=$(cd -- "$OCMASK_SCRIPT_DIR/.." && pwd)

cd -- "$OCMASK_REPOSITORY"

exec env \
  PYTHONPATH="$OCMASK_REPOSITORY/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m ocmask.cli evaluate changesim \
  --manifest "$OCMASK_MANIFEST" \
  --output "$OCMASK_OUTPUT" \
  --full-pipeline \
  --pair-retries 1 \
  --continue-on-error \
  "$@"
