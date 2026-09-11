#!/bin/bash
# SceneDiff-paired single-query protocol (P1): the official 250-pair test split,
# one query per sequence pair chosen by SceneDiff's own co-visibility pairing,
# v10_no_dino with refine OFF. Inputs are FROZEN before launch:
#   manifest : results/scenediff_single_query_covis_v1/manifest_test_top1.json
#   queries  : .../manifest_test_top1.runner_queries.json  (t1_idx = annotation idx,
#              t1_idx_original = the real frame; both recorded in the manifest)
#   config   : configs/scenediff_v10_no_dino_no_refine.yaml
# Chunked (10 x 25) with .done markers, as the runner commits detect manifests only
# after a whole batch; re-running this script resumes. No validation run precedes
# this by decision (2026-09-11); a 5-pair smoke run verified the export/evaluator path.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT
EXP=scenediff_single_query_covis_v1
ROOT=results/scenediff_single_query_covis_v1
CONFIG=configs/scenediff_v10_no_dino_no_refine.yaml
QUERIES=$ROOT/manifest_test_top1.runner_queries.json
CHUNKS=data/scenediff_benchmark/test_split_250_chunks
LOGS=$ROOT/SceneDiff/_experiments/$EXP/logs; mkdir -p "$LOGS"
echo "$(date '+%F %T') COVIS_P1_TEST_START commit=$(git rev-parse --short HEAD) config_sha=$(sha256sum $CONFIG | cut -c1-16) queries_sha=$(sha256sum $QUERIES | cut -c1-16) manifest_sha=$(sha256sum $ROOT/manifest_test_top1.json | cut -c1-16)"
for c in "$CHUNKS"/chunk_*.txt; do
  n=$(basename "$c" .txt)
  [ -f "$LOGS/$n.done" ] && { echo "$(date '+%F %T') skip $n (done)"; continue; }
  echo "$(date '+%F %T') === $n START ($(wc -l < "$c") pairs)"
  python scripts/run_scenediff_diagnostic.py --root "$ROOT" --experiment "$EXP" --config "$CONFIG" \
      --pair-ids-file "$c" --queries-file "$QUERIES" \
      --detect-extra-args=--sequential-model-lifecycle > "$LOGS/$n.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- $n exit $rc"
  if [ $rc -eq 0 ]; then
    cp "$ROOT/SceneDiff/_experiments/$EXP/summary.json" "$LOGS/$n.summary.json"; touch "$LOGS/$n.done"
    grep -E "pooled IoU" "$LOGS/$n.log" | tail -1
  else
    echo "$(date '+%F %T') $n FAILED rc=$rc -- continuing; rerun this script to retry"
    grep -E "RECONSTRUCTION FAILED|EVAL FAILED|Traceback" "$LOGS/$n.log" | tail -3
  fi
done
echo "$(date '+%F %T') COVIS_P1_TEST_DONE"
