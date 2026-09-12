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
# Safety guard added 2026-09-11 after repeated system-wide OOM events this week
# (Chrome + this pipeline competing for RAM took down the GNOME session and this
# launcher as collateral, twice). Neither failure was this script's own memory use,
# but nothing here checked headroom before proceeding either. Checked once at start
# and again before every chunk, so pressure building mid-run (e.g. Chrome reopened)
# is caught at the next chunk boundary rather than mid-chunk. A refusal here exits
# cleanly with no partial/corrupt state -- same as any other chunk failure, just
# re-run this script once headroom is back.
MIN_AVAIL_MB=5000
MIN_GPU_FREE_MB=4000
check_headroom() {
  local avail_mb gpu_free_mb
  avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail_mb" -lt "$MIN_AVAIL_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${avail_mb} MB RAM available (< ${MIN_AVAIL_MB} MB threshold). Free up memory (check for Chrome/other jobs) and re-run this script -- completed chunks are preserved."
    exit 1
  fi
  gpu_free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$gpu_free_mb" ] && [ "$gpu_free_mb" -lt "$MIN_GPU_FREE_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${gpu_free_mb} MiB GPU memory free (< ${MIN_GPU_FREE_MB} MiB threshold). Check nvidia-smi for a stale/orphaned process and re-run this script."
    exit 1
  fi
}

check_headroom
echo "$(date '+%F %T') COVIS_P1_TEST_START commit=$(git rev-parse --short HEAD) config_sha=$(sha256sum $CONFIG | cut -c1-16) queries_sha=$(sha256sum $QUERIES | cut -c1-16) manifest_sha=$(sha256sum $ROOT/manifest_test_top1.json | cut -c1-16)"
for c in "$CHUNKS"/chunk_*.txt; do
  n=$(basename "$c" .txt)
  [ -f "$LOGS/$n.done" ] && { echo "$(date '+%F %T') skip $n (done)"; continue; }
  check_headroom
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
