#!/bin/bash
# Refine-ON arm of the SceneDiff-paired single-query protocol (P1), for a paired
# comparison with the refine-OFF baseline: SAME frozen manifest, SAME queries, SAME
# root (so reconstruction / localization / raw render are reused as cache hits);
# the only difference is the config -- configs/ablate_v10_no_dino.yaml, which
# differs from the OFF config in exactly one key, refine.enabled: true.
# Declared 2026-09-11 before any P1 result was inspected (Tessa's request).
# Uses the resident DI2FIX worker so refine costs seconds per query, not minutes.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT
EXP=scenediff_single_query_covis_v1_refine
ROOT=results/scenediff_single_query_covis_v1
CONFIG=configs/ablate_v10_no_dino.yaml
QUERIES=$ROOT/manifest_test_top1.runner_queries.json
CHUNKS=data/scenediff_benchmark/test_split_250_chunks
WORKER=$ROOT/refine_worker
LOGS=$ROOT/SceneDiff/_experiments/$EXP/logs; mkdir -p "$LOGS" "$WORKER"
# resident DI2FIX worker (difix3d env); the runner falls back to per-run loading if it is not alive
if ! [ -f "$WORKER/heartbeat" ] || [ $(( $(date +%s) - $(stat -c %Y "$WORKER/heartbeat") )) -gt 10 ]; then
  nohup conda run --no-capture-output -n difix3d python scripts/refine_worker.py --dir "$WORKER" > "$LOGS/refine_worker.log" 2>&1 &
  echo "$(date '+%F %T') started refine worker (pid $!)"; sleep 20
fi
# Safety guard added 2026-09-11 -- same rationale as run_scenediff_covis_p1_test.sh.
# Checked after the resident worker starts, so the threshold already accounts for
# its ~5GB GPU reservation.
MIN_AVAIL_MB=5000
MIN_GPU_FREE_MB=4000
check_headroom() {
  local avail_mb gpu_free_mb
  avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail_mb" -lt "$MIN_AVAIL_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${avail_mb} MB RAM available (< ${MIN_AVAIL_MB} MB threshold). Free up memory and re-run this script -- completed chunks are preserved."
    exit 1
  fi
  gpu_free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$gpu_free_mb" ] && [ "$gpu_free_mb" -lt "$MIN_GPU_FREE_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${gpu_free_mb} MiB GPU memory free (< ${MIN_GPU_FREE_MB} MiB threshold). Check nvidia-smi for a stale/orphaned process and re-run this script."
    exit 1
  fi
}

check_headroom
echo "$(date '+%F %T') COVIS_P1_REFINE_START commit=$(git rev-parse --short HEAD) config_sha=$(sha256sum $CONFIG | cut -c1-16) queries_sha=$(sha256sum $QUERIES | cut -c1-16)"
for c in "$CHUNKS"/chunk_*.txt; do
  n=$(basename "$c" .txt)
  [ -f "$LOGS/$n.done" ] && { echo "$(date '+%F %T') skip $n (done)"; continue; }
  check_headroom
  echo "$(date '+%F %T') === $n START ($(wc -l < "$c") pairs)"
  python scripts/run_scenediff_diagnostic.py --root "$ROOT" --experiment "$EXP" --config "$CONFIG" \
      --pair-ids-file "$c" --queries-file "$QUERIES" --refine-worker-dir "$WORKER" \
      --detect-extra-args=--sequential-model-lifecycle > "$LOGS/$n.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- $n exit $rc"
  if [ $rc -eq 0 ]; then cp "$ROOT/SceneDiff/_experiments/$EXP/summary.json" "$LOGS/$n.summary.json"; touch "$LOGS/$n.done"; grep -E "pooled IoU" "$LOGS/$n.log" | tail -1
  else echo "$(date '+%F %T') $n FAILED rc=$rc -- continuing; rerun to retry"; grep -E "RECONSTRUCTION FAILED|EVAL FAILED|Traceback" "$LOGS/$n.log" | tail -3; fi
done
pkill -f "refine_worker.py --dir $WORKER" 2>/dev/null
echo "$(date '+%F %T') COVIS_P1_REFINE_DONE"
