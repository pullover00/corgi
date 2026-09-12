#!/bin/bash
# Movable-object-gate arm of the SceneDiff-paired single-query protocol (P1):
# SAME frozen manifest, SAME queries, SAME root as the refine-OFF baseline
# (run_scenediff_covis_p1_test.sh) so reconstruction / localization / raw render
# are cache hits. Two semantic differences from that baseline, both declared here:
#   1. per-query movable-object whitelist built by scripts/scenediff_movable_masks.py
#      (8 prompts x {image_t1, raw render_t0} at confidence 0.5, unioned), and
#   2. configs/scenediff_v10_no_dino_no_refine_gate.yaml, which adds ONLY
#      enable_movable_object_gate: true / minimum_movable_object_fraction: 0.5.
# Method ported from the PASLCD session 2026-09-12 (commit dd0b97d); comparison
# target is the refine-OFF, replacement-OFF, 237-pair baseline (pooled IoU 0.1032).
# Chunked (10 x 25) with .done markers; re-running resumes. Step 1 is resumable too.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT
EXP=scenediff_single_query_covis_v1_gate
ROOT=results/scenediff_single_query_covis_v1
CONFIG=configs/scenediff_v10_no_dino_no_refine_gate.yaml
QUERIES=$ROOT/manifest_test_top1.runner_queries.json
CHUNKS=data/scenediff_benchmark/test_split_250_chunks
MASKS=$ROOT/movable_masks
LOGS=$ROOT/SceneDiff/_experiments/$EXP/logs; mkdir -p "$LOGS" "$MASKS"

MIN_AVAIL_MB=5000
MIN_GPU_FREE_MB=4000
check_headroom() {
  local avail_mb gpu_free_mb
  avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail_mb" -lt "$MIN_AVAIL_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${avail_mb} MB RAM available (< ${MIN_AVAIL_MB} MB). Free memory and re-run; completed chunks are preserved."; exit 1
  fi
  gpu_free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$gpu_free_mb" ] && [ "$gpu_free_mb" -lt "$MIN_GPU_FREE_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${gpu_free_mb} MiB GPU free (< ${MIN_GPU_FREE_MB} MiB). Check nvidia-smi for a stale process and re-run."; exit 1
  fi
}

check_headroom
echo "$(date '+%F %T') COVIS_P1_GATE_START commit=$(git rev-parse --short HEAD) config_sha=$(sha256sum $CONFIG | cut -c1-16) queries_sha=$(sha256sum $QUERIES | cut -c1-16)"

# --- step 1: movable-object whitelist masks for every query (detection env; resumable) ---
if [ ! -f "$MASKS/MASKS_DONE" ]; then
  echo "$(date '+%F %T') === masks START"
  conda run --no-capture-output -n goldilocs python scripts/scenediff_movable_masks.py \
      --root "$ROOT/SceneDiff" --out "$MASKS" --confidence-threshold 0.5 > "$LOGS/masks.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- masks exit $rc"
  if [ $rc -ne 0 ]; then echo "masks step FAILED -- not starting the gated detect"; tail -5 "$LOGS/masks.log"; exit 1; fi
  touch "$MASKS/MASKS_DONE"
  echo "whitelist coverage summary (fraction of frame): $(python3 -c "
import csv; v=[float(r['union_coverage_fraction']) for r in csv.DictReader(open('$MASKS/union_coverage.csv'))]
v.sort(); n=len(v); print(f'n={n} min={v[0]:.3f} p10={v[n//10]:.3f} median={v[n//2]:.3f} max={v[-1]:.3f} below2pct={sum(x<0.02 for x in v)}')")"
else
  echo "$(date '+%F %T') skip masks (done)"
fi

# --- step 2: gated detect, chunked, cache-hitting the baseline's shared stages ---
for c in "$CHUNKS"/chunk_*.txt; do
  n=$(basename "$c" .txt)
  [ -f "$LOGS/$n.done" ] && { echo "$(date '+%F %T') skip $n (done)"; continue; }
  check_headroom
  echo "$(date '+%F %T') === $n START ($(wc -l < "$c") pairs)"
  python scripts/run_scenediff_diagnostic.py --root "$ROOT" --experiment "$EXP" --config "$CONFIG" \
      --pair-ids-file "$c" --queries-file "$QUERIES" --movable-mask-root "$MASKS" \
      --detect-extra-args=--sequential-model-lifecycle > "$LOGS/$n.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- $n exit $rc"
  if [ $rc -eq 0 ]; then
    cp "$ROOT/SceneDiff/_experiments/$EXP/summary.json" "$LOGS/$n.summary.json"; touch "$LOGS/$n.done"
    grep -E "pooled IoU" "$LOGS/$n.log" | tail -1
  else
    echo "$(date '+%F %T') $n FAILED rc=$rc -- continuing; rerun this script to retry"
    grep -E "RECONSTRUCTION FAILED|EVAL FAILED|Traceback|FileNotFoundError" "$LOGS/$n.log" | tail -3
  fi
done
echo "$(date '+%F %T') COVIS_P1_GATE_DONE"
