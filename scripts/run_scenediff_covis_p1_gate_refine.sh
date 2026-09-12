#!/bin/bash
# GATE + REFINE arm of the SceneDiff-paired single-query protocol (P1) -- the one
# untested cell of the 2x2 (gate on/off x refine on/off). SAME frozen manifest,
# SAME queries, SAME root as every other arm, so reconstruction / localization /
# raw render / DI2FIX refine are all cache hits (all 250 refined renders already
# exist from the refine-ON baseline). Two deliberate differences from the
# refine-OFF gate arm, both consequences of "run the method with refinement":
#   1. config scenediff_v10_no_dino_refine_gate.yaml (refine.enabled true), so
#      SAM3/DINOv2/SAM2 see the refined renders, and
#   2. the whitelist is built on the refined render_t0 (--render-t0-source refine),
#      i.e. on the same frames detection sees, as the method specifies.
# Written with --dump-inventory: stages 1-3 (86.8s + 16.0s + 8.5s of the 113s
# per-query detect) are persisted, so every later three_image_comparison variant
# on these refined images replays at ~1.3s/query (~6 min for 250) instead of 9 h.
# Budget ~45 GB for the bundles. Chunked (10 x 25) with .done markers; re-running resumes.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT
EXP=scenediff_single_query_covis_v1_gate_refine
ROOT=results/scenediff_single_query_covis_v1
CONFIG=configs/scenediff_v10_no_dino_refine_gate.yaml
QUERIES=$ROOT/manifest_test_top1.runner_queries.json
CHUNKS=data/scenediff_benchmark/test_split_250_chunks
MASKS=$ROOT/movable_masks_refined
LOGS=$ROOT/SceneDiff/_experiments/$EXP/logs; mkdir -p "$LOGS" "$MASKS"

MIN_AVAIL_MB=5000
MIN_GPU_FREE_MB=4000
MIN_DISK_GB=80
check_headroom() {
  local avail_mb gpu_free_mb disk_gb
  avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail_mb" -lt "$MIN_AVAIL_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${avail_mb} MB RAM available (< ${MIN_AVAIL_MB} MB). Free memory and re-run; completed chunks are preserved."; exit 1
  fi
  gpu_free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$gpu_free_mb" ] && [ "$gpu_free_mb" -lt "$MIN_GPU_FREE_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${gpu_free_mb} MiB GPU free (< ${MIN_GPU_FREE_MB} MiB). Check nvidia-smi for a stale process and re-run."; exit 1
  fi
  # inventory bundles are ~180 MB/query; refuse to start a chunk that could fill the disk
  disk_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  if [ -n "$disk_gb" ] && [ "$disk_gb" -lt "$MIN_DISK_GB" ]; then
    echo "$(date '+%F %T') ABORT: only ${disk_gb} GB free on the results volume (< ${MIN_DISK_GB} GB); inventory bundles need ~4.5 GB per chunk."; exit 1
  fi
}

check_headroom
echo "$(date '+%F %T') COVIS_P1_GATE_REFINE_START commit=$(git rev-parse --short HEAD) config_sha=$(sha256sum $CONFIG | cut -c1-16) queries_sha=$(sha256sum $QUERIES | cut -c1-16)"

# --- step 1: whitelist masks on image_t1 + REFINED render_t0 (detection env; resumable) ---
if [ ! -f "$MASKS/MASKS_DONE" ]; then
  echo "$(date '+%F %T') === masks START (render_t0 from shared/refine)"
  conda run --no-capture-output -n goldilocs python scripts/scenediff_movable_masks.py \
      --root "$ROOT/SceneDiff" --out "$MASKS" --confidence-threshold 0.5 --render-t0-source refine > "$LOGS/masks.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- masks exit $rc"
  if [ $rc -ne 0 ]; then echo "masks step FAILED -- not starting the gated detect"; tail -5 "$LOGS/masks.log"; exit 1; fi
  touch "$MASKS/MASKS_DONE"
  echo "whitelist coverage (refined render_t0): $(python3 -c "
import csv; v=[float(r['union_coverage_fraction']) for r in csv.DictReader(open('$MASKS/union_coverage.csv'))]
v.sort(); n=len(v); print(f'n={n} min={v[0]:.3f} p10={v[n//10]:.3f} median={v[n//2]:.3f} max={v[-1]:.3f} zero={sum(x==0 for x in v)} below2pct={sum(x<0.02 for x in v)}')")"
else
  echo "$(date '+%F %T') skip masks (done)"
fi

# --- step 2: gated detect on refined renders, chunked, dumping replay inventories ---
for c in "$CHUNKS"/chunk_*.txt; do
  n=$(basename "$c" .txt)
  [ -f "$LOGS/$n.done" ] && { echo "$(date '+%F %T') skip $n (done)"; continue; }
  check_headroom
  echo "$(date '+%F %T') === $n START ($(wc -l < "$c") pairs)"
  python scripts/run_scenediff_diagnostic.py --root "$ROOT" --experiment "$EXP" --config "$CONFIG" \
      --pair-ids-file "$c" --queries-file "$QUERIES" --movable-mask-root "$MASKS" --dump-inventory \
      --detect-extra-args=--sequential-model-lifecycle > "$LOGS/$n.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- $n exit $rc"
  if [ $rc -eq 0 ]; then
    cp "$ROOT/SceneDiff/_experiments/$EXP/summary.json" "$LOGS/$n.summary.json"; touch "$LOGS/$n.done"
    grep -E "pooled IoU" "$LOGS/$n.log" | tail -1
  else
    echo "$(date '+%F %T') $n FAILED rc=$rc -- continuing; rerun this script to retry"
    grep -E "RECONSTRUCTION FAILED|EVAL FAILED|Traceback|FileNotFoundError|No space left" "$LOGS/$n.log" | tail -3
  fi
done
echo "$(date '+%F %T') COVIS_P1_GATE_REFINE_DONE"
