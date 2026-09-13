#!/bin/bash
# SINGLE-QUERY HANDICAP experiment: how much does CORGI's one-frame-per-pair protocol
# cost against the same method run on eight frames per pair?
#
# 25 pairs (15 SD-V + 10 SD-K, data/scenediff_benchmark/multiquery_subset_25.txt, seed
# 20260913, drawn BEFORE looking at any per-pair score), each with its top-8 annotated
# query frames by co-visibility with the reconstruction -> 160 queries. Rank 1 is exactly
# the frame the frozen single-query manifest already used, so those 25 are reused from
# scenediff_single_query_covis_v1_gate_refine untouched and only ranks 2-8 (135 queries)
# are computed here, under a SEPARATE experiment name so the frozen arm is never written to.
#
# Same method as the paper's main arm: configs/scenediff_v10_no_dino_refine_gate.yaml
# (refine ON, movable-object gate ON at 0.5, colour replacement OFF), same whitelist
# construction (SAM3 grounded text on image_t1 + the REFINED render_t0), same mask root.
# Nothing is tuned here and no config differs from the main arm -- the only variable is
# WHICH frame is queried.
#
# The runner takes one query per pair, so the ranks are 7 separate invocations.
# Scoring is the official evaluator over all evaluated frames per scene (it merges
# per-frame masks), plus a per-rank breakdown from per_scene.per_frame.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT

EXP=scenediff_multiquery_covis_v1_gate_refine
ROOT=results/scenediff_single_query_covis_v1
CONFIG=configs/scenediff_v10_no_dino_refine_gate.yaml
QDIR=$ROOT/multiquery_rank_queries
PAIRS=data/scenediff_benchmark/multiquery_subset_25.txt
MASKS=$ROOT/movable_masks_refined          # same root as the main arm; rank-1 masks are already there
WORKER=$ROOT/refine_worker
LOGS=$ROOT/SceneDiff/_experiments/$EXP/logs; mkdir -p "$LOGS" "$WORKER"
RANKS=${RANKS:-"2 3 4 5 6 7 8"}

# Resident DI2FIX worker. Without it the runner reloads DifixPipeline in a fresh process for
# every pair (~60 s each, measured 2026-09-13 on the first attempt at this run); with it the
# refine stage is ~1 s per query. The runner falls back to per-run loading if it is not alive,
# so a dead worker costs time, never correctness.
if ! [ -f "$WORKER/heartbeat" ] || [ $(( $(date +%s) - $(stat -c %Y "$WORKER/heartbeat") )) -gt 10 ]; then
  nohup conda run --no-capture-output -n difix3d python scripts/refine_worker.py --dir "$WORKER" > "$LOGS/refine_worker.log" 2>&1 &
  echo "$(date '+%F %T') started refine worker (pid $!)"; sleep 25
fi

MIN_AVAIL_MB=5000
MIN_GPU_FREE_MB=4000
MIN_DISK_GB=80
check_headroom() {
  local avail_mb gpu_free_mb disk_gb
  avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail_mb" -lt "$MIN_AVAIL_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${avail_mb} MB RAM available (< ${MIN_AVAIL_MB} MB). Completed ranks are preserved; re-run to resume."; exit 1
  fi
  gpu_free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$gpu_free_mb" ] && [ "$gpu_free_mb" -lt "$MIN_GPU_FREE_MB" ]; then
    echo "$(date '+%F %T') ABORT: only ${gpu_free_mb} MiB GPU free (< ${MIN_GPU_FREE_MB} MiB). Check nvidia-smi for a stale process."; exit 1
  fi
  disk_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  if [ -n "$disk_gb" ] && [ "$disk_gb" -lt "$MIN_DISK_GB" ]; then
    echo "$(date '+%F %T') ABORT: only ${disk_gb} GB free (< ${MIN_DISK_GB} GB)."; exit 1
  fi
}

check_headroom
echo "$(date '+%F %T') MULTIQUERY_START commit=$(git rev-parse --short HEAD) config_sha=$(sha256sum $CONFIG | cut -c1-16) ranks='$RANKS'"

# --- step 1: localization / render / refine for every new query frame -------------------
# Reference reconstruction is per PAIR and already cached from the 250-query run, so this
# is localize+rasterize+DI2FIX per new frame only. Logged so the cache claim is checkable.
for r in $RANKS; do
  if [ -f "$LOGS/rank${r}.refine.done" ]; then echo "$(date '+%F %T') skip rank$r refine (done)"; continue; fi
  check_headroom
  echo "$(date '+%F %T') === rank$r REFINE START ($(wc -l < "$QDIR/rank${r}_pairs.txt") pairs)"
  python scripts/run_scenediff_diagnostic.py --root "$ROOT" --experiment "$EXP" --config "$CONFIG" \
      --pair-ids-file "$QDIR/rank${r}_pairs.txt" --queries-file "$QDIR/rank${r}.json" \
      --refine-worker-dir "$WORKER" --through refine > "$LOGS/rank${r}.refine.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- rank$r refine exit $rc  (reconstruction cache hits: $(grep -c 'reference reconstruction: cached' "$LOGS/rank${r}.refine.log"))"
  if [ $rc -ne 0 ]; then
    echo "$(date '+%F %T') rank$r REFINE FAILED -- skipping its detect; other ranks continue"
    grep -E "Traceback|Error|FAILED" "$LOGS/rank${r}.refine.log" | tail -3
    continue
  fi
  touch "$LOGS/rank${r}.refine.done"
done

# The worker has done its job: refine is now cached for every rank, so step 3 never calls it.
# Release its ~5 GB of GPU before the detect stage, which needs SAM3 + DINOv2 + SAM2 resident.
pkill -f "refine_worker.py --dir $WORKER" 2>/dev/null && { echo "$(date '+%F %T') released refine worker"; sleep 5; }

# --- step 2: movable-object whitelist on the new query frames ---------------------------
# One pass over the 25 pairs: every t1_* dir, refined render_t0 (--render-t0-source refine),
# identical prompts/threshold to the main arm. Existing rank-1 masks are cache hits.
if [ ! -f "$LOGS/masks.done" ]; then
  check_headroom
  echo "$(date '+%F %T') === masks START"
  conda run --no-capture-output -n goldilocs python scripts/scenediff_movable_masks.py \
      --root "$ROOT/SceneDiff" --out "$MASKS" --confidence-threshold 0.5 \
      --render-t0-source refine --pairs-file "$PAIRS" > "$LOGS/masks.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- masks exit $rc"
  if [ $rc -ne 0 ]; then echo "masks FAILED -- not starting detect"; tail -5 "$LOGS/masks.log"; exit 1; fi
  touch "$LOGS/masks.done"
else
  echo "$(date '+%F %T') skip masks (done)"
fi

# --- step 3: gated detect, one invocation per rank --------------------------------------
for r in $RANKS; do
  [ -f "$LOGS/rank${r}.refine.done" ] || { echo "$(date '+%F %T') skip rank$r detect (no refine)"; continue; }
  [ -f "$LOGS/rank${r}.detect.done" ] && { echo "$(date '+%F %T') skip rank$r detect (done)"; continue; }
  check_headroom
  echo "$(date '+%F %T') === rank$r DETECT START ($(wc -l < "$QDIR/rank${r}_pairs.txt") queries)"
  python scripts/run_scenediff_diagnostic.py --root "$ROOT" --experiment "$EXP" --config "$CONFIG" \
      --pair-ids-file "$QDIR/rank${r}_pairs.txt" --queries-file "$QDIR/rank${r}.json" \
      --movable-mask-root "$MASKS" --dump-inventory \
      --detect-extra-args=--sequential-model-lifecycle > "$LOGS/rank${r}.detect.log" 2>&1
  rc=$?; echo "$(date '+%F %T') --- rank$r detect exit $rc"
  if [ $rc -eq 0 ]; then
    cp "$ROOT/SceneDiff/_experiments/$EXP/summary.json" "$LOGS/rank${r}.summary.json"; touch "$LOGS/rank${r}.detect.done"
    grep -E "pooled IoU" "$LOGS/rank${r}.detect.log" | tail -1
  else
    echo "$(date '+%F %T') rank$r DETECT FAILED rc=$rc -- continuing; re-run to retry"
    grep -E "RECONSTRUCTION FAILED|EVAL FAILED|Traceback|FileNotFoundError|No space left" "$LOGS/rank${r}.detect.log" | tail -3
  fi
done
pkill -f "refine_worker.py --dir $WORKER" 2>/dev/null
echo "$(date '+%F %T') MULTIQUERY_DONE"
