#!/bin/bash
# SceneDiff overnight ablations (2026-09-10) on the preregistered 10-pair
# diagnostic subset, one T1 query per pair, DINOv2 off everywhere.
#
#   0  scenediff_v10_no_dino                baseline (configs/ablate_v10_no_dino.yaml),
#                                           stage-1-3 inventory dumped for replay
#   3  scenediff_v10_no_dino_no_recovery    recover_unmatched_via_tracking: false  (replay of 0's inventory)
#   4  scenediff_v10_no_dino_no_appearance  SAM3 features removed from correspondence (replay of 0's inventory)
#   2  scenediff_v10_no_dino_no_refine      refine.enabled: false (raw renders -> full detection rerun)
#   1  scenediff_v10_no_dino_ref{1,3,5}     N_T0 reference views (nested subsets of the baseline frame list);
#                                           ref10 == baseline (identical frame set + config)
#
# Every stage is cached/provenance-tracked by run_scenediff_diagnostic.py, so
# a rerun resumes. Usage: bash scripts/run_scenediff_overnight_ablations.sh
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT
WORKER=results/tidy_demo/refine_worker
LOGS=results/scenediff_diagnostic/SceneDiff/_experiments/_overnight_logs; mkdir -p "$LOGS"
DETECT_EXTRA=${DETECT_EXTRA:-}

run() {  # name config [extra runner args...]
  local name=$1 config=$2; shift 2
  echo "$(date '+%F %T') === $name ($config) $*"
  python scripts/run_scenediff_diagnostic.py --experiment "$name" --config "$config" \
      --refine-worker-dir "$WORKER" --detect-extra-args="$DETECT_EXTRA" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  grep -E "RECONSTRUCTION FAILED|EVAL FAILED|pooled IoU|Error|error" "$LOGS/$name.log" | grep -v "it/s" | tail -5
  echo "$(date '+%F %T') --- $name exit $rc"
}

run scenediff_v10_no_dino               configs/ablate_v10_no_dino.yaml --dump-inventory
run scenediff_v10_no_dino_no_recovery   configs/scenediff_v10_no_dino_no_recovery.yaml   --inventory-source-experiment scenediff_v10_no_dino
run scenediff_v10_no_dino_no_appearance configs/scenediff_v10_no_dino_no_appearance.yaml --inventory-source-experiment scenediff_v10_no_dino
run scenediff_v10_no_dino_no_refine     configs/scenediff_v10_no_dino_no_refine.yaml
for n in 1 3 5; do
  run "scenediff_v10_no_dino_ref$n" configs/ablate_v10_no_dino.yaml --reference-views "$n"
done
echo "$(date '+%F %T') OVERNIGHT_ABLATIONS_DONE"
