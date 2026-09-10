#!/bin/bash
# Completes the 2x4 (reference views N x refinement ON/OFF) SceneDiff factorial
# (2026-09-10). Runs ONLY the three missing OFF cells; N=10 ON/OFF and
# N=1/3/5 ON already exist and are not touched.
#
# One-variable pairing: each OFF cell uses configs/scenediff_v10_no_dino_no_refine.yaml,
# which is configs/ablate_v10_no_dino.yaml (the config every ON cell used) with
# exactly one semantic change, refine.enabled true->false (verified by parsed-YAML
# diff before launch). The reference-view count is a RUNNER flag
# (--reference-views N), not a config field, so it is set identically to the
# matching ON run and the deterministic nested frame subsets are reproduced
# bit-for-bit -- in fact the shared_ref<N> reconstruction/render artifacts are
# reused as cache hits, so the reconstruction inputs are literally the same files.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT
CONFIG=configs/scenediff_v10_no_dino_no_refine.yaml
LOGS=results/scenediff_diagnostic/SceneDiff/_experiments/_interaction_logs; mkdir -p "$LOGS"

for n in 1 3 5; do
  name="scenediff_v10_no_dino_ref${n}_no_refine"
  echo "$(date '+%F %T') === $name (N=$n, refine OFF)"
  python scripts/run_scenediff_diagnostic.py --experiment "$name" --config "$CONFIG" \
      --reference-views "$n" --detect-extra-args=--sequential-model-lifecycle \
      > "$LOGS/$name.log" 2>&1
  echo "$(date '+%F %T') --- $name exit $?"
  grep -E "pooled IoU|RECONSTRUCTION FAILED|EVAL FAILED" "$LOGS/$name.log" | tail -3
done
echo "$(date '+%F %T') INTERACTION_EXPERIMENT_DONE"
