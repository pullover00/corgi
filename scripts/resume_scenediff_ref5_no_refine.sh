#!/bin/bash
# Resumes ONLY the N=5 refine=OFF cell after the 2026-09-10 15:13 machine-wide
# OOM (systemd-oomd killed org.gnome.Shell + user@1000.service; our run was
# collateral damage, not a pipeline fault). Pairs 1-4 already completed under
# this exact config and are reused as ArtifactStore cache hits; pairs 5-10
# recompute. Identical invocation to the N=5 branch of
# run_scenediff_refine_reference_interaction.sh -- no config or flag changes.
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
name=scenediff_v10_no_dino_ref5_no_refine
echo "$(date '+%F %T') === $name (N=5, refine OFF) RESUME after OOM"
python scripts/run_scenediff_diagnostic.py --experiment "$name" --config "$CONFIG" \
    --reference-views 5 --detect-extra-args=--sequential-model-lifecycle \
    > "$LOGS/$name.resume.log" 2>&1
echo "$(date '+%F %T') --- $name exit $?"
grep -E "pooled IoU|RECONSTRUCTION FAILED|EVAL FAILED" "$LOGS/$name.resume.log" | tail -3
echo "$(date '+%F %T') INTERACTION_EXPERIMENT_DONE"
