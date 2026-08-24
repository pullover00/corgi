#!/usr/bin/env bash
# Run the full, unmodified object-consistent-masks method on every
# Warehouse_8 and Warehouse_9 pair in ChangeSim (6,090 pairs total: 2,373 +
# 3,717) -- the complement of scripts/run_warehouse67_experiment.sh, meant
# for a second machine so the two halves of the 8,212-pair table3 protocol
# run in parallel on separate GPUs.
#
# This is a thin wrapper around `ocmask evaluate changesim --full-pipeline`:
# every reliability property (one clean process per pair, resumable
# checkpointing via progress.jsonl, atomic per-pair freeze markers, a
# run-fingerprinted output directory that refuses to silently mix runs with
# different config/code/checkpoints) already lives in that command. This
# script only fixes the arguments for this specific experiment and adds
# --save-stage-artifacts, which additionally persists every stage's
# intermediate evidence (proposals, descriptors, similarity matrices,
# accept/reject reasons, before/after label maps) for later ablations --
# see src/ocmask/artifact_capture.py. Nothing here changes any threshold,
# model, or decision rule: the pipeline config is read as-is from
# configs/pipeline.yaml.
#
# Run it directly in a terminal to watch the tqdm progress bar:
#   ./scripts/run_warehouse89_experiment.sh
#
# Interrupting (Ctrl-C) and re-running is safe: already-frozen pairs are
# skipped (validated against their pair_manifest.json), and in-progress
# pairs resume from the beginning of that one pair, not the whole run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- Experiment identity -----------------------------------------------
# A stable (not timestamped) default so re-running this script resumes the
# same run instead of starting a new one. Override to start a distinct run.
OUTPUT_DIR="${OCMASK_WAREHOUSE89_OUTPUT:-$REPO/out/warehouse89_experiment}"
MANIFEST="$REPO/data/changesim/manifest-warehouse89.jsonl"
PIPELINE_CONFIG="$REPO/configs/pipeline.yaml"

# --- Model source/checkpoint environment --------------------------------
# Only set here if the caller hasn't already exported a value, so an
# operator's own environment always wins. These defaults are this
# repository's original development-machine paths; on a different machine,
# export SAM3_SOURCE / SAM3_IMAGE_CHECKPOINT (and SAM31_CHECKPOINT if used)
# yourself before running this script. See README.md's "External SAM3
# dependency" and "Setting up a second machine" sections for what these
# must point at.
: "${SAM3_SOURCE:=/home/tessa/sam3}"
: "${SAM3_IMAGE_CHECKPOINT:=/home/tessa/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt}"
: "${SAM31_CHECKPOINT:=/home/tessa/gaussian-grouping/sam3.1/sam3.1_multiplex.pt}"
export SAM3_SOURCE SAM3_IMAGE_CHECKPOINT SAM31_CHECKPOINT

for var_name in SAM3_SOURCE SAM3_IMAGE_CHECKPOINT SAM31_CHECKPOINT; do
    path="${!var_name}"
    if [[ ! -e "$path" ]]; then
        echo "error: \$$var_name points at a path that does not exist: $path" >&2
        echo "Fix the environment variable (or edit the default in this script) and rerun." >&2
        exit 1
    fi
done

if [[ ! -f "$MANIFEST" ]]; then
    echo "error: manifest not found: $MANIFEST" >&2
    echo "Regenerate it from data/changesim/manifest-table3.jsonl (filter id startswith Warehouse_8_/Warehouse_9_)." >&2
    exit 1
fi

PAIR_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
echo "=== Warehouse_8 + Warehouse_9 full-pipeline experiment ==="
echo "Repository:      $REPO"
echo "Manifest:        $MANIFEST ($PAIR_COUNT pairs)"
echo "Pipeline config: $PIPELINE_CONFIG (unmodified)"
echo "Output:          $OUTPUT_DIR"
echo "Stage artifacts: enabled (--save-stage-artifacts)"
echo "GPU:             $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
echo "==========================================================="
echo

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n goldilocs \
    python -m ocmask.cli evaluate changesim \
    --manifest "$MANIFEST" \
    --output "$OUTPUT_DIR" \
    --pipeline-config "$PIPELINE_CONFIG" \
    --full-pipeline \
    --save-stage-artifacts \
    --continue-on-error \
    --pair-retries 1 \
    --pair-timeout-seconds 1800 \
    --prediction-variant guarded
