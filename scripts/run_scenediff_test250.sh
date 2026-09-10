#!/bin/bash
# Held-out SceneDiff run: the official 250-pair test split, v10_no_dino with
# refine OFF, one fixed T1 query per pair. Pre-registered in
# docs/experiments/scenediff_test250.md BEFORE launch; nothing here is to be
# changed after results are seen.
#
# Chunked (10 x 25 pairs) because the runner commits detect-stage manifests
# only after detect_batch.py finishes a whole batch: a kill mid-batch loses
# every pair of that batch (this is what the 2026-09-10 OOM cost). A chunk
# bounds that loss to ~1 h. Re-running this script skips chunks with a .done
# marker; inside an unfinished chunk the runner itself skips pairs whose
# labels are fresh.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vggt-omega
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
[ -n "${SAM3_IMAGE_CHECKPOINT:-}" ] || SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
export SAM3_IMAGE_CHECKPOINT

EXP=scenediff_v10_no_dino_no_refine
CONFIG=configs/scenediff_v10_no_dino_no_refine.yaml
ROOT=results/scenediff_test250
QUERIES=data/scenediff_benchmark/test250_queries_v2.json
CHUNKS=data/scenediff_benchmark/test_split_250_chunks
EXPDIR=$ROOT/SceneDiff/_experiments/$EXP
LOGS=$EXPDIR/logs; mkdir -p "$LOGS"

echo "$(date '+%F %T') TEST250_START commit=$(git rev-parse --short HEAD) config_sha=$(sha256sum $CONFIG | cut -c1-16) queries_sha=$(sha256sum $QUERIES | cut -c1-16)"
for c in "$CHUNKS"/chunk_*.txt; do
  n=$(basename "$c" .txt)
  if [ -f "$LOGS/$n.done" ]; then echo "$(date '+%F %T') skip $n (done)"; continue; fi
  echo "$(date '+%F %T') === $n START ($(wc -l < "$c") pairs)"
  python scripts/run_scenediff_diagnostic.py --root "$ROOT" --experiment "$EXP" --config "$CONFIG" \
      --pair-ids-file "$c" --queries-file "$QUERIES" \
      --detect-extra-args=--sequential-model-lifecycle > "$LOGS/$n.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') --- $n exit $rc"
  if [ $rc -eq 0 ]; then
    cp "$EXPDIR/summary.json" "$LOGS/$n.summary.json"
    cp "$EXPDIR/reference_sets.json" "$LOGS/$n.reference_sets.json" 2>/dev/null
    touch "$LOGS/$n.done"
    grep -E "pooled IoU" "$LOGS/$n.log" | tail -1
  else
    echo "$(date '+%F %T') $n FAILED rc=$rc -- continuing with next chunk; rerun this script to retry"
    grep -E "RECONSTRUCTION FAILED|EVAL FAILED|Traceback|Error" "$LOGS/$n.log" | tail -3
  fi
done
python3 scripts/scenediff_test250_aggregate.py --root "$ROOT" --experiment "$EXP"
echo "$(date '+%F %T') TEST250_DONE"
