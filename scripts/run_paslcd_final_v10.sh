#!/bin/bash
# Final PASLCD evaluation with the shipped configuration (v10_no_dino).
#
# Canonical protocol = run_paslcd_benchmark.py defaults: all 10 datasets x 2
# instances x 25 queries = 500, full reference sets, DI2FIX refine on. Each
# instance runs as its own run_paslcd_scene.py process (a failure or OOM in
# one cannot take the others down); instances are ordered by descending
# reference-set size so the largest joint VGGT-Omega calls run first. Every
# stage is resumable: reference scenes are saved to reference_scenes/, per-query
# reconstruction is reused from its marker files, stage-1-3 detection bundles
# are dumped to intermediate/<stem>/inventory/, metrics rows are upserted.
#
# Usage (from the repo root, any machine with the three conda envs set up per
# SETUP.md and SAM3_SOURCE / SAM3_IMAGE_CHECKPOINT resolvable):
#   bash scripts/run_paslcd_final_v10.sh [OUTPUT_ROOT] [CONFIG]
# Run it under tmux/screen or systemd-run so it survives a disconnect:
#   tmux new -d -s paslcd_final 'bash scripts/run_paslcd_final_v10.sh'
# Afterwards: python scripts/summarize_paslcd_final.py --root <OUTPUT_ROOT>
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
OUT=${1:-results/paslcd_final_v10_no_dino}
CONFIG=${2:-configs/ablate_v10_no_dino.yaml}

# conda: whichever install is on this machine
if [ -n "${CONDA_EXE:-}" ]; then source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then source "$HOME/anaconda3/etc/profile.d/conda.sh"
else echo "conda not found" >&2; exit 1; fi
conda activate "${VGGT_CONDA_ENV:-vggt-omega}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_SOURCE=${SAM3_SOURCE:-$HOME/sam3}
if [ -z "${SAM3_IMAGE_CHECKPOINT:-}" ]; then
  SAM3_IMAGE_CHECKPOINT=$(ls "$HOME"/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt 2>/dev/null | head -1)
  export SAM3_IMAGE_CHECKPOINT
fi
[ -f "$SAM3_IMAGE_CHECKPOINT" ] || { echo "SAM3_IMAGE_CHECKPOINT not found: '$SAM3_IMAGE_CHECKPOINT'" >&2; exit 1; }
[ -d "$SAM3_SOURCE" ] || { echo "SAM3_SOURCE not found: '$SAM3_SOURCE'" >&2; exit 1; }
[ -d data/PASLCD ] || { echo "data/PASLCD missing (see SETUP.md)" >&2; exit 1; }

mkdir -p "$OUT/run_logs" "$OUT/provenance"
cp "$CONFIG" "$OUT/provenance/config_used.yaml"
git rev-parse HEAD > "$OUT/provenance/git_head.txt" 2>/dev/null
git status --short > "$OUT/provenance/git_status.txt" 2>/dev/null
git diff > "$OUT/provenance/git_diff_uncommitted.patch" 2>/dev/null
(hostname; uname -r; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; free -m | head -2; echo "SAM3_IMAGE_CHECKPOINT=$SAM3_IMAGE_CHECKPOINT") > "$OUT/provenance/machine.txt"
echo "bash scripts/run_paslcd_final_v10.sh $OUT $CONFIG" > "$OUT/provenance/command.txt"

INSTANCES="Lunch_room/Instance_1 Lunch_room/Instance_2 Cantina/Instance_1 Lounge/Instance_1 Zen/Instance_1 Garden/Instance_2 Cantina/Instance_2 Porch/Instance_2 Playground/Instance_1 Pots/Instance_1 Playground/Instance_2 Zen/Instance_2 Printing_area/Instance_1 Garden/Instance_1 Pots/Instance_2 Porch/Instance_1 Meeting_room/Instance_2 Lounge/Instance_2 Printing_area/Instance_2 Meeting_room/Instance_1"
COMMON="--output-root $OUT --config $CONFIG --reference-scene-cache-dir $OUT/reference_scenes --dump-inventory --resume"

n_rows() { grep -c "test_IMG" "$OUT/${1}_${2}/metrics.csv" 2>/dev/null || echo 0; }
n_refined() { ls -d "$OUT/${1}_${2}"/intermediate/*/refined 2>/dev/null | wc -l; }
wait_for_quiet_gpu() {
  while true; do
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    ram=$(free -m | awk 'NR==2{print $7}')
    if [ "$gpu" -lt 2500 ] && [ "$ram" -ge 8000 ]; then echo "$(date '+%F %T') machine quiet (gpu ${gpu}MiB used, ${ram}MB RAM available)"; return; fi
    echo "$(date '+%F %T') waiting: gpu_used=${gpu}MiB ram_avail=${ram}MB"; sleep 60
  done
}

echo "=== FINAL_START $(date '+%F %T') host=$(hostname) ==="
for attempt in 1 2; do
  echo "=== PASS $attempt start $(date '+%F %T') ==="
  for di in $INSTANCES; do
    D=${di%/*}; I=${di#*/}
    if [ "$(n_rows $D $I)" -ge 25 ] && [ "$(n_refined $D $I)" -ge 25 ]; then echo "$(date '+%F %T') $D/$I complete, skipping"; continue; fi
    wait_for_quiet_gpu
    echo "$(date '+%F %T') INSTANCE_START $D/$I"
    python scripts/run_paslcd_scene.py --dataset $D --instance $I $COMMON >> "$OUT/run_logs/${D}_${I}.log" 2>&1
    echo "$(date '+%F %T') INSTANCE_END $D/$I exit=$?"
  done
  echo "=== PASS $attempt exit=0 $(date '+%F %T') ==="
done

# Canonical benchmark_summary.csv/json over the datasets whose BOTH instances completed;
# anything else is reported as a failure by summarize_paslcd_final.py, not retried here.
DONE=""
for D in Cantina Garden Lounge Lunch_room Meeting_room Playground Porch Pots Printing_area Zen; do
  [ "$(n_rows $D Instance_1)" -ge 25 ] && [ "$(n_rows $D Instance_2)" -ge 25 ] && DONE="$DONE $D"
done
echo "=== SUMMARY over datasets:$DONE ==="
[ -n "$DONE" ] && python scripts/run_paslcd_benchmark.py --datasets $DONE $COMMON > "$OUT/run_logs/benchmark_summary_pass.log" 2>&1
python scripts/summarize_paslcd_final.py --root "$OUT" > "$OUT/run_logs/summarize.log" 2>&1
echo "=== FINAL_END $(date '+%F %T') exit=$? ==="
