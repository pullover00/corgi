#!/usr/bin/env bash
# Chains the three pipeline stages, each in its own conda env (they have
# incompatible torch/diffusers pins -- see README.md).
#
# Usage:
#   ./run_pipeline.sh OUTPUT_DIR --t0-frames f1.png f2.png ... --t1-frames g1.png g2.png ...
#     [--t0-reference-index N] [--t1-reference-index N] [--skip-refine]
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 OUTPUT_DIR --t0-frames <frames...> --t1-frames <frames...> [--t0-reference-index N] [--t1-reference-index N] [--skip-refine]" >&2
  exit 1
fi

OUTPUT_DIR="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_REFINE=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" == "--skip-refine" ]; then
    SKIP_REFINE=1
  else
    ARGS+=("$arg")
  fi
done

VGGT_ENV=${VGGT_OMEGA_CONDA_ENV:-vggt-omega}
DIFIX_ENV=${DIFIX3D_CONDA_ENV:-difix3d}
DETECT_ENV=${DETECTION_CONDA_ENV:-goldilocs}

mkdir -p "$OUTPUT_DIR"

echo "== stage 1: reconstruction ($VGGT_ENV) =="
conda run -n "$VGGT_ENV" python "$REPO/scripts/reconstruct.py" \
  "${ARGS[@]}" --output-dir "$OUTPUT_DIR/reconstruction"

RENDER_T0="$OUTPUT_DIR/reconstruction/render_t0.png"
CLEAN_RENDER="$OUTPUT_DIR/reconstruction/clean_render.png"
IMAGE_T1="$OUTPUT_DIR/reconstruction/image_t1.png"

if [ "$SKIP_REFINE" -eq 0 ]; then
  echo "== stage 2: DI2FIX refinement ($DIFIX_ENV) =="
  conda run -n "$DIFIX_ENV" python "$REPO/scripts/refine.py" \
    --render-t0 "$RENDER_T0" --clean-render "$CLEAN_RENDER" --image-t1 "$IMAGE_T1" \
    --output-dir "$OUTPUT_DIR/refined"
  RENDER_T0="$OUTPUT_DIR/refined/render_t0.png"
  CLEAN_RENDER="$OUTPUT_DIR/refined/clean_render.png"
fi

echo "== stage 3: change detection ($DETECT_ENV) =="
conda run -n "$DETECT_ENV" python "$REPO/scripts/detect.py" \
  --render-t0 "$RENDER_T0" --clean-render "$CLEAN_RENDER" --image-t1 "$IMAGE_T1" \
  --output-dir "$OUTPUT_DIR/result"

echo "== done: $OUTPUT_DIR/result =="
