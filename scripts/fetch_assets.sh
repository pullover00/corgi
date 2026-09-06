#!/usr/bin/env bash
# Best-effort setup automation for a new machine -- see SETUP.md for the
# full picture and why each of these is needed. Clones the pinned sibling
# repos and downloads the two checkpoints with a known public source;
# prints exact manual steps for everything else (MASt3R's checkpoint,
# DI2FIX, the PASLCD dataset) rather than guessing at them.
#
# Idempotent: safe to re-run, skips anything already present.
#
# Override EXTERNAL_ROOT to clone the sibling repos somewhere other than
# ./external (e.g. a shared location already holding some of them).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$REPO_ROOT/external}"
mkdir -p "$EXTERNAL_ROOT" "$REPO_ROOT/checkpoints"

clone_pinned() {
  local name="$1" url="$2" commit="$3"
  local dest="$EXTERNAL_ROOT/$name"
  if [ -d "$dest/.git" ]; then
    echo "[$name] already present at $dest, checking out pinned commit"
    git -C "$dest" fetch --depth 1 origin "$commit" 2>/dev/null || true
    git -C "$dest" checkout -q "$commit" 2>/dev/null || echo "  warning: could not check out $commit -- verify manually"
  else
    echo "[$name] cloning $url @ $commit"
    git clone -q "$url" "$dest"
    git -C "$dest" checkout -q "$commit"
  fi
}

echo "=== 1. Sibling repos (see SETUP.md section 1) ==="
clone_pinned dinov2       https://github.com/facebookresearch/dinov2.git       7764ea0f912e53c92e82eb78a2a1631e92725fc8
clone_pinned mast3r       https://github.com/naver/mast3r.git                 f5209afc300cec36239a7ac992263f36847bbba0
clone_pinned sam3         https://github.com/facebookresearch/sam3.git        46957e47805eaa273f4aa7bbbd25a88bca9108ce
clone_pinned vggt-omega   https://github.com/facebookresearch/vggt-omega.git  39a0cb8af88554f15ddcb5354cd52bde588fa014

echo ""
echo "=== 2. Checkpoints with a known public source (see SETUP.md section 2) ==="
python3 - "$REPO_ROOT" <<'PYEOF'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not installed in this Python -- run this script's checkpoint")
    print("section again from inside the goldilocs/vggt-omega conda env, or:")
    print("  pip install huggingface_hub")
    sys.exit(0)

targets = [
    ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", "sam2.1_hiera_large.pt"),
    ("facebookresearch/dinov2_vitb14_reg4_pretrain", "dinov2_vitb14_reg4_pretrain.pth", "dinov2_vitb14_reg4_pretrain.pth"),
]
for repo_id, filename, dest_name in targets:
    dest = repo_root / "checkpoints" / dest_name
    if dest.exists():
        print(f"[{dest_name}] already present, skipping")
        continue
    print(f"[{dest_name}] downloading from {repo_id} ...")
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        dest.symlink_to(path)
        print(f"  -> symlinked to {path}")
    except Exception as exc:  # noqa: BLE001 -- best-effort, report and move on
        print(f"  could not auto-download ({exc}); get it manually per SETUP.md section 2")
PYEOF

echo ""
echo "=== 3. src/dinov2 symlink ==="
if [ -L "$REPO_ROOT/src/dinov2" ] || [ -d "$REPO_ROOT/src/dinov2" ]; then
  echo "src/dinov2 already present, skipping"
else
  ln -s "$EXTERNAL_ROOT/dinov2" "$REPO_ROOT/src/dinov2"
  echo "symlinked src/dinov2 -> $EXTERNAL_ROOT/dinov2"
fi

echo ""
echo "=== Still manual (see SETUP.md) ==="
echo "  - MASt3R checkpoint: checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
echo "    (get from naver/mast3r's own release/HF page, no auto-download here)"
echo "  - DI2FIX checkout (only needed if refine.enabled: true in your config;"
echo "    every result in this project's design-log artifact used --skip-refine)"
echo "  - PASLCD dataset -> data/PASLCD/ (not redistributed by this repo)"
echo "  - conda envs: create vggt-omega / goldilocs / difix3d and"
echo "    'pip install -r envs/<name>.requirements.txt' (see SETUP.md section 3"
echo "    for why torch/CUDA should be installed by hand first)"
echo "  - set reconstruction.vggt_omega_root in configs/pipeline.yaml to:"
echo "      $EXTERNAL_ROOT/vggt-omega"
echo "  - set reconstruction_mast3r.mast3r_root in configs/pipeline.yaml to:"
echo "      $EXTERNAL_ROOT/mast3r"
echo "  - export SAM3_SOURCE=$EXTERNAL_ROOT/sam3 (plus SAM3_IMAGE_CHECKPOINT, see SETUP.md)"
