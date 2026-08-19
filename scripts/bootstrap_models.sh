#!/usr/bin/env bash
set -euo pipefail

# Upstream MASt3R and DINOv2 intentionally ship without installable packaging
# for the exact revisions this pipeline was validated against. Keep their
# pinned source trees under src/, install what does have packaging (SAM2),
# and let ocmask.model_paths / the dinov2 adapter expose those source roots
# at runtime.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mast3r_root="${repo_root}/src/mast3r"
mast3r_commit="f5209afc300cec36239a7ac992263f36847bbba0"

if [[ ! -d "${mast3r_root}/.git" ]]; then
  git clone --recursive https://github.com/naver/mast3r.git "${mast3r_root}"
fi

actual_commit="$(git -C "${mast3r_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${mast3r_commit}" ]]; then
  echo "MASt3R checkout has unexpected commit ${actual_commit}." >&2
  echo "Expected ${mast3r_commit}; move or inspect ${mast3r_root} before retrying." >&2
  exit 1
fi

git -C "${mast3r_root}" submodule update --init --recursive

dinov2_root="${repo_root}/src/dinov2"
dinov2_commit="7764ea0f912e53c92e82eb78a2a1631e92725fc8"

if [[ ! -d "${dinov2_root}/.git" ]]; then
  git clone https://github.com/facebookresearch/dinov2.git "${dinov2_root}"
fi

actual_dinov2_commit="$(git -C "${dinov2_root}" rev-parse HEAD)"
if [[ "${actual_dinov2_commit}" != "${dinov2_commit}" ]]; then
  echo "DINOv2 checkout has unexpected commit ${actual_dinov2_commit}." >&2
  echo "Expected ${dinov2_commit}; move or inspect ${dinov2_root} before retrying." >&2
  exit 1
fi

python -m pip install -r "${repo_root}/requirements-models.txt"

python - <<'PY'
from ocmask.model_paths import configure_mast3r_paths

configure_mast3r_paths()
import dust3r
import mast3r
import sam2

print("MASt3R/DUSt3R/SAM2 source imports succeeded.")
PY

cat <<'MSG'

MASt3R, its DUSt3R/CroCo submodules, and DINOv2 are checked out under src/.
SAM2 is installed as a Python package.

SAM3 and SAM3.1 are NOT handled by this script: at the time this pipeline was
built they were obtained outside any public, scriptable install path. You
must supply your own checkout and checkpoints, then point the pipeline at
them with three environment variables (configs/stages/*.yaml reference these
via ${SAM3_SOURCE}-style placeholders, expanded by ocmask.config.load_config):

  export SAM3_SOURCE=/path/to/your/sam3/checkout
  export SAM3_IMAGE_CHECKPOINT=/path/to/sam3.pt
  export SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt

Stages 2, 3, 7 (via stage 3), 8, 9, and 10 all load SAM3 or SAM3.1 through
these three variables. See README.md's "External SAM3/SAM3.1 dependency"
section before running the pipeline.

MSG
