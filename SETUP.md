# Setup for a new machine

This repo's own code is small; the weight is in three sibling model repos,
several GB of checkpoints, and the PASLCD dataset, none of which are tracked
here. This is the accurate, current list of everything else you need,
written after actually re-deriving several of these paths the hard way once
already this project (see the SAM3 section below) -- follow it rather than
re-discovering it.

Run `scripts/fetch_assets.sh` first; it automates the parts that can be
automated (cloning the pinned sibling repos, downloading the two checkpoints
with known public sources) and prints exactly what's left to do manually
(MASt3R's checkpoint, DI2FIX, PASLCD) with the real commands.

## 1. Sibling repos (code, not checkpoints)

All four are external projects this repo imports from or shells out to. Pin
to these exact commits for a reproducible environment; `fetch_assets.sh`
clones and checks each one out automatically.

| Repo | URL | Commit | Used as |
|---|---|---|---|
| DINOv2 | https://github.com/facebookresearch/dinov2.git | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | `src/dinov2` symlink target (imported directly, not pip-installed) |
| MASt3R | https://github.com/naver/mast3r.git | `f5209afc300cec36239a7ac992263f36847bbba0` | `reconstruction_mast3r.py`'s `mast3r_root` (MASt3R-vs-VGGT-Omega ablation only -- not needed for the main pipeline) |
| SAM3 | https://github.com/facebookresearch/sam3.git | `46957e47805eaa273f4aa7bbbd25a88bca9108ce` | `SAM3_SOURCE` env var target |
| VGGT-Omega | https://github.com/facebookresearch/vggt-omega.git | `39a0cb8af88554f15ddcb5354cd52bde588fa014` | `reconstruction.vggt_omega_root` in `configs/pipeline.yaml` |

DI2FIX has no git history on the machine this was authored on (a plain
checkout, not a clone) -- get it from
[DF3DV/DI2FIX](https://github.com/johnnylu305/DF3DV/tree/main/DI2FIX)
directly, or skip it: `refine.enabled: false` in the config disables this
stage entirely and every result in `docs/goldilocs_analysis.html` /
the design-log artifact was produced with it disabled (`--skip-refine`
everywhere).

If you have access to `pullover00/goldilocs` (the predecessor repo this one
was trimmed from), several of the above were vendored inside it at
`src/dinov2`, `src/mast3r`, and its own `checkpoints/` -- that's the fastest
source to copy from if you have SSH access to a machine that already has it,
rather than a fresh clone + checkpoint re-download.

## 2. Checkpoints

| File | Source | SHA256 |
|---|---|---|
| `checkpoints/sam2.1_hiera_large.pt` | `facebook/sam2.1-hiera-large` (Hugging Face) | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| `checkpoints/dinov2_vitb14_reg4_pretrain.pth` | `facebookresearch/dinov2_vitb14_reg4_pretrain` (Hugging Face) | `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | MASt3R's own release (naver/mast3r repo/HF page) | not pinned here -- verify against MASt3R's own published hash |
| SAM3 image checkpoint | `facebook/sam3` (Hugging Face) | see the HF repo directly |
| VGGT-Omega checkpoint | its own repo/release -- set `reconstruction.vggt_omega_checkpoint` in the config once obtained | -- |

**SAM3's environment variables** (this cost real time to re-derive once
already -- don't re-discover it, just use these, adjusted to wherever you
put the sam3 checkout and Hugging Face cache on the new machine):

```bash
export SAM3_SOURCE=/path/to/sam3                # the cloned facebookresearch/sam3 checkout
export SAM3_IMAGE_CHECKPOINT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('facebook/sam3'))")/sam3.pt
```

Neither is set anywhere persistent (not in a conda env var, not in a shell
profile) on the machine this was authored on -- they must be exported
inline in whatever shell invokes `scripts/detect.py` / `scripts/detect_batch.py`
(directly, or via a parent process whose environment a `conda run -n
goldilocs` subprocess call inherits).

## 3. Conda environments

Three separate environments, because their torch/diffusers pins conflict:

| Env name | Python | Used for | Package list |
|---|---|---|---|
| `vggt-omega` | 3.13 | Reconstruction (stage 1), orchestration scripts | `envs/vggt-omega.requirements.txt` |
| `goldilocs` | 3.11 | Detection (stage 3: SAM3 + SAM2 + DINOv2) | `envs/goldilocs.requirements.txt` |
| `difix3d` | 3.10 | Refinement (stage 2, optional) | `envs/difix3d.requirements.txt` |

These are `pip freeze` exports from the machine this was authored on, not
hand-curated `environment.yml` files -- they pin exact versions including
CUDA-specific torch builds, which may not exist for a different CUDA/driver
setup on the new (stronger) machine. Recommended: create each env with the
right Python version first, install `torch`/`torchvision` matching the new
machine's CUDA version by hand, *then* `pip install -r envs/<name>.requirements.txt`
and let pip skip/resolve what's already satisfied, rather than a blind
`pip install -r ...` into an empty env.

`run_pipeline.sh` / the benchmark scripts look for these by name via
`VGGT_OMEGA_CONDA_ENV` / `DIFIX3D_CONDA_ENV` / `DETECTION_CONDA_ENV` env
vars if you name them differently.

## 4. The PASLCD dataset

`data/PASLCD/<Dataset>/<Instance>/{images,gt_mask,sparse}/` -- 20 scene
instances (10 locations x 2 instances), 25 query photos each, ~20GB total.
Not redistributed by this repo. If you don't already have a source for it,
ask whoever supplied it for this project originally; there is no
automated-download step for it in `fetch_assets.sh`.

## 5. Sanity check

Once everything above is in place:

```bash
conda run -n goldilocs python -c "
import sys; sys.path.insert(0, 'src')
from ocmask_pipeline.config import load_config
from ocmask_pipeline.change_detection import ThreeImageSettings
ThreeImageSettings.from_config(load_config('configs/pipeline.yaml'))
print('config OK')
"
SAM3_SOURCE=... SAM3_IMAGE_CHECKPOINT=... conda run -n goldilocs python -c "
import sys; sys.path.insert(0, 'src')
from ocmask_pipeline.stages.sam3_proposals import Sam3AutomaticMaskGenerator
import os
gen = Sam3AutomaticMaskGenerator(os.environ['SAM3_IMAGE_CHECKPOINT'], source=os.environ['SAM3_SOURCE'])
gen.load()
print('SAM3 load OK')
"
```

Then a single real query end-to-end, e.g.:

```bash
conda run -n vggt-omega python scripts/run_paslcd_scene.py \
  --dataset Cantina --instance Instance_1 --limit 1 --skip-refine
```
