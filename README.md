# Object-Level Scene Change Detection

Given two time-separated captures of a scene (each a short walkthrough video
or an unordered photo set, not a fixed before/after pair), localizes and
classifies added / removed / moved objects between them.

See [docs/METHODS.md](docs/METHODS.md) for the full method description and
current per-pair results. Short version: VGGT-Omega reconstructs both time
steps jointly from many frames each; a mutual depth-conflict filter builds a
canonical "clean" scene with add/remove transients pruned; both renders are
optionally passed through DI²FIX to remove residual splatting artifacts;
SAM3 + DINOv2 + SAM2 then resolve object-level changes between the two
renders and the real target photo, including a tracking-based recovery pass
for objects SAM3's automatic proposals only found in one of the two frames.

## Repository layout

```
src/ocmask_pipeline/
  reconstruction.py     # stage 1: VGGT-Omega + depth-conflict clean render
  refine.py              # stage 2 (optional): DI2FIX render refinement
  change_detection.py    # stage 3: SAM3/DINOv2/SAM2 object-state resolution
  geometry.py             # camera projection, point-cloud splatting, mutual
                           # depth-conflict filter (reverse_depth_filter)
  stages/, adapters/      # SAM3 proposal generator, SAM2 tracker, DINOv2
                           # feature extractor -- model wrappers, no
                           # orchestration logic
scripts/
  reconstruct.py, refine.py, detect.py   # one CLI per stage (see below)
  run_scenediff_pair.py                  # SceneDiff-benchmark-specific glue
                                          # (frame extraction, quirks below)
configs/pipeline.yaml    # all stage settings in one file
docs/METHODS.md          # method description + results
run_pipeline.sh          # chains the three stage CLIs across conda envs
```

## Setup    

Three separate reconstruction/generation models are involved
(VGGT-Omega, DI²FIX/Difix, and SAM3+SAM2+DINOv2), each with its own
conda env because their torch/diffusers pins conflict. This repo does not
vendor any of them -- point the config and env vars below at your own
checkouts.

1. **VGGT-Omega** (stage 1): checkout + conda env with its own
   requirements. Set `reconstruction.vggt_omega_root` and
   `reconstruction.vggt_omega_checkpoint` in `configs/pipeline.yaml`.
2. **DI²FIX** (stage 2, optional): checkout of
   [DF3DV/DI2FIX](https://github.com/johnnylu305/DF3DV/tree/main/DI2FIX)
   (built on [Difix3D](https://github.com/nv-tlabs/Difix3D)) + its own conda
   env (`diffusers`, `torch`, `peft`). Set `refine.di2fix_root`. The
   `nvidia/difix_ref` checkpoint downloads from Hugging Face on first run
   (~5GB). Set `refine.enabled: false` in the config to skip this stage
   entirely.
3. **Detection env** (stage 3): a conda env with SAM3, SAM2, DINOv2, and
   `scipy`/`torch`/`PIL`. Needs:
   - `SAM3_SOURCE` and `SAM3_IMAGE_CHECKPOINT` set in the environment (SAM3
     has no public pip/checkpoint distribution at time of writing).
   - `checkpoints/sam2.1_hiera_large.pt` and
     `checkpoints/dinov2_vitb14_reg4_pretrain.pth` present (symlink or copy
     them in; see `checkpoints/manifest.json` for source/hashes).
   - `src/dinov2` present (symlink to a DINOv2 source checkout -- the
     feature extractor imports it directly, it is not pip-installed).

`run_pipeline.sh` looks for these three envs by name via
`VGGT_OMEGA_CONDA_ENV` / `DIFIX3D_CONDA_ENV` / `DETECTION_CONDA_ENV`
(defaults: `vggt-omega`, `difix3d`, `goldilocs` -- override to match your
setup).

## Usage

Generic (frames already extracted to disk):

```bash
export SAM3_SOURCE=/path/to/sam3
export SAM3_IMAGE_CHECKPOINT=/path/to/sam3.pt
./run_pipeline.sh out/my_pair \
  --t0-frames before/*.png --t1-frames after/*.png \
  --t0-reference-index 0 --t1-reference-index 0
# result: out/my_pair/result/{labels.png, overlay.png, objects_*.png, inference.json}
```

On a SceneDiff benchmark pair (handles that dataset's video-source and
frame-orientation quirks -- see docs/METHODS.md):

```bash
conda run -n vggt-omega python scripts/run_scenediff_pair.py \
  --benchmark-root /path/to/scenediff_benchmark \
  --pair-id kitchen_2_kitchen_3 \
  --output-dir out/kitchen_2_kitchen_3
```

Each stage can also be run standalone (`scripts/reconstruct.py`,
`scripts/refine.py`, `scripts/detect.py` -- each `--help` documents its own
arguments), which is useful for iterating on one stage without re-running
the others.

## Output

`result/` from any of the above contains:

- `render_t0.png`, `clean_render.png`, `target.png` -- the three aligned inputs
- `objects_t0.png`, `objects_clean.png`, `objects_t1.png` -- per-frame SAM3 object inventories
- `labels.png` / `labels_color.png` -- the per-pixel change-class raster (0 unchanged, 1 added, 2 removed, 3 moved, 5 replaced)
- `overlay.png` -- `labels_color.png` alpha-blended over the target photo
- `inference.json` -- per-object decisions, association evidence, tracking-recovery diagnostics, and stage timings

## Not carried over from the original repo

This is a from-scratch trim, not a copy: no MASt3R (single before/after
photo reconstruction is superseded by VGGT-Omega's multi-view approach
here), no ChangeSim benchmark/eval code, no warehouse-specific experiment
scripts, no accumulated one-off analysis scripts. If you need the
single-photo-pair path or ChangeSim reproduction, they remain in the
original `change_detect` repo.
