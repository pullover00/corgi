# Object-Consistent Masks

Scene change detection for RGB-D image pairs: given two photos of the same
place taken at different times, produce a per-pixel map of what changed,
classified as `added`, `removed`, `moved`, or `replaced`.

This repository is the **object-consistent full target masks** method: the
final, winning configuration extracted from a larger research repository
(internally: `GOLDILOCS`) and pruned down to only the code path that produces
it. It edits a frozen baseline change-detection raster ("R4") by finding old
objects whose identity disappears in the same aligned image slot, gating
candidates on whether they are a plausible object, expanding to compatible
neighboring fragments, resolving depth-authoritative ownership against
`REMOVED` pixels, cleaning up obsolete source footprints, arbitrating class
consensus across every coherent target object, and suppressing floor-dominant
`ADDED` components.

## Result

Evaluated on 25 ChangeSim pairs (`fixed10` + `new15`), against a frozen R4
baseline (`r4_no_geometry_ablation`) that this method edits:

| method | changed IoU | unchanged IoU | binary mIoU | added IoU | removed IoU | moved IoU | replaced IoU | multiclass mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 baseline (frozen input) | 41.88 | 94.34 | 68.11 | 36.80 | 31.30 | 17.62 | 14.92 | 39.00 |
| **Object-consistent full target masks** | **42.80** | **94.50** | **68.65** | **38.53** | **33.65** | **17.79** | **21.18** | **41.13** |
| Object-consistent guarded output (more conservative sibling) | 42.57 | 94.48 | 68.53 | 38.53 | 33.65 | 17.79 | 20.05 | 40.90 |

The full-mask variant is what this repository is named for and what
`scripts/run_slot_inconsistency_replacement_experiment.py` writes to
`labels_full_mask.png` (`report.json`'s `full_mask_candidate`). Its guarded
sibling (`labels.png` / `labels_guarded.png`) restricts a confirmed
replacement to a more conservative footprint; both are computed by the same
run and both are still written, since they're cheap byproducts of one
decision pipeline, but full-mask is the one to use.

`docs/method.md` has the full decision procedure (object plausibility,
companion expansion, depth-authoritative ownership, source-footprint cleanup,
dataset-wide target-object arbitration, floor suppression) with worked
failure-case examples, carried over from the original research report.

## Two important caveats before you run this

1. **Not independently re-executed against real models and data.** This
   repository was assembled by pruning a much larger research repository
   down to the dependency chain that produces the result above, then
   rewiring package names, import paths, and output directory conventions.
   The result table above is taken from the original run's frozen
   `report.json`. What has been verified in this checkout: every one of the
   12 scripts under `scripts/` imports cleanly end to end (`python
   scripts/run_X.py --help` fully resolves every module in its dependency
   graph, including cross-script imports), the 36-case unit test suite in
   `tests/` passes (`pytest tests/`, covering the core decision logic in
   `slot_inconsistency.py` and the consolidation logic in `branch_b2.py`
   directly), every config's YAML parses, and the config graph is internally
   consistent (every stage's declared parent-output path matches another
   stage's declared output path). What has **not** been verified: an actual
   run through MASt3R/SAM2/SAM3/SAM3.1/DINOv2 inference on real ChangeSim
   data from this checkout, which requires GPU compute and the external SAM3
   dependency below that weren't available while assembling this port.
   Treat a first full run as a verification step, not an assumption.
2. **SAM3 and SAM3.1 are external dependencies you must supply yourself.**
   See "External SAM3/SAM3.1 dependency" below -- this is the single biggest
   setup obstacle and there is no way around it.

## Pipeline architecture

The method is the last of 11 stages. Every stage is independently runnable
and caches its outputs by pair, so a partial run resumes cheaply. Stages 2
and 3 each run twice per split, at two different SAM3 proposal-grid
densities -- see "Two SAM3 grid densities" below.

| # | Stage | Script | What it does |
|---|---|---|---|
| 1 | Reconstruction + SAM2 baseline | `ocmask` CLI (`evaluate changesim`) | MASt3R sparse reconstruction and dense point-map recovery, point-cloud rendering of each view into the other, SAM2 automatic segmentation, a first change raster. |
| 2 | SAM3 proposals + SAM3.1 tracking | `scripts/run_sam3_pairwise_experiment.py` | Automatic SAM3 mask proposals over the aligned source render and the real target image; SAM3.1 multiplex tracking between them. |
| 3 | SAM2 tracking-v4 baseline | `scripts/run_sam3_pairwise_experiment.py` (different config) | Re-tracks stage 2's proposals with SAM2 instead of SAM3.1; this is the `parent_evaluation` every later stage treats as ground truth about proposal acceptance/rejection. |
| 4 | SAM3 dense features | `scripts/run_sam3_identity_location_experiment.py` | Dense per-pair SAM3 embedding maps, used for appearance-identity comparisons. |
| 5 | DINOv2 dense features | `scripts/run_dinov2_identity_location_experiment.py` | Same role as stage 4, a second independent appearance signal. |
| 6 | Moved-object association cache | `scripts/run_sam3_moved_association_experiment.py` | Forward/backward SAM2 propagation cached for every proposal stage 3 rejected as changed; the final method reads only the raw cache, not this stage's own association variants. |
| 7 | Guarded hybrid | `scripts/run_sam3_guarded_hybrid_experiment.py` | Merges identity/moved evidence onto the stage-3 raster from cached evidence only (no model calls); only its `replacement_only` variant is consumed downstream. |
| 8 | Feature-veto gate (A0-A4) | `scripts/run_sam3_feature_veto_gate_experiment.py` | Five ablation variants of a same-place feature-veto gate; only `a3_guarded_direct_replacement` is consumed downstream. |
| 9 | Obvious-object sentinel | `scripts/run_obvious_object_sentinel_experiment.py` | Fresh SAM3 pass over the real (un-warped) T0 image plus targeted SAM2 "verified absence" tracking; composes four variants, only the `a3_guarded_direct_replacement`-based raster is consumed downstream. |
| 10 | Real-image association resolver ("R4") | `scripts/run_real_image_association_resolver.py` | Five resolver variants (`r0`-`r4`); **`r4_no_geometry_ablation` is the frozen baseline the final method edits.** |
| 11 | **Object-consistent replacement (this method)** | `scripts/run_slot_inconsistency_replacement_experiment.py` | Edits R4 as described above. Writes `labels_full_mask.png` (the headline result), `labels_guarded.png`, `report.json`, and a self-contained `index.html` visual report. |

Run the whole thing with:

```bash
python scripts/run_pipeline.py --splits fixed10,new15
```

or one split, or resume from a partially-completed run (each stage script
skips pairs it already has a valid cache for) -- see `scripts/run_pipeline.py --help`.

### Two SAM3 grid densities

Stages 2 and 3 run at `points_per_side=96` ("grid96") for every stage from 4
onward *except* stage 5. Stage 5's config (inherited as-is from the original
research repository, where it predates the grid96 ablation) reads its parent
proposals at the original `points_per_side=64` ("grid64"). We traced through
stage 5's code and confirmed its dense DINOv2 feature map is a function only
of the raw source/target pixels, not of the specific SAM3 proposal masks --
so this is very likely harmless -- but we did not execute the pipeline to
verify the two lineages produce byte-identical features. `run_pipeline.py`
reproduces this exactly rather than "fixing" it, because fixing it would mean
diverging from the configuration that produced the validated result above.
If you want to investigate or resolve this, start at `configs/stages/changesim-dinov2-identity-location-*-hires.yaml`.

### External SAM3/SAM3.1 dependency

SAM3 and SAM3.1 are used by stages 2, 3 (tracking-v4 only reuses stage 2's
proposals, but stage 9 runs SAM3 fresh), 8, 9, and 10. In the original
research environment they were obtained outside any public, scriptable
install path, so `scripts/bootstrap_models.sh` cannot vendor them the way it
vendors MASt3R and DINOv2. You need your own SAM3/SAM3.1 checkout and
checkpoints, then export:

```bash
export SAM3_SOURCE=/path/to/your/sam3/checkout
export SAM3_IMAGE_CHECKPOINT=/path/to/sam3.pt
export SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt
```

`ocmask.config.load_config` expands `${VAR}`-style placeholders in every
config, so these three variables are all that's needed -- no config file
should ever need editing for this.

## Setup

```bash
conda env create -f environment.yml
conda activate ocmask
./scripts/bootstrap_models.sh   # vendors MASt3R + DINOv2, installs SAM2
```

Then, after accepting the relevant upstream licenses, download:

- MASt3R: `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` to
  `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
- SAM2: `facebook/sam2.1-hiera-large` to `checkpoints/sam2.1_hiera_large.pt`
- DINOv2: `dinov2_vitb14_reg4_pretrain` to `checkpoints/dinov2_vitb14_reg4_pretrain.pth`
- SAM3 / SAM3.1: see "External SAM3/SAM3.1 dependency" above

```bash
python scripts/verify_checkpoints.py --record   # first machine: record hashes
python scripts/verify_checkpoints.py            # subsequent machines: verify
ocmask doctor                                    # reports CUDA/model/checkpoint readiness
```

Sanity-check the install itself (no GPU, checkpoints, or dataset needed) with:

```bash
pip install -e .
pytest tests/
```

### Dataset

This method is evaluated on [ChangeSim](https://github.com/SAMMiCA/ChangeSim).
Place warehouse sequences under `data/changesim/Warehouse_*/...` matching the
official layout; `data/changesim/manifest-table3.jsonl` (8,212 pairs, the
`fixed10` split is a deterministic seeded subset of this) and
`data/changesim/manifest-new15.jsonl` (15 pairs, a disjoint held-out check)
are included in this repository and reference that layout with relative
paths. To evaluate on a different ChangeSim warehouse:

```bash
python scripts/build_changesim_manifest.py data/changesim/Warehouse_6
```

## Repository layout

```
src/ocmask/            core library: reconstruction pipeline, adapters
                        (MASt3R/SAM2/DINOv2), metrics, config loading
src/ocmask/stages/      stages 2-11's algorithmic core (proposals, tracking,
                        identity/feature comparison, the resolver variants,
                        and slot_inconsistency.py -- the method itself)
scripts/                one runner script per stage, plus run_pipeline.py
configs/stage01_*.yaml  stage 1 config
configs/stages/         stage 2-11 configs, one (or one per split) each
data/changesim/         ChangeSim manifests (dataset images not included)
checkpoints/            model weights go here (not committed) + manifest.json
docs/method.md          full decision procedure + worked failure-case examples
```

## Evaluation discipline

`scripts/run_slot_inconsistency_replacement_experiment.py` writes predictions
and their SHA-256 hashes to `predictions_frozen.json` before ground truth is
opened; ground truth is used only afterward, for scoring and the visual
failure audit in `index.html`. This mirrors the discipline used throughout
the original research repository and is why every stage script validates its
declared parent's `selection.json`/`report.json` before trusting it.

## Licenses

The code in this repository is MIT-licensed (see `LICENSE`). It orchestrates
third-party models, each under its own license: MASt3R is CC BY-NC-SA 4.0
(non-commercial research use), SAM2 and DINOv2 are Apache-2.0, and SAM3/SAM3.1
carry whatever license you obtained them under. ChangeSim has its own dataset
license. Check all of these before redistributing a configured environment or
its outputs.
