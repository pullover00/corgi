# Object-Consistent Masks

Scene change detection for RGB-D image pairs: given two photos of the same
place taken at different times, produce a per-pixel map of what changed,
classified as `added`, `removed`, `moved`, or `replaced`.

This repository is the **object-consistent full target masks** method: a
complete, self-contained 11-stage pipeline, extracted from a larger research
repository (internally: `GOLDILOCS`) and pruned down to only the code path
that produces this result. All 11 stages are part of the method and part of
the contribution: MASt3R reconstruction, SAM2/SAM3 segmentation and tracking,
SAM3/DINOv2 appearance features, and a change-detection resolver (stages
1-10) build a first per-pixel change prediction; the final stage then
refines it into the object-consistent result the method is named for, by
finding old objects whose identity disappears in the same aligned image
slot, gating candidates on whether they are a plausible object, expanding to
compatible neighboring fragments, resolving depth-authoritative ownership
against `REMOVED` pixels, cleaning up obsolete source footprints, arbitrating
class consensus across every coherent target object, and suppressing
floor-dominant `ADDED` components.

Stages 1-10 are referred to below as "the base pipeline." It is not a
separate or borrowed prior result: it was built as part of the same effort
and is included here in full, not as a frozen artifact you have to supply
yourself.

## Quickstart: run it on your own two images

```bash
python demo.py --before before.png --after after.png --output out/
```

`--before` is the earlier-time photo, `--after` the later-time photo of
(approximately) the same place from the same viewpoint -- the output change
map is aligned to `--after`. This needs a working GPU environment; see
"Setup" below before your first run, and `ocmask doctor` to check readiness.
`out/labels.png` is the headline per-pixel prediction (`out/overlay.png` is
the same thing rendered over the "after" photo for a quick look); see
`demo.py --help` for the full list of files it writes.

To evaluate the method over a ChangeSim dataset instead of a single pair:

```bash
ocmask evaluate changesim --manifest data/changesim/manifest-new15.jsonl \
    --output out/changesim-eval/ --full-pipeline
```

This runs every pair in the manifest through the same full pipeline and
reports the paper's Table-3-style IoU metrics (`out/changesim-eval/report.json`,
also printed to stdout). See "Dataset" below for manifest details, and
"Evaluating on ChangeSim" for what this command does and does not do.

Both of these call the same underlying entrypoint,
`ocmask.inference.run_pair`, once per image pair.

## Result

Evaluated end to end on 25 ChangeSim pairs (a 10-pair development split plus
a disjoint 15-pair held-out check). The first row is what stages 1-10 alone
produce; the method is stages 1-11 together:

| pipeline | changed IoU | unchanged IoU | binary mIoU | added IoU | removed IoU | moved IoU | replaced IoU | multiclass mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stages 1-10 only (base pipeline) | 41.88 | 94.34 | 68.11 | 36.80 | 31.30 | 17.62 | 14.92 | 39.00 |
| **Stages 1-11 (object-consistent full target masks)** | **42.80** | **94.50** | **68.65** | **38.53** | **33.65** | **17.79** | **21.18** | **41.13** |
| Stages 1-11, guarded output (more conservative stage-11 footprint) | 42.57 | 94.48 | 68.53 | 38.53 | 33.65 | 17.79 | 20.05 | 40.90 |

The full-mask output is what this repository is named for and what
`run_pair`/`demo.py` writes to `labels.png` (stage 11's `labels_full_mask.png`
internally). Its guarded sibling (`labels_guarded.png`) restricts a
confirmed replacement to a more conservative footprint; both are computed by
the same run and both are still written, since they're cheap byproducts of
one decision pipeline, but full-mask is the one to use.

`docs/method.md` has the full decision procedure for stage 11 (object
plausibility, companion expansion, depth-authoritative ownership,
source-footprint cleanup, dataset-wide target-object arbitration, floor
suppression) with worked failure-case examples, carried over from the
original research report.

## Two important caveats before you run this

1. **All 11 stages have been run end to end for real** on this checkout,
   with real GPU compute, real model weights, and a real ChangeSim pair
   (not just import-checked): `run_pair` completed without exception and
   produced a `labels.png` whose changed region visibly matches what
   actually differs between the two real photos. See `docs/rewrite_plan.md`
   for exactly what that one real run covers (and does not -- one pair is
   not a full evaluation; running `ocmask evaluate changesim --full-pipeline`
   over a larger manifest is the natural next check) versus what is still
   validated only by careful reading of the original research scripts.
2. **SAM3 is an external dependency you must supply yourself.**
   See "External SAM3 dependency" below -- this is the single biggest setup
   obstacle and there is no way around it.

## Pipeline architecture

The method is all 11 stages together, run in one process by `run_pair`, end
to end from two raw RGB image pairs to the final object-consistent raster;
none of stages 1-10 is a third-party baseline.

| # | Stage | What it does |
|---|---|---|
| 1 | Reconstruction + SAM2 baseline | MASt3R sparse reconstruction and dense point-map recovery, point-cloud rendering of each view into the other, SAM2 automatic segmentation, a first change raster. |
| 2 | SAM3 automatic proposals | Class-agnostic SAM3 mask proposals over the aligned source render and the real target image. |
| 3 | SAM2 re-tracking | Re-tracks stage 2's proposals with SAM2; this raster is what every later stage refines. |
| 4 | SAM3 dense appearance features + identity calibration | A dense per-pair SAM3 embedding map for each image, pair-internal same/different-identity cosine calibration, and object-level identity/location/replacement classification used as evidence by later stages. |
| 5 | DINOv2 dense appearance features | A second, independent dense appearance signal used by stage 11. |
| 6 | Moved-candidate tracking | Forward/backward SAM2 propagation of exactly the proposals stage 3 flagged as changed; feeds stages 7 and 11. |
| 7 | Motion + replacement evidence fusion | Promotes same-place, confidently-different proposal pairs from stage 3's raster to `replaced`, and splits weakly-supported `moved` pixels back into `removed`/`added` where only one tracking direction actually supports them. |
| 8 | Feature-veto-gated direct replacement | Re-tracks exactly the proposals stage 4's appearance gate flagged, merges the result onto stage 7's raster, and paints the conservative intersection of every confidently-different same-place pair directly as `replaced`. |
| 9 | Real-image sentinel | A fresh SAM3 pass (proposals + features) over the real, un-warped "before" image, independent of MASt3R's rendering -- feeds stage 10's absence verification. |
| 10 | Real-image association resolver | Associates real "before"/"after" object instances by appearance and location, verifies unmatched endpoints are genuinely absent (not just untracked) with a targeted SAM2 check, and pairs two independently-verified-absent endpoints as a replacement. This is the base pipeline's own final prediction, the one stage 11 refines. |
| 11 | **Object-consistent replacement (final refinement)** | Refines stage 10's prediction as described above: this is the method's namesake output. |

Each stage above corresponds to one function (or a short, clearly-named
composition of a few) under `src/ocmask/stages/`; see that directory's
module docstrings, and `docs/rewrite_plan.md`, for the exact original
research script each was extracted from.

### External SAM3 dependency

SAM3 is used by stages 2, 4, 8, 9, and 10. It was obtained outside any
public, scriptable install path, so `scripts/bootstrap_models.sh` cannot
vendor it the way it vendors MASt3R and DINOv2. You need your own SAM3
checkout and checkpoint, then export:

```bash
export SAM3_SOURCE=/path/to/your/sam3/checkout
export SAM3_IMAGE_CHECKPOINT=/path/to/sam3.pt
```

`ocmask.config.load_config` expands `${VAR}`-style placeholders in every
config, so these two variables are all `run_pair`/`demo.py`/
`ocmask evaluate changesim --full-pipeline` need for SAM3. (`configs/pipeline.yaml`
also has a `${SAM31_CHECKPOINT}` placeholder for SAM3.1: the original
research pipeline generated SAM3.1 tracks at stage 2 as well, but nothing
downstream of stage 2 ever reads them -- every later stage re-tracks stage
2's *proposals* with SAM2 instead -- so `run_pair` does not load SAM3.1 at
all and this variable can be left unset.)

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
- SAM3: see "External SAM3 dependency" above

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
official layout; `data/changesim/manifest-table3.jsonl` (8,212 pairs) and
`data/changesim/manifest-new15.jsonl` (15 pairs, a small held-out check) are
included in this repository and reference that layout with relative paths.
To evaluate on a different ChangeSim warehouse:

```bash
python scripts/build_changesim_manifest.py data/changesim/Warehouse_6
```

### Evaluating on ChangeSim

`ocmask evaluate changesim --full-pipeline` (see "Quickstart" above) is
deliberately thin: it runs `run_pair` over every manifest pair, scores each
prediction against the provided ground truth with the same metrics as the
`Result` table above, and writes one `report.json` plus a resumable
`progress.jsonl` (delete it to force a clean re-run). It does not do the
things the original per-stage research scripts under `scripts/` did for
their own ablation study -- per-variant bookkeeping, cross-stage
`selection.json` validation, or a SHA-256 prediction-freeze ledger -- since
there is only one composition to run now, not several to compare
side-by-side.

`--fraction` selects a deterministic, class-stratified subset (same
algorithm as the original research evaluation) instead of the full
manifest, useful for a quick check before committing to a multi-hour run.

(`ocmask evaluate changesim` *without* `--full-pipeline` also exists, but
evaluates a different, narrower thing: the stage-1-only reconstruction
baseline `--ablation` selects between, not this method. It predates
`run_pair` and is kept for that comparison, not as this method's evaluation
path.)

## Repository layout

```
demo.py                 run the method on one image pair (see Quickstart)
src/ocmask/inference.py run_pair(): the single-process, 11-stage orchestrator
src/ocmask/             core library shared by every stage: stage 1's
                         reconstruction pipeline, adapters (MASt3R/SAM2/
                         DINOv2), metrics, config loading, the CLI
src/ocmask/stages/      stages 2-11's algorithmic core (proposals, tracking,
                         identity/feature comparison, the association
                         resolver, and slot_inconsistency.py -- the final
                         refinement)
configs/pipeline.yaml   the one merged config run_pair reads, namespaced
                         per stage
scripts/                original per-stage research scripts (being phased
                         out; see docs/rewrite_plan.md), plus setup/dataset
                         utilities (bootstrap_models.sh, verify_checkpoints.py,
                         build_changesim_manifest.py) that are still current
data/changesim/          ChangeSim manifests (dataset images not included)
checkpoints/             model weights go here (not committed) + manifest.json
docs/method.md           stage 11's full decision procedure + worked
                         failure-case examples
docs/rewrite_plan.md     status of the port from the original multi-script
                         research harness to this single pipeline
```

## Licenses

The code in this repository is MIT-licensed (see `LICENSE`). It orchestrates
third-party models, each under its own license: MASt3R is CC BY-NC-SA 4.0
(non-commercial research use), SAM2 and DINOv2 are Apache-2.0, and SAM3
carries whatever license you obtained it under. ChangeSim has its own
dataset license. Check all of these before redistributing a configured
environment or its outputs.
