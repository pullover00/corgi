# Object-Consistent Masks

Scene change detection for RGB-D image pairs: given two photos of the same
place taken at different times, produce a per-pixel map of what changed,
classified as `added`, `removed`, `moved`, or `replaced`.

This repository is the **object-consistent full target masks** method: a
complete 11-stage pipeline implementation, extracted from a larger research
repository (internally: `GOLDILOCS`) and pruned down to the current deployable
descendant of the recovered method. All 11 stages are part of the method and part of
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

This is the repaired `change_detect` working copy. The source
`object-consistent-masks` directory was copied, not modified in place.

## Quickstart: run it on your own two images

```bash
python demo.py --before before.png --after after.png --output out/
```

`--before` is the earlier-time photo, `--after` the later-time photo of
(approximately) the same place from the same viewpoint -- the output change
map is aligned to `--after`. This needs a working GPU environment; see
"Setup" below before your first run, and `ocmask doctor` to check readiness.
`out/labels.png` is the current full-mask per-pixel prediction
(`out/overlay.png` is the same thing rendered over the "after" photo for a quick look); see
`demo.py --help` for the full list of files it writes.

To smoke-test the dataset evaluator from this checkout (the small fraction is
for plumbing validation, not for reporting a score):

```bash
PYTHONPATH=src python -m ocmask.cli evaluate changesim \
    --manifest data/changesim/manifest-mixed25.jsonl \
    --output out/smoke-guarded-v1 --fraction 0.04 \
    --full-pipeline --prediction-variant guarded
```

The evaluator launches the same `ocmask.inference.run_pair` implementation in
a fresh worker process for each image pair. This isolation is part of the
reproducibility fix, not an optional performance mode. See "Evaluating on
ChangeSim" for the full 8,212-pair command and resume rules.

If this repository is not installed editable in the active environment, keep
`PYTHONPATH=src` as shown. This matters on the original development machine,
where an older editable install may still point at the untouched
`object-consistent-masks` directory.

## Recovered result and deployment default

The exact Goldilocs code, configuration, runner, tests, prediction hashes, and
replayed report that produced the remembered 25-pair result were recovered in
`reference/headline_68_51/`. Replaying that snapshot against the preserved
Goldilocs cache graph reproduced this aggregate exactly:

| profile | changed IoU | unchanged IoU | binary mIoU | added IoU | removed IoU | moved IoU | replaced IoU | multiclass mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical stage 11, `full_mask_candidate` | 42.5781 | 94.4507 | **68.5144** | 38.6125 | 31.5177 | 17.5544 | 20.4514 | **40.5173** |

The remembered 20.55% `replaced` value was a rounding/transcription error; the
recovered raw aggregate is 20.4514%. The headline used
`overall.full_mask_candidate`, not the historical runner's default guarded
file. That output-selection mismatch was one reason a nominal replay could
report the wrong row. The reference directory is a provenance and
compatibility oracle; its retained Goldilocs paths deliberately are not wired
into the production package.

Production evaluation defaults to `--prediction-variant guarded`. A later
50-pair regression check (disjoint pair IDs, but still temporally correlated
within the same sequences) scored guarded at 66.0324% binary / 35.7465%
multiclass mIoU and full at 66.0336% / 35.7258%. The essentially identical
binary score, marginally better multiclass score, and smaller replacement
footprint make guarded the lower-risk ChangeSim-scale choice. Use
`--prediction-variant full` only when the historical full footprint must also
be the report's top-level/default row. Every completed evaluation scores both
already-frozen variants without a second GPU pass; the selected variant remains
part of the run fingerprint so existing top-level report semantics are
unambiguous. `reference/heldout50_correlated/README.md` records the frozen
artifact hashes and the correlation caveat.

For a single pair, `run_pair` and `demo.py` still write full-mask
`labels.png`, conservative `labels_guarded.png`, and stage-10
`labels_base.png`. The benchmark CLI scores the explicitly selected variant;
its default is guarded. These are current production outputs; only the frozen
archive is entitled to the exact historical “headline” label until a clean
25-pair production run establishes a new result.

`docs/method.md` has the full decision procedure for stage 11 (object
plausibility, companion expansion, depth-authoritative ownership,
source-footprint cleanup, dataset-wide target-object arbitration, floor
suppression) with worked failure-case examples, carried over from the
original research report. Verify the recovered archive with
`(cd reference/headline_68_51 && sha256sum -c SHA256SUMS)`;
`reference/headline_68_51/README.md` documents the replay and its limits.
`docs/reproduction_report.md` records the regression localization, repaired
invariants, completed validation, and the remaining GPU validation boundary.

## Why the generated repository regressed

The previous evaluator processed every pair in one long-lived Python process.
MASt3R/CroCo deliberately enables CUDA matmul TF32 when imported; that
false/highest to true/high transition is now an explicit post-reconstruction
boundary on both fresh and cached paths. One damaging cross-pair bug was SAM3's
image predictor entering a BF16 autocast context without closing it: later
MASt3R and SAM2 work silently inherited ambient BF16. Model and compiled-graph
lifetimes also retained GPU allocations across pairs and eventually caused an
out-of-memory failure.

A second problem made the apparent result internally inconsistent: legacy
resume logic trusted only a pair ID. The same output directory accumulated
predictions made by different source/config revisions, and ground-truth masks
with a mismatched shape were silently resized during scoring.

Those operational defects are proven, but they do not explain the entire score
gap. The first successfully processed pair had no preceding SAM3 context to
leak and already differed from its historical proposal cache. For
`Warehouse_9_Seq_0_944`, the historical source/target proposal counts are
92/247 while both the failed export and a later clean worker produce 93/243;
the clean worker's final SHA-256 is identical to the failed export and differs
from the historical frozen hash. The unresolved first-pair divergence is in
fresh upstream proposal generation, before stage 11. Exact 68.51/40.52
reproduction is therefore established only for the frozen historical artifact
graph; the repaired raw pipeline must be evaluated as a new frozen run.

The repaired evaluator now:

- starts every pair in a clean Python subprocess with a declared seed and
  numerical policy;
- scopes and restores CUDA TF32/autocast state and explicitly closes SAM3's
  predictor context;
- hashes code, expanded and on-disk config, benchmark manifest, selected IDs,
  prediction variant, cache metadata, model source trees, checkpoints, runtime,
  and numerical policy into `run_manifest.json`;
- hashes each input image and each frozen output in a per-pair
  `pair_manifest.json`, refusing mixed or stale artifacts; and
- rejects target/prediction shape mismatches instead of resizing labels.

GPU kernels and proposal tie-breaking can still produce small differences
across genuinely different hardware/software stacks. The new checks prevent a
silent stack change from being called the same run; they do not promise
byte-identical masks on an unvalidated stack.

SAM3 remains an external dependency you must supply yourself. See "External
SAM3 dependency" below.

## Pipeline architecture

The method is all 11 stages together, run by `run_pair` end to end from one raw
RGB image pair to the final object-consistent raster; none of stages 1-10 is a
third-party baseline. Dataset evaluation invokes that one-pair entrypoint in a
new process for every pair.

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
python scripts/verify_checkpoints.py            # verify the pinned hashes
ocmask doctor                                    # reports CUDA/model/checkpoint readiness
```

The recovered cache/headline runtime is Python 3.11, PyTorch 2.5.1,
torchvision 0.20.1, NumPy 1.26.4, and PyTorch CUDA 12.4. Earlier repository
files incorrectly declared PyTorch 2.4.1/torchvision 0.19.1 even though the
Goldilocs environment had been upgraded before the headline predictions were
created. `environment.yml`, `pyproject.toml`, and `configs/pipeline.yaml` now
agree on the measured versions. Full evaluation fails before inference when
the required runtime versions do not match and records the remaining runtime,
GPU, CUDA/cuDNN, source-tree, and checkpoint identities in the run manifest.
This is a validated core-version lock, not a byte-for-byte reconstruction of
the historical conda environment: that environment was upgraded in place and
contains conflicting duplicate package metadata (notably Triton and PyYAML).
Every new run fingerprints the entire installer-owned distribution inventory
and refuses to resume across an inventory change, but no trustworthy golden
inventory from headline generation exists.

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

```bash
PYTHONPATH=src python -m ocmask.cli evaluate changesim \
    --manifest data/changesim/manifest-table3.jsonl \
    --output out/changesim-table3-guarded-v1 \
    --full-pipeline --prediction-variant guarded --pair-retries 1
```

This is the primary 8,212-pair command. The checked manifest contains 8,212
unique pairs and 24,636 existing image/target files in the current dataset
layout. `src/ocmask/changesim.py` uses ChangeSim's official palette and reports
precision, recall, F1, and IoU for `unchanged`, `added`, `removed`, `moved`,
and `replaced`, plus binary changed/unchanged metrics. `report.json` contains
the complete pooled metrics, Table-3-style percentages, per-pair confusion
matrices, per-sequence metrics, and the mean, standard deviation, and worst
sequence mIoU.

The full evaluator deliberately has two phases:

1. It starts one clean worker per pair. The worker receives the two RGB inputs
   but no target path, computes all 11 stages, and atomically records hashes of
   its full, guarded, and base predictions.
2. Only after every selected prediction is frozen does the parent process
   open any ground-truth mask. It scores both guarded and full rasters, while
   retaining the requested variant in the report's backward-compatible
   top-level fields. Both pair and report records state that ground truth was
   not used in inference.

If even one pair fails, no target is opened and no partial report is produced;
successful freezes remain available for a clean resume. This makes accidental
test-label feedback mechanically harder and ensures an evaluation interruption
cannot leave a half-written prediction looking valid.
`progress.jsonl` records `prediction_frozen` (or `failure`), while the
per-pair manifest and its hashes are the resume authority.

### Resume and immutable output directories

Rerun exactly the same command with the same output directory to resume. A
valid frozen pair is skipped; a failed or unfinished pair gets a clean worker.
Use `--continue-on-error` to attempt the remainder after a persistently failing
pair. `scripts/run_eval_resilient.sh` is a convenience wrapper that invokes
the evaluator once with clean-process retries and continue-on-error behavior:

```bash
./scripts/run_eval_resilient.sh \
    data/changesim/manifest-table3.jsonl \
    out/changesim-table3-guarded-v1 \
    --prediction-variant guarded
```

The parent holds a nonblocking lock for the full output directory. Each worker
also holds a per-pair lock before importing torch or a model. If the parent is
hard-killed while a worker survives, an immediate resume waits for that worker,
revalidates any atomic freeze it completed, and does not start a second writer
for the same pair.

Do not delete or transplant `progress.jsonl`, and do not reuse an older output
directory after changing source, configuration, manifest, selection, runtime,
checkpoint, external source tree, cache, or prediction variant. Those inputs
are fingerprinted in `run_manifest.json`; a mismatch or legacy unfingerprinted
directory is rejected without deleting anything. Choose a new, descriptive
output name such as `...-guarded-v2` instead. Input and prediction content
hashes additionally prevent an image changed in place from being resumed.

`--fraction` selects a deterministic class-stratified subset. It is useful for
an operational smoke test, but repeatedly examining such a subset makes it
development data, not an unbiased test.

To regression-test the repaired evaluator on the historical 25 development
pairs without using the legacy cache, use a new output directory:

```bash
./scripts/run_eval_resilient.sh \
    data/changesim/manifest-mixed25.jsonl \
    out/repro-mixed25-guarded-clean-v1 \
    --prediction-variant guarded
```

This run is a compatibility/development check, not an unbiased estimate of
generalization. Do not reuse `out/eval-mixed25` or any interrupted pre-repair
directory: those artifacts predate immutable fingerprints and clean workers.

### Optional validated cache and runtime expectation

`--cache-dir <path>` can reuse a config-matching `a3_overnight.py`-style cache
for stages that are complete and valid, falling back to fresh computation for
the rest. On the original development machine, the audited partial cache is:

```bash
./scripts/run_eval_resilient.sh \
    data/changesim/manifest-table3.jsonl \
    out/changesim-table3-guarded-cache-v1 \
    --prediction-variant guarded \
    --cache-dir /home/tessa/goldilocs/outputs/changesim-weekend-final
```

That cache contains validated stage-1 artifacts for 2,027/8,212 pairs and
stages 2-4 for the first 1,000; all other work still runs normally. Cache
metadata is part of the run fingerprint. Legacy stage-1 reuse also requires
the exact current Pillow/NumPy MASt3R-preprocessed RGB arrays to match the two
images stored in its reconstruction; a same-path image changed in place is a
cache miss. See `docs/cache_audit.md` for the exact coverage, validation rules,
and the legacy cache's provenance limitation. The old cache did not record the
generation-time MASt3R/SAM2 source and weight hashes, so omit `--cache-dir` for
the scientifically clean final benchmark unless that missing provenance is
acceptable. Also omit it on another machine or when the audit does not match.

An observed uncached end-to-end pair on the development GPU took about 258
seconds. At that rate, 8,212 serial pairs are roughly 24.5 GPU-days, before
retries; hardware and cache hits can change this substantially. One retained
smoke pair used about 14.6 MB and 41 files, projecting to roughly 120 GB and
337,000 files, so reserve at least 150 GB and sufficient inodes before the
full run. The isolated workers fix state contamination and unbounded cross-
pair growth, but do not make the model stack cheap.

Do not independently run and score shards for the final leakage-sensitive
benchmark: each evaluator invocation opens its shard's targets after freezing
that shard, before other shards are necessarily complete. This repository does
not yet provide a globally validated freeze-only coordinator/aggregator. Use
the single command above for the final 8,212-pair run; multi-GPU sharding is
appropriate only after such a coordinator is added and tested.

### Evaluation protocol: avoid another overfit result

The original fixed10 and new15 pairs were both inspected during method
development. They reproduce history but are not a held-out test. The later
heldout50 check used different pair IDs but frames from the same sequences, so
temporal correlation also prevents treating it as a final generalization
estimate. The mixed25 development set touches all eight warehouse/sequence
combinations in the 8,212-pair manifest. There is therefore no untouched whole
sequence left inside this manifest, and its eventual score must be described as
a full-dataset descriptive result, not a sequence-disjoint test result.

For a defensible ChangeSim result:

- choose settings using sequence-disjoint development data only;
- reserve whole sequences where possible, or contiguous temporal blocks with
  guard bands, rather than interleaved frames from the same video;
- freeze code, configuration, model/source/checkpoint hashes, numerical
  policy, and the split before opening final-test ground truth;
- do not tune thresholds after reading the 8,212-pair report; and
- report pooled class IoUs together with per-sequence dispersion and the worst
sequence, not only one aggregate mIoU.

For the current data, sequence-disjoint model selection requires acquiring or
designating additional sequences outside this 8,212-pair manifest. If that is
not possible, freeze the present method now, score the full manifest once, and
state the overlap limitation explicitly; do not call it an unbiased held-out
estimate.

The complete manifest is highly imbalanced: approximately 92.297% unchanged,
1.995% added, 2.677% removed, 1.556% moved, and 1.474% replaced pixels. Binary
mIoU alone can therefore hide weak rare-class behavior. The guarded profile is
the production default because it reduces aggressive footprint expansion
without selecting a new threshold on the 8,212-pair labels.

(`ocmask evaluate changesim` *without* `--full-pipeline` also exists, but
evaluates a different, narrower thing: the stage-1-only reconstruction
baseline `--ablation` selects between, not this method. It predates
`run_pair` and is kept for that comparison, not as this method's evaluation
path.)

## Splitting ChangeSim across two machines

The 8,212-pair table3 protocol is split by warehouse into two independent,
non-overlapping halves so it can run on two GPUs in parallel:

- `scripts/run_warehouse67_experiment.sh` -- Warehouse_6 + Warehouse_7, 2,122
  pairs (`data/changesim/manifest-warehouse67.jsonl`).
- `scripts/run_warehouse89_experiment.sh` -- Warehouse_8 + Warehouse_9, 6,090
  pairs (`data/changesim/manifest-warehouse89.jsonl`).

Each writes to its own stable output directory (`out/warehouse67_experiment`,
`out/warehouse89_experiment`) so the two runs never collide, and each is
independently resumable per "Resume and immutable output directories" above.
There is no coordinator that merges the two into one `report.json`; score
each half's `out/.../report.json` separately, or write pooled per-class
confusion-matrix totals from both once both are frozen.

### Setting up a second machine

Steps to get `run_warehouse89_experiment.sh` (or any other manifest) running
on a second GPU workstation, starting from nothing:

1. **Push this repository to GitHub yourself** (not done here) and clone it
   on the second machine. `.gitignore` already excludes `data/`, checkpoint
   weights, and the vendored `src/mast3r`/`src/dinov2` checkouts, so nothing
   in this step touches the ChangeSim dataset or model weights.

2. **Conda environment** -- `environment.yml` names the env `ocmask`, but
   every `run_*_experiment.sh` script hardcodes `conda run -n goldilocs`
   (this machine's actual env name, predating a later rename). Either name
   matches works as long as they agree; simplest is to override the name at
   creation time so the scripts work unmodified:

   ```bash
   conda env create -f environment.yml -n goldilocs
   conda activate goldilocs
   ./scripts/bootstrap_models.sh   # vendors MASt3R + DINOv2, installs SAM2
   pip install -e .
   pytest tests/                   # sanity check; no GPU/checkpoints needed yet
   ```

3. **MASt3R / SAM2 / DINOv2 checkpoints** -- download the three checkpoints
   listed under "Setup" above into `checkpoints/`, then:

   ```bash
   python scripts/verify_checkpoints.py
   ```

4. **SAM3** -- on this machine it came from two places, both gated/manual
   rather than scriptable (see "External SAM3 dependency" above); replicate
   both on the second machine with the same Hugging Face account:

   ```bash
   git clone https://github.com/facebookresearch/sam3.git /path/to/sam3
   # Request access at https://huggingface.co/facebook/sam3, then, once
   # approved and `huggingface-cli login` has run on the second machine:
   huggingface-cli download facebook/sam3 sam3.pt --local-dir /path/to/sam3-ckpt

   export SAM3_SOURCE=/path/to/sam3
   export SAM3_IMAGE_CHECKPOINT=/path/to/sam3-ckpt/sam3.pt
   ```

   Both `run_warehouse67_experiment.sh` and `run_warehouse89_experiment.sh`
   also check for `SAM31_CHECKPOINT` on disk even though `run_pair` never
   loads it (see "External SAM3 dependency"). If you don't have a real
   SAM3.1 checkpoint on the second machine, point it at any file that
   exists -- `SAM3_IMAGE_CHECKPOINT` itself works -- purely to satisfy that
   existence check:

   ```bash
   export SAM31_CHECKPOINT="$SAM3_IMAGE_CHECKPOINT"
   ```

5. **ChangeSim data** -- only `Warehouse_8/` and `Warehouse_9/` are needed
   for `run_warehouse89_experiment.sh` (~11 GB + ~15 GB). On this machine
   they live at `/home/tessa/goldilocs/data/changesim/Warehouse_{8,9}`
   (`data/changesim/Warehouse_8` and `Warehouse_9` in this repo are symlinks
   to that path). Copy them to the second machine with `rsync` over SSH,
   resumable if interrupted:

   ```bash
   rsync -avP --partial \
       tessa@<this-machine-hostname>:/home/tessa/goldilocs/data/changesim/Warehouse_8 \
       tessa@<this-machine-hostname>:/home/tessa/goldilocs/data/changesim/Warehouse_9 \
       /path/to/changesim/on/second/machine/
   ```

   Then, in the cloned repo on the second machine, either move the data
   under `data/changesim/Warehouse_8` and `Warehouse_9` directly or symlink
   them there the same way this machine does. Reserve at least 30 GB for
   the copy and roughly 90 GB more for `--save-stage-artifacts` output
   (proportional to the 120 GB/8,212-pair estimate under "Optional
   validated cache and runtime expectation" above, scaled to 6,090 pairs).

6. **Rebuild the Warehouse_8/9 manifest** -- not committed (excluded by the
   same `data/changesim/*` gitignore rule as every manifest except
   `manifest-table3.jsonl`/`manifest-new15.jsonl`), but cheap to regenerate
   from the tracked `manifest-table3.jsonl`:

   ```bash
   python3 -c "
   import json
   with open('data/changesim/manifest-table3.jsonl') as f, \
        open('data/changesim/manifest-warehouse89.jsonl', 'w') as out:
       for line in f:
           row = json.loads(line)
           if row['id'].startswith('Warehouse_8_') or row['id'].startswith('Warehouse_9_'):
               out.write(line)
   "
   ```

7. **Verify and run**:

   ```bash
   ocmask doctor
   ./scripts/run_warehouse89_experiment.sh
   ```

   At this machine's observed ~258 s/pair (see "Optional validated cache and
   runtime expectation" above), 6,090 serial pairs is roughly 18 GPU-days
   before retries; an A6000's actual throughput may differ. Ctrl-C and
   rerunning the same command resumes safely, per "Resume and immutable
   output directories" above.

## Repository layout

```
demo.py                 run the method on one image pair (see Quickstart)
src/ocmask/inference.py run_pair(): the one-pair, 11-stage orchestrator
src/ocmask/full_pair_worker.py
                        clean-process dataset worker; never receives a target
src/ocmask/numerics.py  scoped CUDA/autocast state and deterministic pair setup
src/ocmask/reproducibility.py
                        runtime/source/model fingerprints and immutable resume
src/ocmask/             core library shared by every stage: stage 1's
                         reconstruction pipeline, adapters (MASt3R/SAM2/
                         DINOv2), metrics, config loading, the CLI
src/ocmask/stages/      stages 2-11's algorithmic core (proposals, tracking,
                         identity/feature comparison, the association
                         resolver, and slot_inconsistency.py -- the final
                         refinement)
configs/pipeline.yaml   the one merged config run_pair reads, namespaced
                         per stage
scripts/                setup/dataset utilities plus run_eval_resilient.sh;
                         historical per-stage research scripts are retained
                         for audit but are not the production evaluator
data/changesim/          ChangeSim manifests (dataset images not included)
checkpoints/             model weights go here (not committed) + manifest.json
reference/headline_68_51/
                        exact recovered headline source/config/report/hashes
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
