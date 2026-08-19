# Rewrite plan: from benchmark harness to a single inference pipeline

Status and exact next steps for turning this repo from "11 separate
ChangeSim-benchmark experiment scripts" into "one `ocmask infer --before
--after --output` command that runs the winning method on any image pair."
Written so this can be picked up in a fresh session without re-deriving the
research already done here.

## Why this exists

The repo originally reproduced the full research/benchmarking apparatus:
per-stage manifest/split bookkeeping, `selection.json`/`report.json`
cross-validation between stages, SHA-256 prediction-freeze-before-ground-
truth ledgers, and *exposed* ablation variants (R0-R4, A0-A4, O0-O3,
replacement_only/moved_verification/combined_guarded_hybrid) at every
decision point. None of that belongs in a repo meant to run the winning
method on new data and be linked from a paper. This plan replaces it with
one pipeline that always computes only the winning composition.

## Environment note (read before running anything)

Real GPU + real model weights **are** available on this machine, split
across conda environments:

- `conda activate goldilocs` -- torch 2.5.1+cuda, MASt3R, SAM2, all working.
  Checkpoints at `/home/tessa/goldilocs/checkpoints/`. This is what was used
  to validate stage 1 for real (see below).
- `conda activate sam3` -- has SAM3 source but is currently broken:
  `import sam3` fails on a missing `pycocotools` dependency, and this
  sandbox has no network access to `pip install` it. Whoever continues this
  needs to fix that env (or use a different one) before SAM3-dependent
  stages (2, 3's proposal reuse, 8, 9, 10) can be executed.
- DINOv2/MASt3R source: symlink `src/mast3r` -> `/home/tessa/goldilocs/src/mast3r`
  and `src/dinov2` -> `/home/tessa/goldilocs/src/dinov2` (both are gitignored;
  recreate the symlinks locally, don't commit them).
- Checkpoints: symlink `checkpoints/*.pth`/`*.pt` -> the goldilocs ones the
  same way, or run `scripts/bootstrap_models.sh` + download fresh.
- ChangeSim data for a real test pair: `/home/tessa/goldilocs/data/changesim/`.

**Stage 1 has been validated for real** (`ocmask infer`, unmodified except
for the config default fix already committed): ran MASt3R + SAM2 end to end
on `Warehouse_7/Seq_1/259` with real GPU compute, produced a sane
`labels.png` (2.5% changed pixels, correct value range). This is strong
evidence the already-ported base-pipeline code (`pipeline.py`, `adapters/`,
`cli.py`, `config.py`) is correct, not just import-clean.

## What's done

1. `configs/pipeline.yaml` -- all 11 stages' winning-path settings merged
   into one file, namespaced per stage, with every manifest/split/variant
   bookkeeping key dropped. Values copied verbatim from the validated
   `*-fixed10-densegrid96.yaml` / A3 / R4 configs.
2. `stages/real_image_association_resolver.py` gained `resolve_r4()` +
   `ResolverR4Settings`: a complete, faithful single-pair extraction of
   stage 10 (see "Stage 10" below for exactly what it needs as input).
3. `stages/sam3_pairwise.py`'s `run_cached_pair()` (stage 3) now also
   returns `source_changed_proposal_ids`/`target_changed_proposal_ids` in
   its diagnostics dict, needed by stage 4 (see below) without re-running
   SAM2 tracking a second time.

## What's left, stage by stage

For each stage: what it needs, what already exists as a clean reusable
function, and what still needs to be written.

### Stage 1 -- reconstruction + SAM2 baseline. DONE, validated for real.

Call exactly as today: `ocmask.pipeline.PairwisePipeline(config["reconstruction"], Mast3rAdapter(...), Sam2Adapter(...)).run(image0_path, image1_path, output_dir)`.
Gives `reconstruction.npz`, `render_0_to_1.png` (=`inputs.source_render`),
the real target image, and `config.json`. No extraction needed.

### Stage 2 -- SAM3 proposals + SAM3.1 tracking. Simple, not yet extracted.

Needs a `Sam3AutomaticMaskGenerator(checkpoint, points_per_side=96, ...)`
(`stages/sam3_proposals.py`) run over `source_render` and `target_image` via
`.generate(image) -> list[Sam3Proposal]`, then `proposals_to_objects(...)`
(`stages/sam3_pairwise.py`) to get `ObjectMask` lists. SAM3.1 tracking uses
`Sam31MaskTracker` (`stages/sam31_backend.py`) the same way `Sam2MaskTracker`
is used elsewhere. This is small enough to write directly in `inference.py`
rather than extracting a wrapper -- no script-side entanglement to unwind.

### Stage 3 -- SAM2 re-tracking of stage 2's proposals. Mostly done.

`run_cached_pair()` in `stages/sam3_pairwise.py` (now returning the changed-
proposal-ID lists too) already does this per-pair, cleanly. Call it directly:
`run_cached_pair(artifact_dir, output_dir, tracker, source_proposals, target_proposals)`.
This *is* the "parent" raster (labels.png) that stage 7/8 build on.

### Stage 4 -- SAM3 dense features + per-pair identity-threshold calibration.

**Not yet extracted; this is the next thing to do.** Needs, per
`scripts/run_sam3_identity_location_experiment.py` lines ~433-539:

1. `Sam3FeatureExtractor(source_path, checkpoint)` (`stages/sam3_identity_location.py`)
   `.feature_map(source_render)` and `.feature_map(target_image)` -> two
   dense arrays.
2. `mask_descriptors(feature_map, objects, minimum_feature_cells=...)` on
   each side -> `FeatureDescriptorBatch`.
3. `spatial_iou = pairwise_mask_iou(source_objects, target_objects)`.
4. `source_static`/`target_static` boolean arrays: `True` where a proposal's
   `automatic_proposal_id` is **not** in stage 3's
   `source_changed_proposal_ids`/`target_changed_proposal_ids` (this is what
   the new diagnostics field from this session's commit is for -- build the
   boolean array by membership test, don't recompute tracking).
5. `calibrate_identity_threshold(similarity, spatial_iou, source_static, target_static, source_descriptors.valid, target_descriptors.valid, fallback_threshold=..., control_iou=..., negative_iou=..., maximum_negative_acceptance=..., minimum_positive_acceptance=..., minimum_positive_count=..., minimum_negative_count=...)`
   (`stages/sam3_identity_location.py`) -> `SimilarityCalibration`, whose
   `.threshold` is exactly the `identity_threshold` stage 10's `resolve_r4()`
   needs.

`classify_identity_location`/`compose_identity_labels` (also in that module)
are **not** needed downstream -- they build stage 4's own standalone label
raster, which nothing else in the winning path consumes. Skip them; this
stage's only outputs that matter downstream are `(source_map, target_map,
calibration.threshold)`.

Write this as a new function in `stages/sam3_identity_location.py`, e.g.
`compute_features_and_calibration(...) -> tuple[np.ndarray, np.ndarray, SimilarityCalibration]`.

### Stage 5 -- DINOv2 dense features. Simple, not yet extracted.

Same shape as stage 4's feature extraction but via `adapters/dinov2.py`'s
`Dinov2FeatureExtractor`/`DinoV2AppearanceAdapter` instead of SAM3. Per the
README's "Two SAM3 grid densities" note, the original research config feeds
this from the 64-point-grid proposal lineage rather than the 96 one used
everywhere else; confirm whether that distinction still matters once
everything runs in one process (it may not -- DINOv2 features are a
function of raw pixels, not proposal masks, per that same README note) and
document the decision either way. Nothing in the winning path is known to
consume stage 5's output except stage 11 (`dino_feature_cache_root`), so
verify against `stages/slot_inconsistency.py`'s actual usage before writing
this.

### Stage 6 -- moved-association raw tracking cache.

Only the raw forward/backward SAM2 propagation is needed (for proposals
stage 3 marked changed), not `stages/sam3_moved_association.py`'s
`associate_moved_objects`/variant composition -- confirmed unused by the
winning path in an earlier session. Small enough to write directly in
`inference.py`: call `Sam2MaskTracker.track()` forward and backward on the
changed-candidate masks (same ones identified via stage 3's
`*_changed_proposal_ids`) and pass the raw mask lists straight to stage 11's
`match_object_slots`/`consolidate_hypotheses` -- no need to replicate the
original `tracks.npz` disk-cache format at all, since everything now runs in
one process.

### Stage 7 -- guarded hybrid (replacement_only variant).

Not yet extracted. `stages/sam3_guarded_hybrid.py` has `compose_guarded_variant`
already as a clean function -- read `scripts/run_sam3_guarded_hybrid_experiment.py`
to see exactly which evidence sets get passed in for the `replacement_only`
variant specifically (the script computes 3 variants; only one is needed).

### Stage 8 -- feature-veto gate (A3 composition).

Not yet extracted. Needs, per `scripts/run_sam3_feature_veto_gate_experiment.py`
around lines 586-1435:

1. `pair_and_classify_gate_features(...)` (`stages/sam3_feature_veto.py`) ->
   a `decision` with `.guarded_source_ids`/`.guarded_target_ids` (and
   `.hard_*_ids`, not needed for A3).
2. A **second, separate** SAM2 tracking pass (own `Sam2MaskTracker` call) on
   exactly the proposals `decision` flags, forward and backward -- this is
   real computation, not cache bookkeeping; read `_tracking_cache`-equivalent
   logic around line 1340 to see which proposal IDs get tracked.
3. `ordinary_promoted_objects(...)` + `merge_objects_with_parent(...)` (both
   in `stages/sam3_feature_veto.py`) using `decision.guarded_source_ids`/
   `decision.guarded_target_ids` and stage 3's baseline raster ->
   `guarded_labels`.
4. `direct_replacement_mask(decision.pairs, ...)` + `apply_direct_semantics(guarded_labels, replacement, [])`
   (both in `stages/sam3_feature_veto.py`) -> the A3 raster. This is what
   stage 10's `resolve_r4()` calls `parent_labels`.

A1/A2/A4 and `hard_labels`/`moved_masks`/`full_labels` are dead for the
winning path -- don't compute them.

### Stage 9 -- obvious-object sentinel (real-I0 generation only).

Mostly trivial once you see through the disk-caching wrapper. What's
actually needed:

```python
generator = Sam3AutomaticMaskGenerator(checkpoint, points_per_side=96, ...)
proposals, feature = generator.generate_with_feature_map(image0)  # REAL, un-warped source image
source_objects = proposals_to_objects(proposals)
source_map = feature
generator.release()
```

That's it -- `paths["parent_labels"]` in the original script is *not* a
computed sentinel output, it's a direct file reference to stage 8's A3
raster (confirmed by reading `scripts/run_obvious_object_sentinel_experiment.py`
lines 329-360: `parent_labels = roots["conservative"] / parent_record[...]`).
None of `compose_sentinel`/`o1`/`o2`/`o3_verified_absence` in that script is
needed for the winning path -- stage 10 uses `compose_sentinel` itself
directly (already wired into `resolve_r4()`), not a precomputed sentinel
raster.

### Stage 10 -- R4 resolver. DONE.

`resolve_r4()` in `stages/real_image_association_resolver.py`. Call with:
`reconstruction` (stage 1), `source_objects`/`source_map` (stage 9, real
I0), `target_objects`/`target_map` (stage 2's proposals / stage 4's target
features -- both real I1), `parent_labels` (stage 8's A3 raster),
`identity_threshold` (stage 4's `calibration.threshold`), a live tracker,
and `image0`/`image1` (real, un-warped RGB arrays).

### Stage 11 -- object-consistent replacement. DONE, needs wiring only.

Everything in `stages/slot_inconsistency.py` and `stages/branch_b2.py` is
already clean and unit-tested. Feed it: stage 10's output as `baseline`,
stage 2's proposals consolidated via `stages/branch_b2.consolidate_hypotheses`
(sources from stage 6's forward tracks, targets from stage 2's target
proposals), stage 4's `source_map`/`target_map` and stage 5's DINOv2 maps,
and stage 1's reconstruction for `ground_contact`/floor suppression. Read
`scripts/run_slot_inconsistency_replacement_experiment.py`'s `_inference_pair`
(already ported, unmodified) for the exact call sequence -- it's already
single-pair-shaped, just currently driven by a multi-pair loop reading from
disk instead of the in-memory objects the new pipeline will have on hand.

## After all 11 stages are wired

1. Write `src/ocmask/inference.py`: one `run_pair(image0, image1, output_dir,
   config) -> np.ndarray` calling the above in sequence, in one process.
2. Add `ocmask infer` (or extend the existing one) to call it, plus a batch/
   directory mode.
3. Add a thin, optional `ocmask evaluate` for ChangeSim-format scoring: run
   `run_pair` over a manifest, compare to provided labels, print metrics.
   No variant bookkeeping, no prediction-freeze ledger.
4. Delete: all 10 `scripts/run_*_experiment.py` files (superseded by
   `inference.py`), `configs/stages/*.yaml` (superseded by
   `configs/pipeline.yaml`), the `outputs/s01..s11` naming convention and
   `scripts/run_pipeline.py` orchestrator, `data/changesim/manifest-*.jsonl`
   unless the thin eval mode still wants them, `tests/test_slot_inconsistency.py`'s
   import of `scripts.run_slot_inconsistency_replacement_experiment` (rewire
   to import from `inference.py`/`stages/` instead once that script is gone).
5. Rewrite README.md: usage becomes `ocmask infer --before A.png --after
   B.png --output out/`, no more manifest/split language as the primary
   interface (eval mode can still mention it).
6. Verify: reuse this session's real stage-1 validation approach (the
   `goldilocs` conda env has working MASt3R/SAM2/DINOv2) to smoke-test as
   much as possible; SAM3-dependent stages need the `sam3` conda env fixed
   first (see "Environment note").
