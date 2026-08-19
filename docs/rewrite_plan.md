# Rewrite plan: from benchmark harness to a single inference pipeline

Status of turning this repo from "11 separate ChangeSim-benchmark experiment
scripts" into one `run_pair(image0, image1, output_dir, config)` call (and
the `demo.py`/`ocmask evaluate changesim --full-pipeline` entrypoints built
on it) that runs the winning method on any image pair. Written so this can
be picked up in a fresh session without re-deriving the research already
done here.

## Status: the pipeline is wired end to end

`src/ocmask/inference.py`'s `run_pair` now calls all 11 stages in one
process and returns the final prediction. `demo.py` (single pair) and
`ocmask evaluate changesim --full-pipeline` (ChangeSim manifests) both call
it. See "What's validated" below for exactly how much of this has been
confirmed by real execution versus careful reading only.

## Naming note

Earlier drafts of this pipeline (and the original research code it was
extracted from) used internal ablation-tracking shorthand throughout --
`R4`/`r4_no_geometry_ablation` for stage 10's composition, `A0`-`A4` for
stage 8's ablation ladder, `O0`-`O3` for stage 9's, `replacement_only`/
`moved_verification`/`combined_guarded_hybrid` for stage 7's three variants,
`fixed10`/`densegrid96` for specific evaluation-split/grid-density
configurations. None of that is meaningful to a reader who never saw the
ablation study, so it has been renamed throughout the public-facing
surface (function names, config keys, CLI, docs) to plain descriptive
terms -- e.g. `resolve_r4` -> `resolve_real_image_associations`,
`ResolverR4Settings` -> `AssociationResolverSettings`. A couple of these
internal literal strings still exist purely as *arguments* one already-
existing function (`sam3_guarded_hybrid.compose_guarded_variant`) accepts,
never surfaced in any public name; see that module for why. Where a
docstring cites the original ablation label for provenance/traceability
back to the research (so a reader who *does* have the original scripts can
find the exact corresponding code), that is a one-line footnote, not the
primary name anything is known by.

## Environment note (read before running anything)

Real GPU + real model weights are available on this machine, split across
conda environments -- but by the end of this session, both paths below
actually work, which was not true at the start:

- `conda activate goldilocs` -- torch 2.5.1+cuda, MASt3R, SAM2, DINOv2, **and
  a working `sam3` import** (an editable install pointing at `/home/tessa/sam3`)
  all in one environment. This is what `run_pair` was smoke-tested with --
  it is the only environment that has every model this pipeline needs at
  once, so there was no need to span two environments in one run.
- `conda activate sam3` -- previously broken (`import sam3` failed on a
  missing `pycocotools`, and the sandbox appeared to have no network). Both
  turned out to be transient: network access was available this session,
  and `pip install pycocotools psutil scipy` in the `sam3` env fixed the
  import. Not needed given `goldilocs` already works, but noted in case a
  future session needs a second, isolated SAM3-only process.
- MASt3R/DINOv2/checkpoint symlinks: `src/mast3r`, `src/dinov2`, and
  `checkpoints/*.pth`/`*.pt` all point at the `goldilocs` repo's copies
  (gitignored, recreated this session, do not commit them). ChangeSim
  warehouse directories under `data/changesim/` are symlinked the same way.
- SAM3 environment variables used this session:
  `SAM3_SOURCE=/home/tessa/sam3`,
  `SAM3_IMAGE_CHECKPOINT=/home/tessa/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt`
  (sha256 verified to match `configs/pipeline.yaml`'s pinned
  `sam3_image_checkpoint_sha256` exactly). `SAM31_CHECKPOINT` was located
  (`/home/tessa/gaussian-grouping/sam3.1/sam3.1_multiplex.pt`, sha256 also
  verified against the config) but is **not used by `run_pair`** -- see
  "SAM3.1 is dead for the winning path" below.

## What's validated, and how

Real-execution-validated this session, on a real ChangeSim pair
(`Warehouse_8/Seq_1/763`, using the `SAM3_SOURCE`/`SAM3_IMAGE_CHECKPOINT`
above), running `run_pair` end to end in the `goldilocs` conda environment:
stage 1 (already validated in an earlier session; re-confirmed as part of
every `run_pair` attempt this session), and stages 2-11 in full (SAM3
proposals, SAM2 re-tracking, SAM3 dense features + calibration, DINOv2
dense features, moved-candidate tracking, evidence fusion, the
feature-veto-gated direct-replacement pass, the real-image sentinel, the
association resolver, and the object-consistent replacement refinement
itself) all executed for real with no exceptions, end to end, producing a
`labels.png` whose changed region (a tipped-over barrel disappearing, a
new upright barrel appearing nearby) matches what actually differs between
the two real photos by eye. This is the strongest evidence available that
the full `run_pair` orchestration -- not just each stage in isolation -- is
wired correctly.

Two real bugs were caught and fixed by this real execution (i.e. neither
would have been caught by reading alone):

1. `sam3_guarded_hybrid.moved_verification_evidence` expects its
   `forward_tracks`/`reverse_tracks` dicts to use *membership* to mean
   "accepted" (an entry present in the dict = accepted; a rejected
   candidate is simply absent, never present with value `None`) --
   `resolve_object_consistent_labels`'s `consolidate_hypotheses` call needs
   the opposite shape (a full-length list, `None` for every rejection).
   `inference.py`'s `_accepted_tracks_by_proposal_id` now builds the first
   shape; `forward_track_masks` is built separately, by list comprehension
   straight off the raw tracking attempts, for the second. Symptom before
   the fix: `ValueError: forward track shape differs from proposal grid`
   (a `None` was reaching `np.asarray(None, dtype=bool)`, producing a 0-d
   array).
2. `real_image_association_resolver.py` called `associate_identities`
   without importing it -- a leftover from the earlier session that first
   extracted `resolve_real_image_associations` (then `resolve_r4`) by
   reading alone, without executing it. Fixed by adding the import. This is
   the concrete reason "written carefully by reading, cross-checked against
   the config" is not a substitute for actually running the code once a
   working environment exists.

Everything not mentioned above (which is nearly everything -- these were the
only two defects real execution found across all 11 stages) matched its
careful-reading-based extraction on the first successful run.

## Corrections to this plan found by reading (not in the original draft)

An earlier draft of this plan (before this session) made two claims that
turned out to be wrong once the actual downstream data dependencies were
read carefully. Both are corrected in the code as it stands now; recorded
here so nobody re-introduces the bug by trusting the old claim:

1. **Stage 7's real "parent" baseline is not stage 3's raw raster.** The
   original research config (`configs/experiments/changesim-sam3-feature-veto-gate-*.yaml`'s
   `a0_parent_variant`, defaulted in
   `scripts/run_sam3_feature_veto_gate_experiment.py` to
   `combined_guarded_hybrid` and never overridden in the winning
   `fixed10-densegrid96` config) shows stage 8's real base raster is stage
   7's *combined* composition -- both the same-place replacement evidence
   and the moved-object verification evidence applied together -- not the
   replacement-only evidence an earlier draft of this plan assumed. That
   means stage 6's raw tracks are load-bearing for stage 7 too (moved
   verification needs forward *and* reverse tracks), not just for stage 11
   as originally thought. `sam3_guarded_hybrid.refine_with_motion_and_replacement_evidence`
   computes the combined composition; `run_pair` computes both tracking
   directions in stage 6 accordingly.
2. **Stage 4's `classify_identity_location` is not dead code.** An earlier
   draft of this plan reasoned it only fed stage 4's own standalone label
   raster (true) and concluded nothing downstream needed it. But stage 7's
   `decisions.json` input (`match_records` in the code here) *is*
   `classify_identity_location`'s `match_records` output -- confirmed by
   reading `scripts/run_sam3_identity_location_experiment.py`'s own
   `decisions.json` writer next to `scripts/run_sam3_guarded_hybrid_experiment.py`'s
   reader of the same file. `sam3_identity_location.compute_appearance_features`
   now calls it and returns `match_records` alongside the dense feature
   maps and calibration; only `compose_identity_labels` (the standalone
   raster itself) remains genuinely unused downstream.

## SAM3.1 is dead for the winning path

Every downstream config's `proposal_cache_parent` points at stage 2's own
output directory, but every downstream reader opens only that directory's
`proposal_cache/{source,target}.npz` (the raw SAM3 automatic-mask-generator
cache) -- never a tracking-result file from stage 2's own SAM3.1 pass.
Nothing in the winning composition consumes SAM3.1's tracking output; every
later stage that needs to move a mask between the two frames re-tracks
stage 2's *proposals* with SAM2 instead (stage 3's baseline, stage 6/7/8/10's
own passes). `run_pair` therefore never loads `Sam31MaskTracker` or the
`SAM31_CHECKPOINT` model at all -- a real (if modest -- SAM3.1's checkpoint
is ~3.5GB) engineering simplification versus what an earlier draft of this
plan assumed stage 2 needed to do.

## Stage-by-stage correspondence

For each stage: the function(s) that implement it and the original research
script it was extracted from. All are wired into `run_pair`.

| # | Function(s) | Extracted from |
|---|---|---|
| 1 | `pipeline.PairwisePipeline.run` (pre-existing) | n/a -- already the production pipeline |
| 2 | inline in `run_pair` (`Sam3AutomaticMaskGenerator.generate` + `sam3_pairwise.proposals_to_objects`) | `scripts/run_sam3_pairwise_experiment.py` |
| 3 | `sam3_pairwise.run_cached_pair` (pre-existing) | `scripts/run_sam3_pairwise_experiment.py` |
| 4 | `sam3_identity_location.compute_appearance_features` | `scripts/run_sam3_identity_location_experiment.py` |
| 5 | inline in `run_pair` (`adapters.dinov2.Dinov2FeatureExtractor.feature_map`) | `scripts/run_dinov2_identity_location_experiment.py` |
| 6 | inline in `run_pair` (`Sam2MaskTracker.track`, forward and backward) | `scripts/run_sam3_moved_association_experiment.py`'s cache-building step only (its own association-variant logic is not used, see below) |
| 7 | `sam3_guarded_hybrid.refine_with_motion_and_replacement_evidence` | `scripts/run_sam3_guarded_hybrid_experiment.py` |
| 8 | `sam3_feature_veto.apply_feature_veto_direct_replacement` | `scripts/run_sam3_feature_veto_gate_experiment.py` |
| 9 | inline in `run_pair` (`Sam3AutomaticMaskGenerator.generate_with_feature_map`) | `scripts/run_obvious_object_sentinel_experiment.py` |
| 10 | `real_image_association_resolver.resolve_real_image_associations` | `scripts/run_real_image_association_resolver.py` |
| 11 | `object_consistent_replacement.resolve_object_consistent_labels` | `scripts/run_slot_inconsistency_replacement_experiment.py`'s `_inference_pair` |

Confirmed-dead code, not ported (still present in the original scripts for
provenance, not called by anything in `src/ocmask`):

- Stage 2's SAM3.1 tracking pass (see above).
- Stage 6's own association-variant composition
  (`sam3_moved_association.associate_moved_objects` and siblings) -- only
  the raw forward/backward SAM2 propagation it would have cached is used.
- Stage 7's `moved_verification`-only and `replacement_only`-only variants,
  and stage 8's `a0`/`a1`/`a2`/`a4`-equivalent compositions (hard veto
  without direct replacement, and the moved-reasoning variant) -- only the
  one composition each stage's winning-path successor actually consumes is
  computed.
- Stage 9's `o0`-`o2`-equivalent sentinel compositions and its own
  `targeted_absence_verification` config block -- stage 10 does its own
  live SAM2 absence check via `resolve_real_image_associations`'s `tracker`
  argument, not a precomputed sentinel raster.
- Stage 10's geometry-gated added/removed and parent-replay variants (what
  the original research code labeled `r0`-`r3`) -- only the
  geometry-support-free composition (`r4`) is computed.

## Known remaining gap

None at the "does it run" level: all 11 stages, including stage 11
(`object_consistent_replacement.resolve_object_consistent_labels`), have
completed a real `run_pair` execution without exception (see "What's
validated" above). What has *not* been done:

- Only one real pair has been run this way (`Warehouse_8/Seq_1/763`,
  ChangeSim classes `[2, 3]` -- removed and moved/rotated). A single pair
  cannot rule out an edge case (an empty changed-candidate list, a pair with
  no valid identity-calibration controls, a pair where MASt3R reconstruction
  is poor) that a wider ChangeSim run would exercise. Run
  `ocmask evaluate changesim --full-pipeline` over a larger manifest
  (`data/changesim/manifest-new15.jsonl` is a reasonable first target) as
  the next real-execution milestone, and compare the resulting
  `table3_iou_percent` against this README's `Result` table as a sanity
  check (not an exact match -- different pairs, no frozen-seed guarantee
  across the port).
- Runtime is real but not fast: this one pair took ~230s end to end on an
  RTX 4090 Laptop GPU, dominated by stage 1 (reconstruction, ~90s), stage 2
  (dense SAM3 proposal generation over both images, ~65s), and stage 9 (a
  second SAM3 proposal pass, ~40s) -- each `Sam2Adapter`/`Sam2MaskTracker`
  construction also re-triggers `torch.compile` once (`compile_image_encoder:
  true` in `configs/pipeline.yaml`), which is one-time-per-process, not
  per-pair, but still adds latency to a single-pair `demo.py` run. Nothing
  about this is incorrect, just worth knowing before assuming a ChangeSim
  evaluation over thousands of pairs is a quick check.

## What's left

1. `scripts/run_*_experiment.py` (10 files), `configs/stages/*.yaml`, and
   `scripts/run_pipeline.py` are now fully superseded by `run_pair` +
   `configs/pipeline.yaml` and can be deleted -- not done yet this session
   (kept as a faithful, executable reference for the stage-by-stage
   correspondence table above; deleting them is straightforward once
   nobody needs to cross-check against them anymore).
2. `tests/test_slot_inconsistency.py` still imports
   `scripts.run_slot_inconsistency_replacement_experiment` for a few tests
   of that script's own disk-artifact conventions (including literal
   `r4_no_geometry_ablation` path-selection strings, which are that
   script's actual on-disk format and were deliberately left alone rather
   than renamed out from under it). Once that script is deleted per (1),
   those tests need to either move to test
   `object_consistent_replacement.py` directly or be retired if they were
   only ever testing the script's own bookkeeping.
3. `configs/stages/*.yaml` and the original per-stage `scripts/*.py` still
   contain the internal ablation-code naming described above (`r0`-`r4`,
   `a0`-`a4`, `o0`-`o3`, `densegrid96`, `fixed10`, etc. -- extensively, as
   they are the untouched original research artifacts). They were
   deliberately not renamed this session: they are not part of the public
   API surface (`run_pair`/`demo.py`/`ocmask evaluate changesim
   --full-pipeline` never read them), and are already slated for deletion
   per (1) rather than being a document worth cleaning up in place. If (1)
   is deferred indefinitely for some reason, revisit this.
4. `stages/obvious_change_sentinel.py`'s `evaluate_endpoint_candidates`/
   `EndpointSentinelResult` (local variables/fields named `o1`/`o2`/`o3`)
   are dead code for the winning path -- confirmed by grep, nothing in
   `src/ocmask` calls either name -- and were left as-is for the same
   reason as (3): not on the public surface, not worth the risk of editing
   an otherwise-untouched function with no test coverage of its own for a
   purely cosmetic change. `compose_sentinel` (the function this pipeline
   actually calls) does not use that naming at all.
