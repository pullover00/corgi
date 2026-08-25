# Audit: the `changesim-weekend-final` pre-computed cache

This documents a direct, read-only audit of
`/home/tessa/goldilocs/outputs/changesim-weekend-final/` (an
`a3_overnight.py` job run over the full 8,212-pair ChangeSim manifest),
performed before wiring any cache-reuse code into `run_pair`. Every claim
below was checked against the real files on disk -- JSON/YAML parsing and
`np.load()` header inspection only, no model, no GPU, no execution of any
`ocmask` pipeline code. `src/ocmask/weekend_cache.py` is the code that
consumes this cache at runtime; its own validation logic is unit-tested
against small synthetic fixtures in `tests/test_weekend_cache.py` (see that
file's docstring for why real-cache testing and validation-logic testing
are different things, and why both matter here).

A later hardening pass additionally ran the model-free Pillow/NumPy semantic
input check described below against every cache candidate. It still did not
load a model or use a GPU.

## What the cache is

`job.json` + `shards/shard-0000..0032/{manifest.jsonl, configs/*.yaml,
outputs/<stage>/...}`, 33 shards of up to 250 pairs each, covering the same
`manifest-table3.jsonl` this repository ships (verified: both files hash to
`aad8735c1d97d17d7cfb09f11361f931f787723f75b224a1d34a8fb9f28082f6`, and
`job.json`'s own `source_manifest_sha256` matches too). Eight stages were
planned (`a_baseline_cache` .. `h_conservative_a3`, corresponding to this
repository's stages 1, 2, 3, 4, ~6, ~7, ~8-tracking, ~8-A3 respectively),
but the job stopped after stage `d_identity` (stage 4): **stages
`e_moved`/`f_replacement_parent`/`g_feature_tracks`/`h_conservative_a3`
(this pipeline's stages 6-8) have zero data in any of the 33 shards** --
confirmed by listing every shard's `outputs/` directory, not sampling.
Stages 5 (DINOv2) and 9-11 (sentinel, association resolver,
object-consistent replacement) were never part of this job at all -- it
predates them.

## Stage 1 (`a_baseline_cache`): completeness, per shard

`progress.jsonl` entries deduplicated by pair `id` (pairs were retried
after transient `CUDAOutOfMemoryError` failures, so a naive count of
`"status":"success"` lines overcounts -- shard-0000's raw line count is
1140 for 250 pairs), then each surviving `artifacts` path checked with
`Path.is_dir()`. A meaningful fraction of `artifacts` paths point outside
the shard entirely, into `/home/tessa/goldilocs/outputs/benchmark-iou-area-gate-batch1000/pairs/<hash>/`
(imported via `a3_overnight.py`'s `--reuse-baseline` mechanism -- artifacts
are referenced, not copied); those were confirmed to exist and were counted
as valid.

| shard | verified | shard | verified | shard | verified |
|---|---:|---|---:|---|---:|
| 0000 | 250/250 | 0011 | 33/250 | 0022 | 35/250 |
| 0001 | 250/250 | 0012 | 35/250 | 0023 | 20/250 |
| 0002 | 250/250 | 0013 | 25/250 | 0024 | 33/250 |
| 0003 | 250/250 | 0014 | 26/250 | 0025 | 32/250 |
| 0004 | 195/250 | 0015 | 35/250 | 0026 | 36/250 |
| 0005 | 29/250  | 0016 | 28/250 | 0027 | 26/250 |
| 0006 | 28/250  | 0017 | 32/250 | 0028 | 29/250 |
| 0007 | 30/250  | 0018 | 29/250 | 0029 | 25/250 |
| 0008 | 26/250  | 0019 | 39/250 | 0030 | 27/250 |
| 0009 | 27/250  | 0020 | 30/250 | 0031 | 30/250 |
| 0010 | 32/250  | 0021 | 34/250 | 0032 | 21/212 |

**Total: 2,027 / 8,212 pairs (24.7%) have a verified-on-disk,
algorithm-config-matching, semantic-input-matching, successful stage-1
artifact.** Shards 0-3 are fully complete (the job's
early, contiguous progress before something -- most likely sustained GPU
memory pressure, consistent with the logged `CUDAOutOfMemoryError`
failures -- dropped per-shard success rates to roughly 8-16% for the
remaining 29 shards).

## Stages 2-4 (`b_sam3_sam31`, `c_sam3_sam2`, `d_identity`): completeness

Present (any files at all) in **shards 0000-0003 only** -- confirmed by
listing `outputs/` for all 33 shards, not sampling. All three stages are
**fully complete (250/250, zero failures) in all four of those shards** --
1,000 pairs total, verified via each stage's own completion record
(`b`: `proposal_cache_complete.json`'s `pair_count`; `c`/`d`:
`report.json`'s `protocol.pairs_succeeded`/`failures`).

## Correction to an earlier summary

A first-pass audit (done by the orchestrating agent before this one, and
handed to this session as a starting point) reported shard-0000's stage 1
as "221/250" and said only shard-0000 had stages b/c/d. Both undercounted:

- **221/250 only counted `pairs/` subdirectories physically present inside
  the shard**, missing the 29 pairs whose `progress.jsonl` `artifacts`
  field points at the external `benchmark-iou-area-gate-batch1000`
  directory instead. All 29 of those external directories were confirmed
  to exist; shard-0000's real total is 250/250.
- **Stages b/c/d are present (and fully complete) in shards 0000-0003**,
  not shard-0000 alone -- 1,000 pairs, not ~250.

This module's own `ChangesimWeekendCache`, run against the real cache
directory (read-only lookup, no model/GPU code), reproduces these corrected
numbers exactly after strict semantic input validation: scanning all 8,212
manifest pairs gives 2,027 stage-1 hits and 1,000 hits each for stages 2, 3,
and 4. The remaining stop reasons are exactly 6,185 unavailable stage-1
progress records and 1,027 unavailable stage-2 artifacts; there are no
semantic mismatches or validation failures among the 2,027 stage-1 hits.

## Config-compatibility: verified directly, not assumed

For each of stages 1-4, a real cached artifact's *own recorded
configuration* was diffed against the live `configs/pipeline.yaml`,
programmatically (`ocmask.config.load_config`, not hand comparison):

- **Stage 1**: a real pair's `config.json` (from
  `a_baseline_cache/pairs/<hash>/config.json`) against
  `pipeline.yaml`'s `reconstruction:` section. Every algorithmic setting and
  checkpoint path matches. The cached file additionally carries top-level
  `schema_version: 1`; conversely, it predates the live config's provenance
  fields `mast3r.{checkpoint_sha256,source,source_commit}` and
  `sam2.{checkpoint_sha256,source_commit}`. Lookup tolerates only those
  specifically named fields when absent from a legacy artifact. If an
  artifact records one, its value must match. All other reconstruction keys
  remain exact. The current run manifest independently fingerprints current
  checkpoint files and source trees. Because the legacy artifact did not
  record those identities, this cannot retroactively prove its historical
  weight/source bytes; omit `--cache-dir` if that stronger provenance standard
  is required.
- **Stage 2** (`configs/b.yaml`): `proposals:` block and
  `sam3_image_checkpoint_sha256` match `sam3_proposals.proposals`/
  `sam3_proposals.sam3_image_checkpoint_sha256` exactly.
- **Stage 3** (`configs/c.yaml`): `proposals:` block matches; its
  `baseline_protocol.tracking:` block matches every key it shares with
  `reconstruction.tracking` (the live section has two additional keys,
  `minimum_track_score`/`no3d_movement_iou`, that `run_cached_pair` doesn't
  read).
- **Stage 4** (`configs/d.yaml`): `sam3:`/`matching:`/`classification:`
  blocks match `sam3_features.sam3`/`.matching`/`.classification` exactly
  (checkpoint sha256 included).
- Checked shard-to-shard homogeneity too: `configs/{b,c,d}.yaml` diffed
  across shards 0000 vs 0001/0002/0003 -- the only differences are
  per-shard path/bookkeeping fields (`recommended_output`,
  `parent_evaluation`, `experiment_id`, `proposal_cache_parent`,
  `manifest`), never an algorithmic parameter. One shard's config check
  result is therefore valid for all four.

`SAM3_IMAGE_CHECKPOINT`'s sha256 was independently re-verified against the
actual file on disk this session (`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`)
and matches both the cache's and the live config's pinned value.

## Stage-1 semantic input binding

The legacy stage-1 `inputs.json` records paths, not content hashes. A path
match alone is therefore insufficient when a dataset image can change in
place. `ChangesimWeekendCache.lookup()` now reconstructs the exact model input
without importing torch, MASt3R, or DUSt3R: pipeline RGB conversion and resize
to 640x480; DUSt3R long-edge resize to 512 with its pinned Pillow interpolation
and patch-16 center crop; then the float32 `ImgNorm -> SparseGA RGB -> *255`
round trip and uint8 truncation. It requires those arrays to equal
`reconstruction.npz`'s `image0` and `image1` byte-for-byte before offering any
cached stage.

This was checked directly on cached pair `Warehouse_6_Seq_0_2`: both 384x512
RGB arrays match with zero differing bytes. A full 8,212-pair lookup scan found
the same for all 2,027 available stage-1 artifacts. Unit tests also cover the
two important boundaries: changing decoded pixels at the same path rejects the
cache with `stage1_semantic_input_mismatch`, while rewriting a PNG so its raw
file bytes differ but its preprocessed pixels are identical remains safe.
Unreadable images or malformed reconstruction archives stop reuse with
`stage1_semantic_input_validation_failed` and fall back to fresh inference.

## Format-compatibility: verified against real cached bytes

- Stage 2's `proposal_cache/{source,target}.npz` keys
  (`masks_packed, height, width, predicted_iou, stability_score, points,
  crop_boxes`) match `sam3_pairwise.save_proposal_cache`/
  `load_proposal_cache`'s format exactly -- `load_proposal_cache()` (an
  existing, pure-NumPy, already-tested function) can read a real cached
  file with zero modification.
- Stage 4's `sam3_features.npz` (`source`/`target`, shape
  `(256, 72, 72)`, dtype `float16`) matches `Sam3FeatureExtractor.feature_map`'s
  output contract exactly.
- Stage 4's `decisions.json` (`matches`, `diagnostics`) matches the historical
  `classify_identity_location` output shape. It is validation/audit metadata,
  not a live inference input: the pipeline always re-pools descriptors and
  reruns the current CPU calibration/classification code from the cached dense
  maps. CPU replay on real cached pairs 0, 70, and 220 reproduced both the
  historical calibration and match records exactly.

## One real incompatibility found, and how it's handled

The cached `c_sam3_sam2` per-pair `diagnostics.json` does **not** have
`source_changed_proposal_ids`/`target_changed_proposal_ids` -- those fields
were added to `run_cached_pair` after this cache was generated (see
`docs/rewrite_plan.md`). Reading them naively from the cache would have
raised `KeyError` the first time cache-based stage-3 reuse actually ran.
Fixed by deriving the same information from `tracking_attempts.json`
instead (present in every stage-3 pair, cached or fresh, since
`run_cached_pair` always writes it): a proposal counts as changed exactly
when `post_consistency_gate_accepted` is false for it in the
`source_to_clean`/`target_to_clean` stage -- see
`weekend_cache.changed_proposal_ids_from_tracking_attempts`.

## Design: what `run_pair` actually does with this

`src/ocmask/weekend_cache.ChangesimWeekendCache` is strictly opt-in
(`ocmask evaluate changesim --full-pipeline --cache-dir <path>`; omitted by
default). For each pair, `lookup(pair_id, image0, image1)` first binds both
resolved input paths to the cache manifest and stage-1 `inputs.json`, then
binds their exact current post-preprocessing bytes to the two RGB arrays in
the cached reconstruction, and finally returns whichever *prefix* of stages
1-4 is valid -- stage 2 is only ever
offered if stage 1 also hit, stage 3 only if stage 2 also hit, and so on. This
"unbroken chain" rule exists
because stage 2's cached proposals were generated from stage 1's *exact*
cached reconstruction bytes; reusing stage 2+ against a freshly (and
possibly floating-point-non-identically) recomputed stage 1 for the same
pair risks a silent proposal ordering/count mismatch with no crash to
signal it. Stage 4 reuse additionally cross-checks the cached
`decisions.json`'s recorded `source_gate_changed_count`/
`target_gate_changed_count`/`source_valid_descriptor_count`/
`target_valid_descriptor_count` against what the live "visible" proposal
lists actually produce; any mismatch falls back to full computation for
that pair. When those checks pass, only `sam3_features.npz` skips model work:
calibration and match records are recomputed with the live implementation.
Every other failure mode (missing file, pair present in that stage's
`failures.json`, config mismatch) also falls back silently to full
computation for that stage -- never a crash, never a stale result used.

Stages 5-11 are never affected: nothing in the cache covers them, so
`run_pair` always computes them fresh regardless of `--cache-dir`.

## End-to-end validation status

The cache path has now been exercised by the real GPU pipeline on
`Warehouse_6_Seq_0_2`, not only by format inspection:

- `out/repro-cache-smoke-v3` consumed stages 1-3 and recomputed stage 4. Its
  stage-4 lookup stopped on a location-only checkpoint-path mismatch between a
  Hugging Face snapshot symlink and the same pinned blob. This exposed and
  fixed a portability bug: compatibility now ignores filesystem location
  fields while still requiring the exact source commit, checkpoint hash, and
  algorithm settings.
- `out/repro-cache-smoke-v4` consumed all four stages. Its pair manifest records
  `lookup_stop_reason: all_available`; every stage is both offered and used.
  Stage 4 records `dense_maps_reused: true`,
  `decision_policy: current_cpu_recomputed`, and historical audit equality for
  both calibration and match records.
- The v3 and v4 base, guarded, and full masks are byte-identical. Their pooled
  confusion matrices and every report metric are identical too: 77.894934%
  binary and 42.853917% multiclass mIoU. Only seven diagnostic SAM cosine
  floats differ, all by at most `1.1921e-7`; no decision differs.
- Stage 4 fell from 18.263 seconds to 0.870 seconds and total prediction time
  from 139.416 seconds to 117.025 seconds. Autocast is false at every recorded
  boundary, and worker cleanup restores the initial numerical state.
- Repeating the exact v4 command validates the immutable pair manifest and
  skips the inference worker; the frozen prediction hash and report remain
  unchanged.

This validates the live stage-4 cache loader against a live recomputation over
the same cached stages 1-3. It does not retroactively establish the missing
generation-time MASt3R/SAM2 source and checkpoint identities for legacy stages
1-3, nor has a final-source, same-pair, entirely cache-free control completed.
For a final benchmark requiring complete provenance, omit `--cache-dir`. For a
speed-oriented run that accepts the legacy provenance boundary, first compare
a small cached selection with a cache-free selection on the target machine,
then freeze the choice before any final targets are scored.
