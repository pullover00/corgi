# Reproduction and production-readiness report

This report separates three questions that were previously conflated:

1. Can the remembered 25-pair Goldilocs result be recovered exactly?
2. Why did the generated `object-consistent-masks` evaluation regress?
3. What is safe to carry into an 8,212-pair ChangeSim evaluation without
   treating the development pairs as a test set?

## Result recovery

The exact historical stage-11 source, runner, configuration, tests, output
hashes, and replay report are frozen under `reference/headline_68_51/`.
Replaying that recovered snapshot against the preserved Goldilocs artifact
graph reproduced:

| metric | IoU (%) |
|---|---:|
| changed | 42.5781 |
| unchanged | 94.4507 |
| binary mIoU | **68.5144** |
| added | 38.6125 |
| removed | 31.5177 |
| moved | 17.5544 |
| replaced | 20.4514 |
| multiclass mIoU | **40.5173** |

The originally remembered replaced value, 20.55%, is not present in the raw
aggregate; 20.4514% is the recovered value. The headline raster was
`overall.full_mask_candidate`. The historical runner's default guarded raster
scores 68.3943% binary and 40.2953% multiclass, so selecting the default file
does not reproduce the headline.

This exact replay is an historical oracle, not proof that a newly generated
raw pipeline is byte-identical. It depends on the preserved Goldilocs cache
graph, whose upstream generation-time dependencies were not all recorded.

## Regression localization

The stored failed export makes the loss location explicit:

| 25-pair raster | binary mIoU (%) | multiclass mIoU (%) |
|---|---:|---:|
| historical frozen stage-10 resolver | 68.1098 | 38.9969 |
| historical headline stage 11, full | 68.5144 | 40.5173 |
| failed export tracking raster | 58.5231 | 28.9698 |
| failed export stage-10 resolver | 60.1990 | 29.6782 |
| failed export stage 11, full | 60.5306 | 30.8727 |

Stage 11 still improved the failed export. Most of the loss was already
present in reconstruction/proposal/tracking inputs, so copying stage-11
thresholds alone could not fix it. For example, pair
`Warehouse_9_Seq_0_944` has matching reconstruction/render hashes but a
different SAM3 proposal inventory: historical source/target counts 92/247,
fresh export counts 93/243.

Direct source/runtime inspection found these operational and reproducibility
faults:

- The old evaluator ran all pairs in one Python process. Importing the
  MASt3R/CroCo stack changes torch's matmul state from TF32 disabled and
  `highest` precision to TF32 enabled and `high` precision.
- SAM3's image predictor manually entered a CUDA BF16 autocast context and did
  not close it. Later pairs therefore ran MASt3R/SAM2 under inherited ambient
  BF16. Compiled models and predictors also retained GPU allocations across
  pairs; the preserved logs eventually show 14.79-14.95 GiB allocated at
  out-of-memory failures.
- Resume trusted only a pair ID. One output directory could silently combine
  artifacts made by different source/config/runtime revisions.
- Evaluation silently resized a prediction when its shape differed from the
  target, hiding an invalid artifact boundary.
- The generated environment declared torch 2.4.1/torchvision 0.19.1 although
  the measured Goldilocs runtime was torch 2.5.1/torchvision 0.20.1 with CUDA
  12.4.

They are not a complete causal explanation for the score gap. Pair
`Warehouse_9_Seq_0_944` was the first successful pair in the failed run, so it
could not inherit BF16 from an earlier SAM3 invocation. A later clean worker
still produced the failed export's full-mask hash
`17c13c1e1db2a72d0c38f52009a9f80da1c4cb98e8d4f5a34f0e1c2660803eb6`, not
the historical frozen hash
`0b027789732ed5ce0d38754d2fd38bfbc1d8018713f30202e54696444015cf1e`.
Thus process isolation fixes order dependence, leakage, and OOM, but does not
by itself recreate the historical SAM3 proposal inventory. The remaining
first-pair upstream numerical divergence requires a fresh GPU investigation
and must not be attributed to stage 11.

## Production repair

The production evaluator now enforces the following invariants:

- one seeded, clean subprocess per image pair;
- explicit pre-MASt3R and post-reconstruction numerical states;
- scoped SAM3 TF32/BF16 contexts, explicit predictor/model release, and
  numerical-state telemetry at model boundaries;
- separate SAM2 tracker lifetimes for stages 3, 6-8, and 10;
- a two-phase target firewall: all predictions must be frozen and hashed
  before the parent process opens any ground-truth mask;
- a run fingerprint covering implementation, expanded and on-disk config,
  manifest and selected IDs, input bytes, checkpoints, external model source
  trees, installed distributions, numerical policy, runtime, and optional
  cache assets;
- atomic per-pair manifests, input/output hash verification, a full-lifetime
  output-directory lock, an orphan-safe per-pair worker lock, durable attempt
  logs, timeouts, retries, and strict resume validation;
- strict target dtype/range/label/shape validation, with no resizing; and
- pooled class metrics plus per-sequence mean, standard deviation, and worst
  sequence, with both guarded and full predictions scored from the same frozen
  GPU pass.

The recorded checkpoint hashes are:

| model | SHA-256 |
|---|---|
| MASt3R | `e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2` |
| SAM2.1 Hiera Large | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| DINOv2 ViT-B/14 reg4 | `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| SAM3 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

The core version lock matches the measured development runtime, but the old
environment had been upgraded in place (`--no-deps`) and now exposes duplicate
Triton and PyYAML distribution metadata. A normal clean solve will not recreate
that inconsistent state. The evaluator records and rechecks the entire
installer-owned distribution inventory for each run, so artifacts from two
different inventories cannot be silently resumed together; it does not claim
that the current inventory is the unknown golden inventory used to generate
the historical headline caches.

## Completed validation

- The recovered historical archive passes its checksum manifest and reproduces
  the 68.5144/40.5173 report exactly.
- All 115 CPU tests pass. They cover numerical-state restoration, target
  firewall behavior, immutable resume/fingerprints, timeout and retry
  handling, strict ChangeSim parsing/scoring, cache validation, and the frozen
  headline archive.
- An uncached end-to-end repaired-pipeline smoke on
  `Warehouse_9_Seq_0_70` completed in a clean worker and scored 71.2342%
  binary / 42.8638% multiclass. Its reconstruction, geometry, and renders
  were byte-identical to the corresponding Goldilocs artifacts; stage-3 and
  final pixel agreement were 99.8125% and 99.3786%, respectively. This run
  preceded the final stage-4 cache-location compatibility patch.
- A final-source cached end-to-end smoke on `Warehouse_6_Seq_0_2` used all of
  stages 1-4 from the audited cache and computed stages 5-11 fresh. It scored
  77.8949% binary / 42.8539% multiclass. Its guarded and full predictions are
  byte-identical (`2e8df253...`), and its base prediction is `eaf1e17f...`.
  Stage-4 dense maps were reused, while current CPU calibration and match
  classification were recomputed and exactly matched the historical records.
  This validates execution/cache semantics, not headline equivalence: the
  current full hash `2e8df253...` differs from this pair's historical frozen
  full hash `3488693f...`, and the current base hash `eaf1e17f...` differs from
  historical `b741b1a2...`.
- Repeating that exact evaluator command skipped inference via the frozen pair
  manifest and reproduced the same report. Autocast was disabled at every
  recorded model boundary and cleanup restored the worker's initial numerical
  state.
- The 8,212-entry manifest has unique pair IDs and 24,636 present RGB/target
  paths. The observed pixel distribution is about 92.297% unchanged, 1.995%
  added, 2.677% removed, 1.556% moved, and 1.474% replaced.

The final-source cache smoke is in `out/repro-cache-smoke-v4/`. The exact
historical report is in `reference/headline_68_51/replay/report.json`.

## Validation still required

A cache-free run of the repaired final source over all 25 development pairs
has not completed in this checkout. Do not claim that the new raw pipeline has
reproduced 68.51/40.52 until that run finishes. Use a new output directory:

```bash
./scripts/run_eval_resilient.sh \
    data/changesim/manifest-mixed25.jsonl \
    out/repro-mixed25-guarded-clean-v1 \
    --prediction-variant guarded
```

The legacy cache is optional and its current image/config/artifact bytes are
strictly bound into the run fingerprint. It did not record the historical
MASt3R/SAM2 source and weight identities, however. Omit `--cache-dir` for the
scientifically clean final evaluation if that missing generation provenance is
not acceptable.

## Generalization protocol

The fixed10 and new15 sets were inspected during development. The later
heldout50 uses disjoint pair IDs but correlated frames from the same sequences.
None is an unbiased test set, and the 68.51/40.52 result must not be used as an
estimate of performance on unseen sequences. Mixed25 already touches every one
of the eight warehouse/sequence combinations present in the 8,212-pair
manifest, so that manifest contains no untouched whole sequence. Its eventual
aggregate is necessarily descriptive rather than sequence-disjoint.

For the 8,212-pair result, freeze the guarded configuration without further
threshold tuning, freeze predictions for the complete manifest before opening
any targets, and report per-class and per-sequence results. For future model
selection, split by whole sequence (or temporally contiguous blocks with guard
bands), never by interleaved frames. The guarded profile is a conservative risk
choice based on correlated regression data; it is not claimed to be
statistically superior.

A genuinely sequence-disjoint estimate now requires additional ChangeSim
sequences outside the current manifest (or another dataset). Without those,
freeze the current method before the full run, inspect its targets only after
all predictions are frozen, report the overlap limitation, and do not tune from
the resulting 8,212-pair report.

The serial clean-worker evaluator is intentionally conservative. At roughly
258 seconds per uncached pair on the development RTX 4090 Laptop GPU, 8,212
pairs are about 24.5 GPU-days. Do not score independent shards for a final
leakage-sensitive benchmark: this repository does not yet include a validated
global freeze-only multi-GPU coordinator.
