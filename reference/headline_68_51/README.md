# Recovered 68.51 / 40.52 headline snapshot

This directory is an immutable reference bundle for the exact Goldilocs
experiment that produced the reported 25-pair result. It is not imported by
the production pipeline.

## Verified result

The recovered code was replayed on 2026-08-21 against the preserved Goldilocs
caches. The replay reproduced the original aggregate exactly:

| metric | IoU (%) |
| --- | ---: |
| changed | 42.5781 |
| unchanged | 94.4507 |
| binary mIoU | 68.5144 |
| added | 38.6125 |
| removed | 31.5177 |
| moved | 17.5544 |
| replaced | 20.4514 |
| multiclass mIoU | 40.5173 |

The remembered REPLACED value of 20.55% does not match the original raw
aggregate; the machine-readable report is authoritative at
20.451374185173102%.

The headline is `overall.full_mask_candidate`. The default `labels.png` in the
historical runner is the guarded candidate, which scores 68.3943% binary and
40.2953% multiclass. Using that default is a subtle reproduction error.

## Contents

- `src/goldilocs/experiments/slot_inconsistency.py`: exact inference module.
- `scripts/run_slot_inconsistency_replacement_experiment.py`: exact runner.
- `configs/experiments/changesim-branch-b2-consolidated-reciprocal-densegrid96.yaml`:
  exact configuration and cache routing.
- `tests/test_slot_inconsistency.py`: tests present at the headline cutoff.
- `replay/report.json`: complete report regenerated from the recovered code.
- `replay/predictions_frozen.json`: pair IDs plus baseline, guarded, and full
  prediction SHA-256 values.
- `COMPATIBILITY.md`: the behavior a compatibility profile must preserve.
- `SHA256SUMS`: byte hashes for this archive and the original replay JSON.

The runner and YAML retain historical Goldilocs imports, output paths, and
cache roots deliberately. They are evidence, not a directly portable command.
This bundle is therefore a recovered replay oracle that depends on the
preserved Goldilocs checkout/cache graph; it is not a self-contained archive of
every imported helper, model, RGB/target image, and upstream cache tensor.

## Provenance

The headline implementation was never committed. The Goldilocs repository has
only the initial `1a0c450` baseline commit; the relevant source, runner, config,
tests, and outputs were untracked or dirty and were edited again after the
headline run. This snapshot was reconstructed from successful `apply_patch`
records in:

`/home/tessa/.codex/sessions/2026/08/17/rollout-2026-08-17T18-40-58-01a00f18-7f2d-7da2-9713-2cdbf6d0334a.jsonl`

The reconstruction cutoff is immediately after the last successful source edit
at `2026-08-18T09:33:18.722Z` and immediately before the headline evaluation.
Replaying it against the preserved cache graph regenerated every reported
metric exactly. `predictions_frozen.json` records `ground_truth_used: false`;
the runner opens ground truth only after all predictions and hashes are frozen.
The archived test snapshot also passes in the reconstructed runtime: 24 tests
passed.

The archived JSON files have one final newline added by `apply_patch`. Their
parsed content is identical to the generated files. `SHA256SUMS` records both
the raw generated hashes and the newline-normalized archive hashes.

## Scope and overfitting warning

This result is a reproduction oracle, not an unbiased generalization estimate.
Both fixed10 and at least part of new15 were inspected during iterative method
development. The `tune_on_new15: false` field describes the final replay
protocol; it cannot undo earlier adaptive tuning. Pair
`Warehouse_9_Seq_0_944` contributes an unusually large improvement (multiclass
47.00% to 61.17%, REPLACED 13.47% to 58.17%), so the 25-pair gain is
concentrated.

For ChangeSIM-scale use, first make compatibility mode match these frozen
prediction hashes. Then freeze code, configuration, checkpoints, dependency
versions, seeds, and upstream cache fingerprints. Select any deployable profile
using sequence-disjoint development data and score a separately frozen,
sequence-disjoint test only once. Prefer the guarded profile when a small score
loss buys lower expansion risk, and always report per-class and per-sequence
metrics in addition to pooled pixel mIoU.
