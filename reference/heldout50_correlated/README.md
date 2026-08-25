# Correlated heldout50 regression oracle

This records the later 50-pair check used only to choose the conservative
deployment default. The pair IDs were disjoint from fixed10/new15, but frames
came from the same ChangeSim sequences used during development, so this is a
regression oracle—not an unbiased test set.

The preserved Goldilocs directory is:

`/home/tessa/goldilocs/outputs/experiment-slot-inconsistency-replacement-heldout50`

Its frozen files hash to:

| file | SHA-256 |
|---|---|
| `report.json` | `eca19251c903b1b955f22612e0a7aa80e45f8f7330fae1fff58d8df69c944ff3` |
| `effective_config.json` | `0dbbe3e9f2c76d0a10403fc6f811dfd65cb9ec36dc7b86876dc8f53f9b75f347` |
| `predictions_frozen.json` | `457ff5b406e8da015919b85979fb9855e1b078acfe63a446740e07c168814c0f` |

`predictions_frozen.json` states `ground_truth_used: false` and contains the
50 pair IDs plus baseline, guarded, and full prediction hashes.

| profile | binary mIoU (%) | multiclass mIoU (%) |
|---|---:|---:|
| stage-10 baseline | 65.9610 | 35.1251 |
| guarded candidate | 66.0324 | 35.7465 |
| full-mask candidate | 66.0336 | 35.7258 |

The guarded/full difference is tiny; guarded is preferred as a risk policy
because it expands fewer pixels, not as a claim of statistically established
superiority. The production code and 8,212-pair manifest must remain frozen
before any final target masks are scored.
