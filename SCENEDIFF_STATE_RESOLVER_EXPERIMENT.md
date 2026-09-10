# Controlled SceneDiff state-resolver experiment

## Protocol and implementation

All variants use the same preregistered 10 SceneDiff pairs, one fixed T1 query per pair, the existing multi-frame T0 reconstruction, and the `v6_no_dino` method. DINOv2 remained disabled. Reconstruction, localization, raw/clean rendering, DI2FIX, SAM3 proposals/features, SAM2 tracking, geometry, thresholds, recovery, and final class logic outside the stated resolver branches were held fixed.

Two independent flags were added so the earlier combined resolver could be separated cleanly:

- `v6_no_dino_stateA`: preserve a correspondence that passed identity but failed only `same_location_iou`. Existing geometric identity evidence yields MOVED when geometry is resolvable and rejects same-location; otherwise the association becomes internal UNKNOWN. Both endpoints remain consumed.
- `v6_no_dino_stateB`: stateA plus explicit UNKNOWN bookkeeping for unmatched ADDED/REMOVED candidates that fail the existing visibility/coverage test.

The old `enable_conservative_state_resolver` flag remains backward compatible and enables both branches. The exact configs are `configs/scenediff_v6_no_dino_stateA.yaml` and `configs/scenediff_v6_no_dino_stateB.yaml`; compared with `configs/model_ablation_m2_no_dino.yaml`, their only semantic additions are the corresponding resolver flags.

The original run had loose proposal/descriptor/track dumps but not the dense SAM3 maps required to reproduce unchanged proposal-miss recovery. A one-time, separately named `v6_no_dino_inventory_seed` cache was therefore materialized. It reused all shared reconstruction/localization/render/DI2FIX artifacts; its regenerated baseline labels were pixel-identical on all 10 queries to the stored `v6_no_dino` labels. Both requested variants then loaded those frozen inventories, dense maps, tracks, geometry, and ceiling/sky masks and ran only state resolution and downstream output/evaluation (2.5–9.6 seconds per query). Original baseline outputs were not overwritten.

## Aggregate results

| Variant | Pooled IoU | Mean pair IoU | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v6_no_dino` | **0.3010** | **0.3498** | 0.3681 | **0.6229** | **0.4627** | **515,616** | 885,304 | **312,136** |
| `v6_no_dino_stateA` | 0.2965 | 0.3312 | **0.4148** | 0.5096 | 0.4573 | 421,836 | **595,179** | 405,916 |
| `v6_no_dino_stateB` | 0.2965 | 0.3312 | **0.4148** | 0.5096 | 0.4573 | 421,836 | **595,179** | 405,916 |

StateA removes **290,125 FP pixels** (32.8%) but also loses **93,780 TP pixels** (18.2%). Precision rises by 0.0467, while recall falls by 0.1133; pooled IoU falls by 0.0045, mean pair IoU by 0.0185, and F1 by 0.0054. StateB is pixel-identical to stateA on all 10 queries.

### False-positive pixels by predicted class

| Variant | ADDED FP | REMOVED FP | MOVED FP | REPLACED FP |
|---|---:|---:|---:|---:|
| `v6_no_dino` | 126,123 | 637,904 | 0 | 121,277 |
| `v6_no_dino_stateA` | 61,725 | 305,490 | 112,575 | 115,389 |
| `v6_no_dino_stateB` | 61,725 | 305,490 | 112,575 | 115,389 |

StateA removes 64,398 ADDED FP and 332,414 REMOVED FP, but introduces 112,575 MOVED FP. The remaining net reduction is real, but it is bought with the larger recall regression above.

### Final object-decision counts

| Variant | ADDED | REMOVED | MOVED | REPLACED | UNCHANGED | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| `v6_no_dino` | 42 | 70 | 0 | 124 | 148 | 0 |
| `v6_no_dino_stateA` | 26 | 43 | 3 | 121 | 143 | 22 |
| `v6_no_dino_stateB` | 26 | 43 | 3 | 121 | 143 | 110 |

UNKNOWN in stateB includes the same 22 ambiguous identity/location associations as stateA plus 88 visibility-unsupported unmatched candidates. Those 88 had already been suppressed from the stateA pixel output by the unchanged visibility filter.

## Per-query IoU

| Pair | Baseline | stateA | stateB | stateA − baseline |
|---|---:|---:|---:|---:|
| P01 184214 0030→0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| P01 095114 0001→0011 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| closet_1→closet_2 | 0.4045 | 0.1678 | 0.1678 | **−0.2367** |
| bedroom_28→bedroom_29 | 0.5502 | 0.5502 | 0.5502 | 0.0000 |
| store_57→store_58 | 0.6918 | 0.7223 | 0.7223 | **+0.0305** |
| store_39→store_40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bedroom_32→bedroom_33 | 0.5721 | 0.5924 | 0.5924 | **+0.0204** |
| table_5→table_6 | 0.2009 | 0.2014 | 0.2014 | +0.0005 |
| gym_3→gym_4 | 0.4417 | 0.4417 | 0.4417 | 0.0000 |
| living_room_49→living_room_50 | 0.6364 | 0.6364 | 0.6364 | 0.0000 |

By per-query IoU, stateA is **better on 3, worse on 1, tied on 6**. StateB versus stateA is **better on 0, worse on 0, tied on 10**.

## StateA diagnostics

The baseline records **58 location-mismatch rejection events**: 31 from direct identity and 27 from the clean-render bridge. Because rejected endpoints can be reconsidered by the bridge, these correspond to **34 distinct T0/T1 associations**. Thirty of the 58 recorded events—**16 distinct associations**—ended with both a REMOVED T0 endpoint and an ADDED T1 endpoint in the final baseline decisions.

StateA consumes each distinct association at its first accepted identity:

- **12 become MOVED at the resolver**, of which 9 subsequently fail the unchanged visibility filter and 3 remain as final MOVED predictions.
- **22 become UNKNOWN** because geometry is unavailable or supports the same location.
- Of the 16 distinct associations that produced both ADDED and REMOVED in the baseline, 2 become retained MOVED and 14 become UNKNOWN.

The 3 retained MOVED objects occupy 178,029 pixels: **65,454 TP and 112,575 FP**, for 36.8% MOVED precision. Both kitchen MOVED predictions are entirely false positive (51,848 and 26,928 pixels); the bedroom_32 bag supplies all 65,454 MOVED TP pixels but also 33,799 FP.

Required case inspections:

1. **ADDED+REMOVED → MOVED:** in `bedroom_32_bedroom_33`, T0 object 4 / T1 object 6 passed identity, failed spatial IoU at 0.211, and failed the existing geometric same-place test. Baseline draws separate red/green regions over the bag; stateA draws a single blue MOVED union. It is the one meaningful MOVED result and contributes 65,454 GT-overlapping pixels.
2. **UNKNOWN:** in `store_57_store_58`, T0 object 8 / T1 object 10 produced both baseline endpoints, with spatial IoU 0.441 just below the unchanged 0.45 criterion. Existing geometry strongly supports the same location (`geometric_score=0.962`), so stateA suppresses the pair as UNKNOWN. Small false red/green fragments disappear while the true garlic-container addition remains; pair IoU improves 0.6918→0.7223.
3. **Wrong preserved association/regression:** `closet_1_closet_2` falls from IoU 0.4045 to 0.1678. Three low-location clothing associations are interpreted as geometry-supported MOVED and then removed by the existing visibility filter. This consumes endpoints whose baseline ADDED/REMOVED masks overlapped real moved-bucket clothing. TP falls from 152,007 to 58,227—a loss of **93,780 pixels**, exactly the experiment's entire aggregate TP loss. In this crowded/deformable scene, appearance identity plus unreliable/incomplete geometry is not enough to decide object state.

## StateB diagnostics

Relative to stateA, stateB reclassifies **72 candidate ADDED** and **16 candidate REMOVED** decisions as explicit UNKNOWN. It removes **0 ADDED FP, 0 REMOVED FP, and 0 total FP pixels**, loses **0 TP pixels**, and changes recall by **0.0000**. All 10 label maps are pixel-identical.

This is not an implementation failure. Under the frozen `v6_no_dino` configuration, the proposed stateB evidence test is already the active visibility filter: unmatched T1 masks are emitted as ADDED only when at least the existing 0.8 fraction lies on reference-render coverage, and unmatched T0 masks are emitted as REMOVED under the same support rule. StateB gives the rejected candidates the more accurate name UNKNOWN but adds no new evidence or suppression.

Required case inspections therefore concern the candidates whose semantic record changes, not newly removed output pixels:

1. **Correct false REMOVED UNKNOWN:** `store_57_store_58`, T0 object 16 covers 41,802 pipeline pixels in an unreconstructed right-side region, has render support 0.0, and overlaps zero T1 GT pixels. StateB records it as UNKNOWN; stateA had already filtered it from output.
2. **Correct false ADDED UNKNOWN:** `closet_1_closet_2`, T1 object 15 covers 14,334 pixels, has render support 0.0, and overlaps zero GT pixels. Again, stateB makes the uncertainty explicit but removes no additional output.
3. **True change marked UNKNOWN:** `closet_1_closet_2`, T1 object 12 has only 0.038 render support, but 25,705 of its 26,465 mask pixels (97.1%) overlap moved-bucket GT. This demonstrates the inherent recall limitation of coverage-only evidence. The TP had already been suppressed by stateA/baseline visibility logic, so stateB causes no incremental recall loss.

## Answers

1. **Does stateA alone improve CORGI?** No overall. It improves precision and three queries, but pooled IoU, mean IoU, F1, and recall all decline; one deformable closet scene loses 93,780 TP pixels.
2. **Does it recover meaningful MOVED predictions?** Only partially. One bedroom bag is meaningful, but two of three retained MOVED objects are entirely false positive, and aggregate MOVED precision is only 36.8%. Geometry also promotes 9 more associations to MOVED that visibility subsequently suppresses.
3. **Does stateB provide additional benefit?** No. It is pixel-identical to stateA because its allowed evidence and threshold reproduce the already-active visibility filter.
4. **Does stateB improve precision at an unacceptable recall cost?** No incremental precision or recall change occurs. The precision/recall tradeoff belongs entirely to stateA: +0.0467 precision at −0.1133 recall, with slightly worse IoU and F1. The experiment therefore does not support shipping either resolver as implemented.

The mechanistic conclusion is narrower than “state resolution does not matter.” The baseline's rejected-identity → unmatched-state transition clearly creates FP, and stateA removes many of them. The failure is that binary geometric same-place evidence cannot reliably distinguish true displacement, proposal-boundary mismatch, and incomplete geometry; consuming the association irrevocably can erase real change. A useful next resolver would need explicit competing state hypotheses with uncertainty retained through visibility reasoning, rather than an early MOVED/UNKNOWN commitment.

## Artifacts

- Baseline summary: `results/scenediff_diagnostic/SceneDiff/_experiments/v6_no_dino/summary.json`
- stateA summary: `results/scenediff_diagnostic/SceneDiff/_experiments/v6_no_dino_stateA/summary.json`
- stateB summary: `results/scenediff_diagnostic/SceneDiff/_experiments/v6_no_dino_stateB/summary.json`
- Replay seed: `results/scenediff_diagnostic/SceneDiff/*/*/v6_no_dino_inventory_seed/inventory/bundle.pkl`
- Configs: `configs/scenediff_v6_no_dino_stateA.yaml`, `configs/scenediff_v6_no_dino_stateB.yaml`
