# Headline stage-11 compatibility contract

The current exported stage 11 contains post-headline behavior. Merely copying
the numeric thresholds is insufficient. A `headline_68_51` compatibility
profile must preserve the following control flow.

## Inputs and output selection

1. The scored raster baseline is association-resolver
   `r4_no_geometry_ablation` for each split. Tracking-v4 artifacts supply
   evidence only; their label raster is not the baseline.
2. Source candidates, forward tracks, consolidated SAM3 target proposals,
   SAM-feature maps, DINO-feature maps, and reconstruction/depth must come from
   the exact cache graph named in the archived YAML.
3. The headline output is the full-mask result (`labels_full_mask.png`), not the
   guarded default (`labels.png` / `labels_guarded.png`).

Fresh upstream generation is a different experiment unless every input cache
is fingerprinted. In the failed export, reconstruction/geometry/render hashes
for `Warehouse_9_Seq_0_944` matched the historical artifacts, but the SAM3
proposal inventories did not (historical source/target 92/247; fresh 93/243).
That difference changes tracking, R4, and all later object decisions before
stage 11 runs.

The stored failed-export artifacts (known to be a mixed-revision run, so used
only diagnostically) make the location of the regression explicit:

| 25-pair raster | binary mIoU | multiclass mIoU |
| --- | ---: | ---: |
| historical frozen R4 baseline | 68.1098% | 38.9969% |
| historical headline stage 11 (full) | 68.5144% | 40.5173% |
| failed export tracking raster | 58.5231% | 28.9698% |
| failed export resolver raster | 60.1990% | 29.6782% |
| failed export stage 11 (full) | 60.5306% | 30.8727% |

Stage 11 still improved those stored predictions, but it began from a
non-equivalent upstream raster. The primary observed score loss therefore
precedes stage 11; its later code paths are a second compatibility difference,
not the sole cause. A clean, fingerprinted run is required for final numbers.

## Settings

All common `slot_inconsistency` values in the archived YAML are the headline
values. A compatibility profile must omit or disable these later settings:

- `asymmetric_cleanup_*`
- `weak_identity_*`
- `arbitration_*`

No other common slot threshold differs between the archived YAML and the
current exported `configs/pipeline.yaml`.

## Exact control-flow differences

| operation | headline behavior | later exported behavior to disable |
| --- | --- | --- |
| Plausibility | Accept a compact target, an aligned fragment, or a target whose single strict cleanup result passes. | Two-signal fragmented-identity veto. |
| Cleanup evidence | One `eligible` result requires identity mismatches, one-to-one or compact geometry, and target ADDED/REMOVED support. The same eligible rows define old-source cleanup and trusted full targets. | Separate `source_cleanup_eligible`, asymmetric old-envelope route, and force-cleanup rows. |
| Rejected targets | Run `dominant_changed_class_consensus` only on targets rejected by plausibility. | Dataset-wide target arbitration, target-to-clean absence gating, and source-fragment rewrite to REMOVED. |
| Depth ownership | Return one ownership mask. A plausible target with valid depth is unified across existing R4 class stripes. | Protected-REMOVED mask, identity override mask, and source-front component protection. |
| Old footprint | Drop exactly `old_mask & ~new_mask & (labels == REMOVED)`. | Connected-REMOVED expansion, protection, and forced override. |
| Full raster | Paint all ownership-approved target pixels REPLACED. | Any later cleanup/arbitration pre-rewrite. |
| Guarded raster | Paint baseline-changed target pixels plus strict trusted targets; apply the same ownership mask. | Later cleanup/arbitration pre-rewrite. |
| Floor cleanup | Suppress only ADDED components meeting the archived robust floor-plane component thresholds. | Any broader semantic cleanup. |

The archived inference sequence is the normative implementation:

1. consolidate source and target hypotheses;
2. match aligned object slots;
3. compute DINO patch retention plus SAM, DINO, color, support-surface, and
   identity-elsewhere evidence;
4. deduplicate replacement candidates by target and promotion score;
5. apply strict cleanup and object-plausibility gates;
6. add only independently verified adjacent companions;
7. run consensus only for plausibility-rejected targets;
8. compute front-most ownership with target-object unification;
9. rasterize full and guarded variants;
10. suppress floor-dominant ADDED components;
11. freeze prediction hashes before loading ground truth.

## Evaluation semantics

- Normalize raw ChangeSIM class `4` to canonical REPLACED class `5`.
- Accumulate one pixel-level 6-by-6 confusion matrix over all pairs; do not
  average per-image IoUs.
- Binary changed means any nonzero canonical label, including WARPED.
- Multiclass mIoU averages canonical labels `0, 1, 2, 3, 5` with support;
  WARPED (`4`) is not a scored ChangeSIM ground-truth class.
- Require prediction and target shapes to match rather than silently resizing
  during a reproduction audit.

Compatibility is established only when the 25 baseline and full-prediction
array hashes match `replay/predictions_frozen.json`, followed by exact aggregate
metrics. Matching aggregate metrics alone is weaker and can hide pair-level
differences.
