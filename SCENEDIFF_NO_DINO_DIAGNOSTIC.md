# SceneDiff `v6_no_dino` diagnostic

## Protocol and execution diagnosis

This run used the preregistered 10-pair subset in `data/scenediff_benchmark/diagnostic_subset.txt` and its fixed queries in `diagnostic_subset_queries.json`. Each pair used the existing 10-frame T0 reference reconstruction and exactly one T1 query frame. The method config was `configs/model_ablation_m2_no_dino.yaml`; stored inference settings confirm `use_dino_features: false`, and the direct DINO similarity matrices are identically zero. No thresholds or detection logic were changed.

The previous diagnostic produced no `inference.json` files because its recorded invocation stopped at `--through refine`. The runner deliberately returned before detection; this was not a model failure. Running the same diagnostic through detection fixed the execution issue. Artifact manifests reported every reconstruction, localization, render, and DI2FIX/refinement stage as compatible and cached, so only method-specific detection and evaluation ran.

One evaluation-only integration issue then affected `store_39_store_40`: this preregistered removed-only pair has no video2 object masks, and the evaluator could not infer an output canvas. `scripts/scenediff_gt_eval.py` was minimally changed to read the selected native T1 frame shape and construct the already-specified empty T1 ground truth. No prediction was altered. The final run attempted, reconstructed, inferred, and evaluated all **10/10** pairs; there are no failed pairs.

## Quantitative results

| Pair | Type | px/im IoU | Precision | Recall | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| P01 184214 0030→0032 | moved bucket | 0.000 | 0.000 | 0.000 | 0 | 136,324 | 8,810 |
| P01 095114 0001→0011 | mixed | 0.000 | 0.000 | 0.000 | 0 | 112,776 | 5,678 |
| closet_1→closet_2 | moved bucket | 0.405 | 0.797 | 0.451 | 152,007 | 38,648 | 185,111 |
| bedroom_28→bedroom_29 | moved bucket | 0.550 | 0.551 | 0.996 | 23,590 | 19,205 | 83 |
| store_57→store_58 | added only | 0.692 | 0.700 | 0.983 | 89,619 | 38,410 | 1,508 |
| store_39→store_40 | removed only | 0.000* | 0.000 | 0.000 | 0 | 231,289 | 0 |
| bedroom_32→bedroom_33 | mixed | 0.572 | 0.600 | 0.924 | 78,914 | 52,523 | 6,510 |
| table_5→table_6 | mixed | 0.201 | 0.306 | 0.369 | 60,538 | 137,297 | 103,500 |
| gym_3→gym_4 | mixed | 0.442 | 0.442 | 0.996 | 81,421 | 102,591 | 309 |
| living_room_49→living_room_50 | mixed | 0.636 | 0.645 | 0.979 | 29,527 | 16,241 | 627 |
| **Pooled pixels** |  | **0.301** | **0.368** | **0.623** | **515,616** | **885,304** | **312,136** |

Pooled binary F1 is **0.463**. Mean per-pair px/im IoU is **0.350**. The pooled IoU is the appropriate aggregate over TP/FP/FN; the mean gives each pair equal weight.

\* As preregistered, removed-only objects are out of scope for T1-space ground truth because they do not appear in video2. Consequently `store_39_store_40` has empty GT, IoU zero by construction, and its 231,289 predicted pixels measure hallucination volume rather than removed-object recall.

The resolver produced **42 ADDED, 70 REMOVED, 0 MOVED, 124 REPLACED, and 148 UNCHANGED object decisions**. Their output occupied 426,701 ADDED, 804,209 REMOVED, 0 MOVED, and 170,010 REPLACED pixels. Background false-positive attribution was:

| Predicted class | FP pixels | Share of all FP |
|---|---:|---:|
| ADDED | 126,123 | 14.2% |
| REMOVED | 637,904 | 72.1% |
| REPLACED | 121,277 | 13.7% |
| MOVED | 0 | 0.0% |

ADDED plus REMOVED therefore account for **764,027 / 885,304 = 86.3%** of all false-positive pixels. The run also recorded 82 direct accepted identities, 65 SAM2 recovery events, 105 visibility-filtered candidates, no recorded geometric rescue, and 58 location-mismatch rejections (31 direct and 27 clean-render bridge). Those rejected pairs release their endpoints back into the unmatched-object path.

## Qualitative diagnosis

1. **False ADDED/REMOVED detections remain dominant.** REMOVED alone supplies 72.1% of all FP pixels; ADDED and REMOVED together supply 86.3%. The removed-only market view and first kitchen view show broad red masks on static shelving, appliances, and render boundaries. The prediction is not merely a class-label issue: these masks dominate the changed-pixel area.

2. **The main failure is the correspondence-to-state handoff, amplified by render defects.** Gross localization is generally sound: cached camera-alignment residuals span only 0.000285–0.002704. Render quality is nevertheless causal in the worst examples. The first kitchen render has large black/unreconstructed regions and distorted object boundaries; the market render is incomplete around a changed viewpoint, both of which seed unmatched reference proposals. But render quality is not sufficient to explain the errors: the second kitchen and table renders are visually well aligned and still miss small annotated objects while marking static objects. Across the set, 58 otherwise plausible identities are rejected on location and their endpoints can become independent ADDED/REMOVED hypotheses. The final unmatched-object classification then turns correspondence uncertainty, proposal fragmentation, or render holes into confident physical state changes. Visibility filtering removes 105 candidates but does not resolve that ambiguity.

3. **The richer multi-view T0 reference helps, but does not cure the PASLCD failure mode.** Five pairs reach IoU above 0.4, four exceed 0.55, and the pooled recall is 0.623. In well-covered views such as `store_57_store_58` and `living_room_49_living_room_50`, CORGI cleanly outlines the newly present garlic container, cup, and toy. This is consistent with useful multi-view reconstruction and geometry. It is not a causal comparison—SceneDiff and PASLCD have different images and annotations—but it is substantially stronger than the PASLCD-30 `v6_no_dino` mean IoU of 0.173. The persistence of 86.3% ADDED/REMOVED FP shows that additional T0 views improve favorable cases without solving state resolution.

4. **MOVED is effectively absent: exactly zero MOVED decisions and zero MOVED pixels.** This occurs even though the subset contains three moved-bucket-only and five mixed pairs. Some moved-bucket GT pixels are recovered, but under ADDED, REMOVED, or REPLACED labels. SceneDiff's moved bucket means presence in both videos rather than verified displacement, so it cannot establish MOVED class accuracy; it does establish that CORGI's explicit MOVED mechanism is not firing.

5. **There are clear SceneDiff successes.** `store_57_store_58` (0.692), `living_room_49_living_room_50` (0.636), `bedroom_32_bedroom_33` (0.572), and `bedroom_28_bedroom_29` (0.550) are substantially above the PASLCD diagnostic average. The store and living-room overlays are especially convincing at the principal changed objects, although each retains smaller false changes on static content.

## Decision

**C. The same structural state-resolution failure dominates → test the state-resolver next.**

The present method is capable of strong localization when the render is complete, so the result does not argue for abandoning SceneDiff. However, expanding the benchmark now would mainly measure the already-visible structural error: unmatched or location-rejected proposals are promoted directly to ADDED/REMOVED, producing 86.3% of all FP pixels and no MOVED output. The next experiment should be the already-defined minimal state resolver on these same cached 10 pairs, without tuning or algorithm changes after inspection; a larger SceneDiff run should wait for that result.

## Reproducibility

- Config: `configs/model_ablation_m2_no_dino.yaml`
- Fixed subset: `data/scenediff_benchmark/diagnostic_subset.txt`
- Fixed query map: `data/scenediff_benchmark/diagnostic_subset_queries.json`
- Experiment outputs: `results/scenediff_diagnostic/SceneDiff/*/*/v6_no_dino/`
- Aggregate record: `results/scenediff_diagnostic/SceneDiff/_experiments/v6_no_dino/summary.json`
- Invocation: `python scripts/run_scenediff_diagnostic.py --experiment v6_no_dino --config configs/model_ablation_m2_no_dino.yaml`
