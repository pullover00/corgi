# SceneDiff diagnostic iteration — pre-registration

Written **before** any run on this subset. Everything below that constrains
the experiment (subset, criteria, variants, metrics scope) is fixed here;
results are appended underneath later, and nothing above the results line
is edited afterwards.

## Why SceneDiff now

Every design decision so far was validated on PASLCD. The last three
findings there — the visibility filter costing no recall, recall recovery
acting as a precision mechanism, above-horizon suppression being a 15×
discriminator — may all be PASLCD-shaped. This iteration is a diagnosis on
a different, real-video dataset, not a tuning pass. **No parameter sweep.**

## A discrepancy that has to be fixed first

The existing `run_scenediff_batch.py` wrote a detect manifest carrying only
`render_t0 / clean_render / image_t1 / output_dir`. Without the position,
coverage, confidence and corroboration buffers, the visibility filter,
geometric identity and reference corroboration all silently no-op (each is
gated on its buffer being supplied). **Every SceneDiff number produced so
far therefore came from a different effective pipeline than PASLCD** — the
0.1386 pooled IoU (27 pairs) had no visibility filter and no geometric
identity. The new `run_scenediff_diagnostic.py` passes every buffer plus
`above_horizon`. Direct comparison of its numbers with the old 0.1386 is
not valid and is not attempted.

## Methodology

    t0: `reconstruction.frames_per_video` (10) frames sampled uniformly from
        original_video1, always including the annotators' representative t0
        frame  ->  reconstruct_reference_scene, once, in isolation
    t1: exactly ONE frame — the representative t1 frame — from
        original_video2  ->  localize_and_render_query
    then refine (DI²FIX) -> detect -> evaluate

Never multiple t1 frames jointly. Original videos only (`original_video*`),
never the `vis/` visualization videos.

## Subset: 10 pairs, criteria fixed in advance

Selected by `scripts/select_scenediff_diagnostic_subset.py` (seed 0) from a
pool of 344 annotated pairs with both original videos on disk. Selection
order: (1) a declared quota of 2 worst + 2 best pairs by the *previous*
shipped30 pixel IoU — the only performance-informed input, requested so
known failures and successes are revisited; (2) coverage fills so every
change kind, both difficulty bands (by annotation count, never by score),
rigid and deformable objects, and as many scene types as possible occur,
with at most 2 pairs per scene type (kitchen is 50% of the pool). Viewpoint
spread is not annotated in SceneDiff and is not a criterion.

| pair | scene | change kind | objs | difficulty | deform. | reason |
|---|---|---|---|---|---|---|
| P01-20240203-184214_0030_…_0032 | kitchen | moved_bucket_only | 1 | easy | rigid | quota: prior worst (0.000) |
| P01-20240204-095114_0001_…_0011 | kitchen | mixed | 3 | hard | rigid | quota: prior worst (0.000) |
| closet_1_closet_2 | closet | moved_bucket_only | 4 | hard | unknown | quota: prior best (0.518) |
| bedroom_28_bedroom_29 | bedroom | moved_bucket_only | 2 | medium | unknown | quota: prior best (0.687) |
| store_57_store_58 | candy | added_only | 1 | easy | rigid | coverage: added_only |
| store_39_store_40 | market | removed_only | 2 | medium | rigid | coverage: removed_only |
| bedroom_32_bedroom_33 | bedroom | mixed | 4 | hard | deformable | coverage: deformable |
| table_5_table_6 | table | mixed | 3 | hard | unknown | coverage: new scene type |
| gym_3_gym_4 | gym | mixed | 2 | medium | rigid | coverage: new scene type |
| living_room_49_living_room_50 | living_room | mixed | 2 | medium | rigid | coverage: new scene type |

Coverage: change kinds {moved_bucket_only 3, mixed 5, added_only 1,
removed_only 1}; 8 scene types; difficulty {easy 2, medium 4, hard 4};
{rigid 6, deformable 1, unknown 3}. "Change kind" is derived from the
annotations' `in_video1`/`in_video2` flags (removed = in 1 only, added = in
2 only, moved_bucket = in both); SceneDiff's own "moved" definition is
"present in both videos", not verified displacement.

## What the pixel metric can and cannot see

`scenediff_gt_eval.py` builds GT in image_t1's own pixel space from the
video2 annotations at the query frame. Consequences, stated up front:

- **ADDED** and **moved-bucket** objects: in scope.
- **REMOVED** objects: **out of scope.** They never appear in the query
  frame; their old footprint in query-camera coordinates is unknowable
  without the very reconstruction under test (circular). REMOVED recall is
  not measurable here and will not be reported as a number.
- Per-class analysis on SceneDiff is therefore ADDED / moved-bucket for
  recall, and all predicted classes for precision (any predicted pixel on GT
  background is a false positive regardless of class).

## Variants (model-set ablation), base = provisional

Base config: `ablate_v5_horizon_suppression.yaml` (v0 + above-horizon
suppression), chosen provisionally as the most promising PASLCD variant
pending v4/v5. If the PASLCD analysis selects a different base,
`make_model_ablation_configs.py --base <other>` regenerates all five and
this line is updated. Each variant is the base plus exactly the listed
`three_image_comparison` overrides:

| variant | models | overrides |
|---|---|---|
| m1_full | SAM3 masks + SAM2 + SAM3 feat + DINOv2 + geometry | — |
| m2_no_dino | SAM3 masks + SAM2 + SAM3 feat + geometry | `use_dino_features: false` |
| m3_no_sam3_features | SAM3 masks + SAM2 + DINOv2 + geometry | `use_sam_features: false` |
| m4_tracking_geometry | SAM3 masks + SAM2 + geometry | `enable_appearance_correspondence: false`, both `use_*_features: false` |
| m5_features_geometry_no_sam2 | SAM3 masks + SAM3 feat + DINOv2 + geometry | `enable_tracking: false` |

Structural facts these flags act on (verified in code, 2026-09-09): every
appearance gate — identity, bridging, recall recovery, part suppression —
was a strict AND over `sam ≥ 0.65` and `dino ≥ 0.60`, and the ranking
score was `min(sam, dino)`. Requiring both is a recall reducer by
construction; whether the second descriptor buys precision is what m2/m3
measure. m5 loses clean-render bridging and recall recovery inherently
(both are track-driven) — that loss is part of what removing SAM2 costs.

Shared stages (reference reconstruction, localization, render, refine) are
computed once per pair under experiment "shared" and reused by all five;
each variant's manifests record those stages' hashes as upstream.

## Per-variant records

For each: pooled t1-space IoU / precision / recall, mean per-pair IoU,
changed-pixel fraction, ADDED and moved-bucket recall, per-class predicted
pixels on GT background (false-positive attribution by class), counts of
ADDED/REMOVED decisions, `tracking_recoveries`, geometric rescues (from
`decisions[].evidence`), `visibility_filter_rejected`,
`horizon_suppressed`, and stage timings from `inference.json`.

## Query-frame rule (added before the first run, after the GT-availability check)

A dry check of GT availability at the representative t1 frame failed for 2
of the 10 pairs. Cause: `representative_frame_index` picks the most common
video2 frame index among in_video2 annotations, tie-breaking to the first;
for `table_5_table_6` that chose frame 0, which has no decodable mask,
while frame 85 carries GT for *both* in-scope objects. The rule is replaced
uniformly (`scripts/scenediff_select_query_frames.py`, output
`diagnostic_subset_queries.json`, read by the runner):

1. candidates = every video2 frame an in_video2 annotation names
2. choose the one with the most in-scope GT objects whose masks decode
   there; ties -> most GT pixels -> lowest index
3. removed-only pairs (`store_39_store_40`) have no in-scope GT at any
   frame: use the frame at the same relative position through video2 as the
   t0 representative frame is through video1

Consequences, stated now: `store_39_store_40` stays in the subset but its
pixel IoU is **0 by construction** — every predicted pixel is a false
positive — so it is reported as a hallucination-volume measurement, not as
a failure. No pair was swapped. This rule is data-availability only; no
model output was consulted.

---

## Results

*(appended after the runs; nothing above this line is edited afterwards)*
