# PASLCD suppression ablation (v0–v5)

Detect-only ablation over a fixed subset of **30 refine-complete PASLCD
queries spanning 15 scene instances**, driven by
`scripts/run_suppression_ablation.py`. Every variant reuses the *same* cached
`reconstruction/` and `refined/` inputs under
`results/paslcd_ablation_cache/`, so only stage 5 (object-state resolution)
is recomputed and the variants are directly comparable to each other.

**Machine:** all variants ran locally on `tessa-Alienware-x16-R2` (RTX 4090
Laptop) in the `goldilocs` env with `sam2.vos_optimized: true`. The office PC
runs that flag `false` (a Blackwell sm_120 bug) and scores the identical 30
queries differently — measured at −0.033 mIoU, +0.089 precision, −0.083
recall. **Do not compare these numbers against office-PC numbers.** Within
this table the comparison is valid.

**Sample caveat:** the 30 queries were selected to be diverse and biased
toward queries with many unresolved leftover objects. These are not
benchmark numbers; they are variant-to-variant deltas.

## What differs between variants

Only these five config keys differ anywhere across the six configs. `None`
means the key is absent from the YAML and the dataclass default applies.

| key | v0 | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| `enable_color_replacement_detection` | True | **False** | True | True | **False** | True |
| `recover_unmatched_via_tracking` | True | True | **False** | True | **False** | True |
| `enable_visibility_filter` | True | True | True | **False** | True | True |
| `enable_horizon_suppression` | – | – | – | – | – | **True** |
| `maximum_above_horizon_fraction` | – | – | – | – | – | **0.5** |

Config hashes (sha256, first 12), for provenance:

```
v0_baseline               b53ec2bdaf71
v1_no_color_replacement   f0792bc328fc
v2_no_recall_recovery     81766502ca58
v3_no_visibility_filter   64f26712f408
v4_no_color_no_recovery   4ccbdfdbd2ec
v5_horizon_suppression    c3eecb3045f8
```

## v4 → v5: exactly what changes

**v5 is not v4 plus a feature.** v4 is a double-knockout (color replacement
*and* recall recovery both off); v5 restores both and adds one new
mechanism. Relative to v4, three keys move at once:

| key | v4 | v5 | direction |
|---|---|---|---|
| `enable_color_replacement_detection` | False | True | restored to default |
| `recover_unmatched_via_tracking` | False | True | restored to default |
| `enable_horizon_suppression` | (absent → False) | **True** | **the new mechanism** |

**Therefore v5's correct comparison baseline is v0, not v4** — v5 = v0 + one
flag. v4 exists to test additivity between v1 and v2 and is a dead end for
the horizon question. Any v5-minus-v4 delta conflates three changes and
should not be reported as the effect of horizon suppression.

### The new mechanism (v5)

`enable_horizon_suppression` drops a removed/added/moved/replaced decision
when more than `maximum_above_horizon_fraction` (0.5) of its mask lies above
the horizon, where "above the horizon" is a per-pixel boolean computed in
`reconstruction.above_horizon_map()` from the **query camera's pose and
intrinsics only**.

Motivation, measured on these same 30 queries (`v0_baseline` predictions vs
PASLCD GT):

| criterion | share of FP px | share of real changed px | ratio |
|---|---|---|---|
| top quarter of image | 45.5% | 3.0% | **15.1×** |
| top third of image | 54.1% | 5.4% | 10.0× |
| on a dominant RANSAC plane | 62.8% | 70.8% | 0.9× |
| 3D elevation above 90th pct | 15.0% | 2.5% | 5.9× |

Two design decisions follow from that table:

1. **Not plane-based.** Planes do not discriminate (0.9×) because real
   changes sit *on* floors and tables.
2. **Not elevation-based.** Elevation is much weaker (5.9×) for a structural
   reason: ceiling and sky are exactly where the reconstruction is least
   reliable, so filtering them by their own recovered geometry filters on
   noise. The view-direction criterion depends only on camera pose.

The up-axis is recovered without gravity (VGGT's world frame is arbitrary):
the **axis** comes from RANSAC on the dominant ground plane (aligned with
scene vertical in 26/30 queries), and the **sign** from the query camera's
own up vector — the obvious "most content lies above the ground" heuristic
picks the wrong side in 14 of those 30.

Verified on synthetic cameras (`estimate_world_up` / `above_horizon_map`):

| camera pitch | above-horizon pixels | horizon row (of 480) |
|---|---|---|
| level | 50.0% | 240 |
| down 20° | 12.5% | 60 |
| down 45° | 0.0% | – |
| straight down | 0.0% | – |
| up 30° | 100.0% | – |

The bottom rows are the safety property: a downward-looking robot head
suppresses nothing, where a fixed "top N% of the image" crop would delete
real objects at the far end of a table.

### Expected vs measured

An image-space proxy (zeroing the top 20% of every prediction) scored
**+0.0247 mIoU / +0.0777 precision / −0.0001 recall, 17 queries better, 0
worse, 13 tied**. That is a *proxy*, not this mechanism — it should agree
closely on PASLCD because those cameras are near-level, but v5 is the number
that counts and is reported as the result regardless of which way it lands.

### Prerequisite

v5 needs `above_horizon.npy` in each cached query's `reconstruction/`
directory. The cache predates the feature and never stored camera poses, so
`scripts/backfill_above_horizon.py` re-runs VGGT-Omega localization to
recover them, writing only that one new buffer and leaving every other
cached file untouched. **Without the backfill the filter silently no-ops and
v5 scores identically to v0** — if v5 ties v0 to four decimal places, check
that the maps exist before interpreting it.


**Backfill incident (2026-09-09 04:06).** The first backfill run was killed
by the kernel OOM-killer (system RAM, not GPU) on `Lunch_room_Instance_1`,
whose reference set is 109 images -- the largest in the sample; 72-image
instances passed. It is re-run with `--max-reference-images 64` (uniform
stride, first/last kept). This is a deviation from the cache, whose renders
came from the full reference set: `above_horizon.npy` for the 9 instances
backfilled after the cap is derived from a 64-image reference cloud. The
map is a coarse per-pixel boolean from the query camera's pose and the
scene up-axis, so the subsample changes it negligibly, but the manifest of
any v5 result must carry this note. The 12 maps written before the kill
used the full sets.

## Results

| variant | mIoU | F1 | precision | recall | ΔmIoU vs v0 | better/worse/tied |
|---|---|---|---|---|---|---|
| v0_baseline | 0.1675 | 0.2669 | 0.3109 | 0.2774 | — | — |
| v1_no_color_replacement | 0.1632 | 0.2596 | 0.3179 | 0.2602 | −0.0043 | 7 / 9 / 14 |
| v2_no_recall_recovery | 0.0697 | 0.1212 | 0.0963 | 0.3243 | −0.0978 | 0 / 25 / 5 |
| v3_no_visibility_filter | 0.1192 | 0.1969 | 0.1926 | 0.2792 | −0.0483 | 0 / 21 / 9 |
| v4_no_color_no_recovery | 0.0672 | 0.1169 | 0.0948 | 0.3079 | −0.1003 | 0 / 25 / 5 |
| v5_horizon_suppression | 0.1334 | 0.2168 | 0.3647 | 0.1896 | −0.0341 | 10 / 9 / 11 |
| **v6_ceiling_sky_suppression** | **0.1727** | **0.2755** | **0.3199** | **0.2774** | **+0.0052** | **6 / 0 / 24** |

## v5 result: NOT what the proxy predicted, and NOT shipped as the SceneDiff base

Measured (2026-09-09, all 30 maps present, RAM-safe rerun):

| | mIoU | F1 | precision | recall |
|---|---|---|---|---|
| v0_baseline | 0.1675 | 0.2669 | 0.3109 | 0.2774 |
| v5_horizon_suppression | 0.1334 | 0.2168 | 0.3647 | 0.1896 |
| delta | **−0.0341** | −0.0501 | +0.0538 | **−0.0878** |

The image-space proxy predicted +0.0247 mIoU / −0.0001 recall. Precision
moved the predicted direction; recall collapsed 88× more than predicted.
This is exactly the "an aggregate score is not proof of the mechanism"
check the diagnostic protocol calls for, and it failed the naive read.

### Root cause, isolated

Splitting the 30 queries by their logged `above_horizon` fraction (from
`localize_and_render_query`'s own diagnostic print):

| group | n | mean ΔmIoU |
|---|---|---|
| 22 queries, fraction < 55% | 22 | **+0.0430** (better than the +0.0247 proxy) |
| 8 queries, fraction == exactly 100% | 8 | **−0.2462** (catastrophic) |

The mechanism is sound and better than predicted where the up-axis is
correct. It fails completely, not partially, on 8/30 (27%) of queries:
recall on those 8 collapses to near-zero (examples: 0.584→0.006,
0.463→00.015, 0.383→0.000) because the ENTIRE frame reads as "above
horizon," so every decision's mask exceeds the 0.5 rejection fraction.

**Visual cause** (`render_t0.png` for 3 failing vs. 2 passing queries):
the 8 failures are near-frontal, floor-poor views -- a kitchen counter shot
almost head-on (Cantina, both instances, 4/4 queries fail), a garden
wall/shelf shot with no ground visible (Pots_Instance_1, 2/2 fail), a
meeting-room shot with much more wall/ceiling than floor in frame
(Meeting_room_Instance_1/IMG_1863). The passing control (Porch) and the
passing Meeting_room query (IMG_1870, same instance as a failing one) both
show the floor clearly dominant. **`estimate_world_up`'s RANSAC finds the
scene's single largest coplanar point cluster and assumes it is the
ground.** When the reference photos for an instance are dominated by
near-frontal, floor-poor framing, the largest flat surface is a cabinet
front, fence panel, or wall instead -- and once that wrong plane is treated
as "ground," a near-frontal query camera can end up with nearly every pixel
reading as "above" it.

This also explains why it is not purely a per-instance property: two
queries against the *same* reference reconstruction (Meeting_room_Instance_1,
IMG_1863 vs IMG_1870) landed on opposite sides, because each query re-aligns
the reference cloud into its own coordinate frame before RANSAC runs, and
small numerical differences between two comparably-sized planar candidates
can flip which one wins.

### Decision: v0_baseline is the SceneDiff base, not v5

Against the selection criteria fixed before this experiment (precision/
recall balance, catastrophic false positives, stability across scenes,
whether the behavior is fixable) -- v5 fails on stability and catastrophic
failure specifically: a 27% catastrophic-failure rate that erases nearly
all recall on the affected queries is disqualifying on its own, independent
of the aggregate mIoU. **v0_baseline is selected as the most promising
variant and the SceneDiff diagnostic base**, since nothing in this
diagnosis found a problem with it beyond the pre-existing MOVED/recall
findings already on record.

### Candidate fix (not implemented -- flagged for the ranked list, not a sweep)

The likely-correct fix is a second, largely scene-content-independent
signal: real cameras (handheld or head-mounted) are rarely rolled far from
gravity-up, so the median of the *camera's own* up vector
(`-extrinsic[:3,:3][1]`) across every reference image in an instance -- not
just the query -- should approximate true up almost regardless of what the
dominant visible surface is. Using it to validate or replace the
RANSAC-plane-derived axis (e.g., reject the RANSAC axis when it disagrees
with the median camera-up by more than some angle) is a plausible one-line
robustness fix, but implementing and re-validating it is new work, not a
threshold sweep on an already-run mechanism, and is deferred to the ranked
next-steps list rather than done now.

## v6: semantic ceiling/sky detection replaces the geometric approach

Implements the user's proposed fix directly: instead of inferring "above
horizon" from RANSAC plane-fitting + camera pose (which failed
catastrophically on 8/30 queries, see above), ask a real detector "is there
a ceiling or sky region here" via SAM3's grounded text-prompt mode
(`change_detection.detect_ceiling_sky_mask`, prompts `("ceiling", "sky")`,
union of all detections >= 0.5 confidence). No RANSAC, no up-axis, no
assumption about which surface is the floor. Empty detection -> empty mask
-> nothing suppressed, which is the "only triggered when we actually see a
ceiling" property the user required -- inherent to using a real detector,
not a tuned fallback.

**Implementation bug caught before the 30-query run**: the first version
assumed this could reuse the SAM3 model already loaded for proposal
generation (`generator`). It could not -- `Sam3AutomaticMaskGenerator`
builds its model with `enable_segmentation=False` (only the grid-point
automatic-proposal path is needed there), so the grounding head was never
loaded, and the first call raised `KeyError: 'pred_masks')` inside
`_forward_grounding`. Caught by an end-to-end smoke test on one real cached
query before it could waste a 30-query run. Fixed with a genuinely separate,
lazily-loaded `Sam3TextPromptDetector` (`stages/sam3_proposals.py`), built
once and reused across a whole `detect_batch.py` batch the same way
`generator`/`dino_extractor`/`tracker` already are -- costs one extra SAM3
model load, not the "free, already-warm" reuse the design first assumed.

**Empirical validation before the full run**: text-prompted "ceiling"
correctly returned zero detections on both PASLCD queries that broke the
geometric approach at 100% (Cantina_Instance_1/IMG_2870, a near-frontal
kitchen-counter shot; Pots_Instance_1/IMG_E2658, a garden wall/shelf shot
with no ground visible), and found a precise 6.6%-of-frame region on
Meeting_room_Instance_1/IMG_1863's real ceiling (vs. the geometric
approach's wrong 100%). "sky" scored 0.973 confidence with a pixel-accurate
mask on a real outdoor scene (Playground), correctly excluding houses,
fence, and playground equipment.

### Result (30/30 queries, full run)

| | mIoU | F1 | precision | recall |
|---|---|---|---|---|
| v0_baseline | 0.1675 | 0.2669 | 0.3109 | 0.2774 |
| v5_horizon_suppression (geometric) | 0.1334 | 0.2168 | 0.3647 | 0.1896 |
| **v6_ceiling_sky_suppression (semantic)** | **0.1727** | **0.2755** | **0.3199** | **0.2774** |

**Per-query: 6 better, 0 worse, 24 tied.** Recall is unchanged to four
decimal places (+0.0000) -- no query lost any true positive to this
mechanism. The worst single-query change is exactly 0.0000
(Cantina_Instance_1/IMG_2870, the smoke-tested case), confirming the
isolated validation matches the full-batch behavior exactly. All 6 gains
come from Lounge, Lunch_room, and Meeting_room -- scenes with a real,
correctly-detected ceiling -- with no losses anywhere, including Cantina and
Pots where v5 catastrophically failed.

The gain (+0.0052 mIoU) is smaller than the naive image-crop proxy's
+0.0247 or v5's precision gain (+0.0538), because it is deliberately
conservative: it removes decisions only where a real ceiling/sky region was
actually found, not everything above an arbitrary line or a possibly-wrong
plane. That conservatism is the entire point -- it trades away some of the
proxy's optimistic upside for the zero-regression, zero-catastrophic-
failure property the geometric version could not deliver.

### v6 supersedes v5 and v0 as the PASLCD base

Against the fixed selection criteria (precision/recall balance,
catastrophic false positives, stability across scenes, whether the
behavior is understandable and fixable): v6 is a strict improvement over
v0 (0 losses, +mIoU, +precision, unchanged recall) and has none of v5's
instability. **v6_ceiling_sky_suppression is now the most promising
variant and the SceneDiff diagnostic base**, replacing the earlier
v0_baseline selection.

## v7: occlusion-aware REMOVED suppression + REPLACED disabled (user-proposed)

Two independent changes bundled into one variant, cache-only (detect-side
only, no reconstruction rerun):

1. **Occlusion-aware REMOVED suppression** (new mechanism,
   `enable_occlusion_aware_removal_suppression`): a REMOVED decision whose
   render_t0 footprint is mostly (>50%) covered by an ADDED object's
   image_t1 footprint is reclassified rather than reported as a separate
   change. Fixes a real bug in the process: the final labels raster drew
   in priority order (ADDED, REMOVED, MOVED), so wherever the two
   overlapped, REMOVED used to win the pixel -- backwards from what is
   visually true (something new is sitting there right now). Validated with
   synthetic unit tests (full overlap, partial overlap, unrelated removal)
   and an end-to-end smoke test on a real cached query before the full run
   (276 pixels changed, all REMOVED->UNCHANGED, zero ADDED pixels touched).
2. **`enable_color_replacement_detection: false`** -- REPLACED disabled
   entirely, per instruction ("we do not need that anymore").

### Result (30/30 queries)

| | mIoU | F1 | precision | recall |
|---|---|---|---|---|
| v0_baseline | 0.1675 | 0.2669 | 0.3109 | 0.2774 |
| v6_ceiling_sky_suppression | 0.1727 | 0.2755 | 0.3199 | 0.2774 |
| v7_occlusion_and_no_replace | 0.1680 | 0.2678 | 0.3320 | 0.2576 |

v7 vs v0: +0.0005 mIoU (essentially flat), +0.0211 precision, **-0.0198
recall**. Per-query: 11 better, 6 worse, 13 tied.

**The flat aggregate hides two changes pulling in opposite directions --
disentangled below rather than left as one bundled number:**

- **Occlusion suppression fired 27 times across 16/30 queries.** Among
  those 16, 13 improved or were flat and only 3 worsened -- and all 3 of
  those also lost REPLACED detections in the same query (confounded with
  the second change, not evidence against occlusion suppression on its
  own). This mechanism looks like a genuine, validated improvement.
- **Every one of the 6 regressed queries had REPLACED active in v0**
  (counts 1-10), with no exceptions. The worst,
  Cantina_Instance_1/Inst_1_test_IMG_2870 (-0.0599 mIoU), had 10 REPLACED
  decisions in v0 alone. This revises the earlier v1 finding ("REPLACED is
  marginal and scene-dependent, negative only on Meeting_room") -- Cantina
  depends on it heavily too, which the earlier framing did not surface.
  Disabling REPLACED costs real mIoU on the scenes that rely on it, exactly
  as instructed; flagged here as new evidence for that decision, not a
  reversal of it.

**Recommendation if isolating further:** ship occlusion-aware suppression
on its own (v6 + occlusion, REPLACED left on) to get its clean gain without
the REPLACED trade-off -- not run in this session; would need one more
cache-only detect pass to confirm.

## v8: isolating occlusion suppression from the REPLACED confound

Ran `v8_occlusion_only` = v6 + occlusion-aware suppression, with REPLACED
left **on** (unlike v7, which also disabled it). This is the true isolation
the "recommendation" above called for.

| variant | n | mIoU | F1 | precision | recall |
|---|---|---|---|---|---|
| v6_ceiling_sky_suppression (base) | 30 | 0.1727 | 0.2755 | 0.3199 | 0.2774 |
| **v8_occlusion_only** | 30 | **0.1723** | **0.2752** | **0.3237** | **0.2748** |
| v8 vs v6 | | **-0.0004** | -0.0003 | +0.0038 | -0.0026 |

**Result: a wash, not a clean win.** This corrects the v7 write-up's framing
above ("13/16 improved, only 3 worsened, all 3 confounded with lost REPLACED
detections") -- that framing turns out to have been wrong about the
*mechanism*, not just under-isolated. With REPLACED held constant, the exact
same 27 suppressions across the same 16/30 queries occur (confirming the
mechanism itself doesn't depend on REPLACED), but per-query:

- 12 queries improve (mostly tiny: +0.0001 to +0.019)
- **2 queries regress**, and one of them --
  `Cantina_Instance_1/Inst_1_test_IMG_2870` (-0.0488 mIoU alone) -- outweighs
  all 12 gains combined
- 16 queries unaffected (fired but zero net effect, or didn't fire)

**Root cause of the regression** (checked directly against decision counts,
not assumed): in that query v6 correctly labels 6 objects REMOVED; v8's
occlusion suppression reclassifies 3 of them to UNCHANGED because their 2D
masks overlap >50% with an unrelated ADDED object's mask -- but they aren't
actually occluded by it, they're at a different depth that happens to
project to the same screen region in this single view. The heuristic (2D
pixel overlap as a proxy for "hidden behind") is correct exactly when it
should be (Cantina_2908, Garden_E7267 improve) and wrong exactly when two
unrelated objects share screen space without true 3D occlusion -- and there
is no way to tell the two cases apart from pixel overlap alone.

**Conclusion: occlusion-aware suppression is not ready to ship as a default
on PASLCD.** The mechanism is directionally correct but the 2D-overlap
implementation isn't reliable enough -- it trades real fixes for a
comparably-sized real regression, netting to roughly zero. A depth-aware
version (checking whether the ADDED object's position buffer places it in
front of the REMOVED object, using the same world-position buffers already
computed for the geometric-identity test, rather than only checking 2D mask
overlap) would resolve the ambiguity directly and is the natural next step
-- not another threshold sweep on the 50% cutoff, which cannot distinguish
these two cases no matter where it's set.

### v8 post-mortem: the implementation was wrong, not just the idea

Visual inspection of the Cantina masks (user, 2026-09-09) showed REMOVED
slabs drawn inside the ADDED open-drawer region. Traced to two defects in
`suppress_removed_behind_added`, confirmed by pixel counts (IMG_2870: ADDED
pixel count identical between v6 and v8, 4986; only 2551 REMOVED pixels
went away):

1. The labels raster draws ADDED then REMOVED, so REMOVED already owned
   every shared pixel. v8 cleared only the *non*-overlapping part of a
   suppressed object and left the overlap as-is -- i.e. still REMOVED.
   The suppressed decisions never reached the mask.
2. Retained (minority-overlap) REMOVED objects also overdrew the addition
   they touched.

And the metric regression had a different cause than the earlier
"unrelated objects sharing screen space" story: 96.3% of the 2551 wiped
pixels are GT change. The open drawer is under-segmented by SAM3 (tray
items, not the drawer), so the hidden drawer front's uncovered remainder
was real change that no ADDED mask covered; reverting it to unchanged is
what cost recall.

**Fix (in code, replaces v8's behaviour):** a majority-covered REMOVED
footprint is merged into the addition (relabelled ADDED, whole footprint)
and the raster is rebuilt with REMOVED under ADDED. By construction the
changed-pixel set equals v6's -- verified pixel-identical on both
Cantina_1 queries via fast replay -- so the binary PASLCD metric is
exactly v6 (0.1727) and only the class semantics change. IMG_2908 now
comes out as one clean ADDED drawer; IMG_2870 keeps three REMOVED objects
under the 50% cutoff (overlap 0.474 / 0.372 / 0.000; the first two are
99-100% GT change and physically also hidden behind the open drawer, the
third is a genuine removal on the fridge). A 2D-overlap fraction cannot
separate those cases; a depth test (image_t1_positions closer to the
camera than render_t0_positions inside the REMOVED footprint => occluded)
can, and all three position buffers are already in the cache.

**Fast-replay validation on real data (same session):** slow re-runs of
both queries reproduced v8 pixel-identically; replays from the dumped
bundles matched the slow path in every decision except float round-off in
2-4 `dino_cosine` values at the 7th decimal (no decision flips). Replay
wall time 8-26s/query, dominated by the ceiling/sky text-detector load,
versus 130-200s slow.

## v9: depth-based occlusion test (replaces the 2D overlap rule)

Motivation: on IMG_2870 the 2D rule kept two hidden drawer-front pieces
(overlap 0.474 / 0.372, both ~100% GT change) and no overlap threshold
separates "hidden behind something new" from "old surface gone, cavity
now visible" -- both overlap the addition in 2D.

**Mechanism.** `render_t0_positions` and `image_t1_positions` are both
expressed through the query camera, so along a REMOVED footprint the t1
surface is either *nearer* (occluded), *farther* (revealed) or *equal*
(no depth evidence -- thin objects, the appearance decision stands). The
camera centre is not stored anywhere; it is recovered per query as the
least-squares intersection of the (p_t0, p_t1) lines at pixels displaced
by >5% of scene scale (`_estimate_camera_centre`): 18k lines on each
Cantina_1 query, centre stable to 3 decimals across displacement cutoffs
0.05/0.1/0.2, median point-line residual 0.2-0.8% of scene scale, every
valid t1 point inside one forward half-space. Margin 0.05 x scene_scale;
background rate on GT-unchanged pixels: 0.4-0.8% "nearer", 2-3%
"farther". An object is occluded when >=25% of its depth-valid footprint
is nearer and nearer outweighs farther; occluded footprints are merged
into the addition as in the v8 fix. The 2D rule is the fallback when the
buffers are missing or the centre estimate is rejected.

**Result on the two Cantina_1 queries (fast replay, v9 config = v8 config +
two new defaults):**

| query | t0 obj | 2D overlap | t1 nearer | t1 farther | GT change | 2D rule | depth rule |
|---|---|---|---|---|---|---|---|
| 2870 | 3 | 0.47 | 1.00 | 0.00 | 1.00 | keep REMOVED | **merge** |
| 2870 | 5 | 0.52 | 0.67 | 0.00 | 1.00 | merge | merge |
| 2870 | 6 | 0.99 | 1.00 | 0.00 | 1.00 | merge | merge |
| 2870 | 11 (fridge) | 0.00 | 0.00 | 0.00 | 0.00 | keep | keep (no evidence) |
| 2870 | 21 (cavity) | 0.37 | 0.00 | 0.24 | 0.99 | keep | keep (revealed) |
| 2870 | 24 | 0.70 | 0.59 | 0.00 | 0.57 | merge | merge |
| 2908 | 4 | 0.94 | 0.50 | 0.00 | 0.47 | merge | merge |
| 2908 | 11 (sink) | 0.00 | 0.01 | 0.01 | 0.00 | keep | keep |
| 2908 | 22 | 0.69 | 0.53 | 0.01 | 0.49 | merge | merge |
| 2908 | 23 (cavity) | 0.84 | 0.00 | 0.16 | 1.00 | merge | **keep (revealed)** |
| 2908 | 28 | 0.98 | 0.63 | 0.00 | 0.75 | merge | merge |

The two disagreements are both where the signed-depth map shows the
cavity behind the pulled-out drawer (t1 farther): the depth rule keeps
those REMOVED, which is the physically correct reading (the surface is
gone, something is not in front of it), and it rescues the hidden piece
the 2D rule missed. Changed-pixel set still pixel-identical to v6 on both
queries (binary metric = v6 by construction); the depth stage costs
0.1-0.2s. The 30-query semantic result needs the inventory dump pass
(not run yet).

## v10: DINOv2 removed (user decision, 2026-09-09)

`v10_no_dino` = v9 + `use_dino_features: false`: DINOv2 is neither
computed nor consulted by any identity gate/score/recovery step (the
helpers `_descriptor_valid` / `_appearance_pass` / `_appearance_score`
already drop a disabled descriptor rather than veto through it). The
stated basis is a prior experiment showing DINOv2 adds nothing; that
result is not written up in this repo (the m1-m5 model-set ablation was
planned in scenediff_diagnostic.md but never run here), so v10 vs v9 on
the 30 PASLCD queries is the measurement.

Smoke test on Cantina_1/IMG_2870 before the full run: slow path without
DINO, replay from its bundle, and replay of the DINO-full bundle under v10
are all pixel-identical to each other, and identical to v9 -- DINO changed
nothing on that query. A bundle dumped without DINO now carries provenance
and refuses to load under a DINO-on config (verified: the v9 replay of the
DINO-less bundle raises). Note the stage-02 timer barely moves with DINO
off (25.0s vs 26.5s): that stage is dominated by proposal selection and
descriptor pooling, not the DINOv2 forward pass, so the saving is VRAM and
model load, not wall time.

**30-query result (fast replay from the cache, 223s for all 30):**

| variant | n | mIoU | F1 | precision | recall |
|---|---|---|---|---|---|
| v6_ceiling_sky_suppression (base, DINO on) | 30 | 0.1727 | 0.2755 | 0.3199 | 0.2774 |
| **v10_no_dino** (v9 + no DINO) | 30 | **0.1734** | **0.2764** | **0.3226** | **0.2766** |
| v10 vs v6 | | +0.0007 | +0.0009 | +0.0027 | -0.0008 |
| v6_no_dino_state_resolver (other session: m2_no_dino + conservative resolver) | 30 | 0.1466 | 0.2368 | 0.2668 | 0.2730 |

Per query v10 vs v6: 11 up, 4 down, 15 flat, every delta within
+/-0.007. Removing DINOv2 from every identity decision is neutral-to-
slightly-positive on PASLCD, consistent with the "DINOv2 is useless"
premise; the -0.026 of the other session's run is therefore attributable
to its conservative state resolver, not to dropping DINO. v10 is the new
base for anything that follows.

The bundles were written by that other session's run (same stage-1-3
config, verified section-by-section), which is why v10 needed no slow
pass at all.

### Depth-test availability across the 30 queries

The first v10 replay showed the camera-centre estimate accepted on only
14/30 queries -- all 16 rejections were the 0.02 residual cutoff, set
from the two Cantina queries (0.006-0.008). Across all 30 the median
residual spans 0.002-0.055 of scene scale, and the "t1 nearer" false-
positive rate on GT-unchanged pixels stays at 0.2-1.8% over the whole
range, so the cutoff was rejecting usable fits. Relaxed to 0.10; the
re-replay is metric-identical (by construction) with the depth test
active on 30/30 queries: 21 REMOVED objects merged into additions across
10 queries, all on depth evidence, none via the 2D fallback. Class labels
changed on 8 queries (30-756 px each) relative to the 0.02-cutoff run,
kept as `v10_no_dino_resid0.02/` for comparison.

Caveat found in the same table: the "t1 farther" direction is NOT reliable
everywhere -- 23-43% of GT-unchanged background reads as farther in
Playground and Lounge (a systematic depth bias between the single-view t1
depth and the splat render in large/outdoor scenes; Cantina and
Meeting_room sit at 1-10%). That direction only ever keeps an object
REMOVED, so its failure is conservative (no merge), but the "revealed
cavity" reading is trustworthy on close-range indoor scenes, not
universally.

**v0 baseline reminder for context:** v8 vs v0_baseline (0.1675) is
+0.0048 mIoU -- that gain is almost entirely v6's ceiling/sky suppression
(+0.0052 on its own, see above), not the occlusion suppression added on top
of it (-0.0004). v6's ceiling/sky suppression remains the one variant in
this whole ablation series with an unambiguous, well-understood positive
effect.

### Findings so far

- **Recall recovery is the single largest contributor** (−0.098 mIoU when
  removed) and is, despite its name, a **precision** mechanism on PASLCD:
  disabling it *raises* recall (+0.047) and collapses precision (−0.215).
  This is the first isolated ablation of it on PASLCD rather than ChangeSim.
- **The visibility filter costs essentially no recall** (+0.0018 when
  disabled) while carrying +0.118 precision. This refutes the hypothesis
  that motivated the experiment — that the filter was eating recall because
  96.5% of missed pixels had geometry present.
- **Color replacement / REPLACED is marginal and scene-dependent**
  (−0.0043, 7 better / 9 worse). Its whole negative comes from Meeting_room;
  excluding those 4 queries the mean flips to +0.0017. The earlier "17%
  precision, 29% of FP pixels" statistic was a misleading decision criterion
  — the class also carries recall nothing else recovers.
- **Neither suppressor is where the recall is going.** Disabling both
  recovers only ~+0.05 recall combined while wrecking precision, so the
  missing 62% of changed pixels is lost upstream at proposal/matching, not
  in the filters.

- **v1 and v2 are additive and independent.** v4 (both off) lands at
  −0.1003 mIoU against a predicted −0.1021 from v1 + v2; per class the
  ADDED/REMOVED false-positive deltas of v4 are pixel-identical to v2's
  (+548,875 / +531,106) and the REPLACED deltas identical to v1's
  (−15,016 FP / −4,326 TP). The two mechanisms act on disjoint pixels.
- **What the two suppressors buy, in pixels** (`analysis.md` §D): removing
  recall recovery adds +549k ADDED and +531k REMOVED FP pixels for only
  +1.8k / +6.4k TP; removing the visibility filter adds +82k / +174k FP for
  +173 TP. Catastrophic-FP queries (FP > 50k px and precision < 0.10) go
  6 → 23 without recovery and 6 → 13 without the filter.
- **Where the residual baseline error is** (§B/§C): REMOVED carries 52.7%
  of FP pixels at 0.257 precision, ADDED 40.7% at 0.288; REPLACED 6.6% at
  0.224. **MOVED never fires — 0 TP and 0 FP pixels across all 30 queries**;
  every MOVED candidate is `location_mismatch_rejected` (5.3/query). Of the
  baseline's FP pixels, 39% sit in the top 20% of the image (ceiling/sky)
  against 2% of GT — the v5 target — and only 1% in render holes.
- **Runtime split** (per query, this laptop): SAM3 inventory+features
  92.7 s (63%), DINOv2 35.5 s (24%), SAM2 tracking 6.9 s (5%), resolution
  2.1 s. Removing DINOv2 is the only model removal that would matter for
  speed.
