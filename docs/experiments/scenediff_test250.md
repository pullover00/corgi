# SceneDiff 250-pair test split — held-out run (pre-registration)

Written **2026-09-10 22:32, before launch**. Everything above the "Results"
line is fixed now; results are appended underneath and nothing above that
line is edited afterwards.

## Purpose

A single, frozen-configuration run of CORGI on the official SceneDiff
**test split** so the paper has a SceneDiff number that was not produced on
the 10-pair development subset. No tuning, no reruns after inspection, no
per-pair intervention.

## Frozen inputs (hashes recorded before launch)

| input | value |
|---|---|
| method | `v10_no_dino`, **refine OFF** (the shipped baseline) |
| config | `configs/scenediff_v10_no_dino_no_refine.yaml`, sha256 `b5a8eee5ae05675d…` — `configs/ablate_v10_no_dino.yaml` with the single change `refine.enabled: false` |
| code | branch `scenediff-refinement-reference-interaction`; the launcher prints the exact commit at `TEST250_START` in the run log |
| pair list | `data/scenediff_benchmark/test_split_250.txt` — the 250 pairs of `splits/test_split.json` (150 varied + 100 kitchen) in the split file's own order, sha256 `75b50acd73c24367…` |
| query frames | `data/scenediff_benchmark/test250_queries.json`, sha256 `6ce9ccb9f4842d90…` — one T1 frame per pair, chosen by the pre-registered data-availability rule of `scenediff_select_query_frames.py` (max decodable in-scope GT objects; ties → most GT pixels → lowest index; removed-only pairs → same relative position as the T0 representative frame). Regression-checked: it reproduces the 9 diagnostic-subset presets that fall in this split exactly. No model output was consulted. |
| T0 reference | 10 evenly spaced frames of `original_video1` plus the annotators' representative T0 frame (runner default) |
| detection | `detect_batch.py --sequential-model-lifecycle` (identical results, lower peak VRAM; required on this 16 GB card) |
| machine | Alienware x16 R2, RTX 4090 Laptop (Ada), driver 580.173.02; vggt-omega env torch 2.12.0+cu130; goldilocs env torch 2.5.1+cu124; SAM2 `vos_optimized: true` |
| outputs | `results/scenediff_test250/SceneDiff/`, experiment `scenediff_v10_no_dino_no_refine` (a fresh tree — no artifact from the diagnostic tree is reused) |

## Execution

`scripts/run_scenediff_test250.sh`: 10 chunks of 25 pairs, sequential, each
chunk a separate runner invocation with its own log. Chunking exists only
because the runner commits detect-stage manifests after a whole batch; a
kill loses the current chunk, not the run. Re-running the launcher skips
finished chunks. Expected wall time ≈ 14–15 h.

## Metric (as in the diagnostic runs; not SceneDiff's official metric)

Pooled t1-space pixel IoU = TP/(TP+FP+FN) over the single query frame per
pair, with precision, recall, F1 and mean per-pair IoU. GT is the union of
ADDED and moved-bucket object masks in the query frame; **REMOVED objects
are out of scope** (they are not in the query frame). This is a restricted
metric and must not be presented as comparable to the SceneDiff paper's
multi-frame point-in-box AP. Running the official evaluator on these same
outputs is a separate, later step.

## Aggregates that will be reported (all declared now)

1. **All 250 pairs** of the official split.
2. **Evaluable pairs only** — those with any in-scope GT pixels. The query
   selection found **46/250 pairs are removed-only with no in-scope GT at any
   frame**; their IoU is 0 by construction and they contribute only FP
   (hallucination volume). n = 204 expected.
3. **Held-out** — minus the 9 diagnostic-subset pairs that are in the test
   split (`P01-…184214_0030→0032`, `P01-…095114_0001→0011`, `closet_1_closet_2`,
   `bedroom_28_bedroom_29`, `store_57_store_58`, `store_39_store_40`,
   `bedroom_32_bedroom_33`, `gym_3_gym_4`, `living_room_49_living_room_50`).
   These were visible during development; `table_5_table_6` is in val.
4. **Held-out ∩ evaluable.**

Per-pair IoU/TP/FP/FN for every pair. Reconstruction and evaluation
failures are listed by name and counted; they are not silently dropped.

## Disclosures fixed in advance

- The 10-pair diagnostic subset (4 of them chosen by prior performance) was
  used for diagnosis and one-variable ablations; the v10 detection changes
  were validated on PASLCD and are pixel-identical to v6_no_dino on that
  subset. Nothing in this run's configuration was chosen using the test split.
- Numbers from this machine are **not** interchangeable with the Blackwell
  SceneDiff PC (SAM2 `vos_optimized` differs; VGGT-Omega numerics differ):
  pooled 0.3287 vs 0.2676 on the same 10 pairs. Every number below is from
  this machine only.
- Reconstruction-failure pairs (expected: some `P0x` kitchen pairs, as on
  2026-09-08) are reported as failures, not excluded.

## What will NOT happen after launch

No threshold, config, query-frame, pair-list or code change; no rerun of any
pair "to check"; no per-pair exclusion beyond the four declared aggregates.
If the run is interrupted, the launcher is re-run as is.

## Addendum A — protocol defect found after launch, before any evaluation (2026-09-10)

**Status when found (~22:40):** chunk 1 of 10, ~10 pairs reconstructed, **zero pairs through
detection or evaluation**. No metric had been computed or seen. File mtimes record the ordering:
the runner patch (22:41), the corrected selector (22:42), the v2 query file and affected-pair
list (22:43) and this addendum's draft (22:43) all predate the first chunk aggregate that became
visible (chunk_01 summary, 23:31) by ~48 minutes. The run was stopped at 23:35 with chunks 1-2
partially done; the decision to fix was Tessa's (option A, 2026-09-10).

**Defect.** SceneDiff's annotation frame indices (`video1_frame_idx`, `video2_frame_idx`) index
the 30 fps review videos `video{1,2}.mp4`. This pipeline deliberately reads
`original_video{1,2}` (the review videos have objects repainted with flat colors) and assumed a
shared index space. They differ for half the split: `original_video*` is **10 fps for all 100 P0x
kitchen pairs** (exactly 1/3 the frames), 60 fps for 19 and 120 fps for 6 varied pairs, and
~30 fps (identity) for the rest. Verified by image correlation over the 183 pairs with a non-zero
query index: `video2.mp4[i]` matches `original[round(i*fps/30)]` at **median 0.992** and
`original[i]` at **median 0.266**.

**Consequence under the frozen protocol.** 38 pairs fail at frame extraction (index past the end
of a shorter original -- `bathroom_9_bathroom_10` was the trigger) and others silently use a query
frame from the wrong moment, evaluated against GT for a different instant. Counting both the T1
query frame and the annotators' representative T0 frame (which is added to the T0 reference set
and is remapped by the same rule), **98 of 250 pairs** are affected: 31 through T1
only, 41 through both, 26 through T0 only. The same defect explains the 2026-09-08 kitchen
reconstruction failures, and it affects one diagnostic-subset pair
(`P01-…095114_0001→0011`, annotation index 68 read as original frame 68 instead of 23) -- one of
the two near-zero P01 pairs in the earlier reports was therefore partly this bug, not the method.

**Fix (data handling only).** `annotation_to_original_index` in `scripts/run_scenediff_batch.py`
maps an annotation index to the original video by `round(i * fps_orig / 30)`, clamped to the last
frame. Ratios within 5% of 1 are treated as identity (clamp only), so the ~30 fps pairs keep their
exact frames and continuity with every earlier SceneDiff run is preserved -- `closet_1_closet_2`
still uses frame 298. The annotation index remains the query directory name and is what the
evaluator receives, since GT masks are keyed in annotation space. **No threshold, config, matching
logic, metric, pair list or aggregate definition changed.** `t1_idx` is identical to v1 for all
250 pairs; only the extracted frame moves.

**New frozen input:** `data/scenediff_benchmark/test250_queries_v2.json`, sha256 `c13aad0cd139eeb8…`,
recording both `t1_idx` (annotation) and `t1_idx_original` (extraction). The affected-pair list is
`data/scenediff_benchmark/test250_frame_fix_affected_pairs.json`.

**Cache handling.** The artifact store does not hash the extracted frame, so stale outputs would
have read as fresh. The 4 affected pairs already present in the results tree
(`bathroom_3_bathroom_4`, `bathroom_9_bathroom_10`, `bedroom_18_bedroom_19`, `cabinet_1_cabinet_2`)
were deleted, along with all chunk `.done` markers; unaffected pairs are reused as cache hits.

**Unchanged:** everything else in this pre-registration, including the four declared aggregates,
the metric, the pair list and its order, and the commitment that nothing is altered after results
are seen.

---

## Results

*(appended after the run; nothing above this line is edited afterwards)*

**Run:** started 2026-09-10 23:39, finished 2026-09-11 09:49 (10 h 10 min), commit `80cf9a4`,
config `b5a8eee5ae05675d…`, queries `c13aad0cd139eeb8…`. 240/250 pairs evaluated, 10 failures.
Every failure was identified by name and index *before* it occurred (see below); no unforecast
failure happened.

### The four declared aggregates

| aggregate | n | pooled IoU | P | R | F1 | mean IoU |
|---|---:|---:|---:|---:|---:|---:|
| All evaluated (official split) | 240 | 0.1240 | 0.1392 | 0.5319 | 0.2206 | 0.1418 |
| Evaluable only (in-scope GT) | 197 | 0.1458 | 0.1673 | 0.5319 | 0.2545 | 0.1728 |
| Held-out (minus 9 diagnostic pairs) | 231 | 0.1204 | 0.1351 | 0.5265 | 0.2150 | 0.1334 |
| Held-out ∩ evaluable | 189 | 0.1416 | 0.1623 | 0.5265 | 0.2481 | 0.1631 |

Pooled totals over all evaluated pairs: TP 9,995,584, FP 61,818,429, FN 8,794,966.
43 evaluated pairs have empty in-scope GT (IoU 0 by construction, FP only); 58 score exactly 0.

### The headline finding: the development subset was optimistic by ~2.7x

The 9 diagnostic-subset pairs that fall in this split average **mean IoU 0.3574**,
against **0.1334** for the 231 held-out pairs. The
10-pair development number reported in `SCENEDIFF_OVERNIGHT_ABLATIONS.md` and
`SCENEDIFF_REFINEMENT_REFERENCE_INTERACTION.md` (pooled 0.3287 for this same configuration) is
therefore **not** representative of SceneDiff: the held-out pooled figure is **0.1204**. This is
the expected consequence of a subset that was partly chosen by prior performance (2 best + 2
worst of the earlier shipped30) and then used throughout development, and it is the reason this
run exists. Per-pair diagnostic scores here reproduce the earlier ones closely
(e.g. store_57 0.727, bedroom_28 0.685, closet_1 0.416), so the gap is subset composition, not
a change in method behaviour.

### Character of the errors at scale

Recall holds up (0.5319) while precision collapses (0.1392):
FP outnumbers TP 6.2:1. The method generally *finds* the changed object and then
over-predicts across the rest of the scene — the ADDED/REMOVED false-positive dominance already
measured at 86.3% of FP pixels on the diagnostic subset, now confirmed on 240 pairs. Shelf-heavy
retail scenes are the clearest cases (e.g. `store_31_store_32`: recall 99.8% of 20,003 GT pixels,
with 1.63M FP). Best pairs: table_15 0.971, kitchen_18 0.845, bathroom_9 0.815, bus_1 0.777.

### Failures (10), all pre-identified

Nine are OpenCV seek failures on videos whose later frames are unseekable although they decode
sequentially — a pre-existing reader quirk in `extract_frames`, unrelated to the frame-index fix,
and present in the frozen protocol. An exhaustive test of every sampled T0 index across all 250
pairs predicted exactly this set of 9, and each failed at the predicted index:
`living_room_37_living_room_38`, `living_room_39_living_room_40`,
`P01-…184214_0032→0038`, `P01-…095114_0000→0001`, `P01-…120411_0001→0007`,
`P02-…195833_0000→0004`, `P02-…111822_0048→0049`, `P04-…151722_0028→0033`,
`P04-…162750_0007→0018`.
The tenth, `workspace_1_workspace_2`, is an evaluator edge case: it has `in_video2` annotations
but none whose masks decode near any candidate frame, so the query selector fell back to the
positional rule while `load_query_gt`'s empty-GT path requires *no* `in_video2` objects at all.
It is the only such pair in the split. Neither defect was fixed: both lie in code the
pre-registration froze, and the options were reported to Tessa rather than acted on.

### Breakdown by split category (SD-V / SD-K)

Added 2026-09-11 on request. The grouping is the split file's own (`varied` = SD-V, 150 pairs;
`kitchen` = SD-K, 100 pairs), not a post-hoc partition, and each of the four declared aggregates
is reported within it.

| category | aggregate | n | pooled IoU | P | R | F1 | mean IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| SD-V (varied) | all evaluated | 147 | 0.1380 | 0.1594 | 0.5071 | 0.2425 | 0.1716 |
| SD-V (varied) | evaluable only | 116 | 0.1670 | 0.1994 | 0.5071 | 0.2862 | 0.2174 |
| SD-V (varied) | held-out | 140 | 0.1333 | 0.1539 | 0.4995 | 0.2353 | 0.1582 |
| SD-V (varied) | held-out ∩ evaluable | 110 | 0.1614 | 0.1925 | 0.4995 | 0.2779 | 0.2013 |
| SD-K (kitchen) | all evaluated | 93 | 0.0828 | 0.0858 | 0.7006 | 0.1529 | 0.0948 |
| SD-K (kitchen) | evaluable only | 81 | 0.0898 | 0.0934 | 0.7006 | 0.1649 | 0.1088 |
| SD-K (kitchen) | held-out | 91 | 0.0831 | 0.0861 | 0.7033 | 0.1534 | 0.0953 |
| SD-K (kitchen) | held-out ∩ evaluable | 79 | 0.0902 | 0.0938 | 0.7033 | 0.1655 | 0.1098 |

SD-V evaluated 147/150 (3 failures), SD-K 93/100 (7 failures — the seek quirk is concentrated in
the P0x videos).

**The two categories fail differently, and the aggregate hides it.**

| | SD-V | SD-K |
|---|---:|---:|
| held-out pooled IoU | 0.1333 | 0.0831 |
| pooled recall | 0.4995 | **0.7033** |
| pooled precision | 0.1539 | **0.0861** |
| FP : TP | 5.3 : 1 | **10.7 : 1** |
| median GT pixels / evaluable pair | 85,158 | **20,627** |

SD-K has the *higher* recall of the two — the method finds more of the annotated change in
kitchen scenes — but its precision is roughly half SD-V's and it over-predicts twice as heavily.
The proximate reason is scale: SD-K's changed objects are about a quarter the size of SD-V's
(median 20.6k vs 85.2k GT pixels), so the same absolute volume of spurious prediction costs far
more IoU. Reporting SD-K as simply "worse" would misdescribe it: it is not a detection failure
but a precision failure on small targets.

Best pairs: SD-V table_15 0.971, kitchen_18 0.845, bathroom_9 0.815, bus_1 0.777;
SD-K P02-…120927_0014 0.687, P01-…150506_0050 0.644, P01-…152323_0002 0.444.

Note the 9 diagnostic-subset pairs in this split are almost all SD-V (7 of 9), so the
development-subset optimism documented above is mostly an SD-V effect; SD-K was barely
represented during development (2 pairs, both near-zero).

### Caveats that stand

- This is the restricted t1-space pixel metric, **not** SceneDiff's official multi-frame
  point-in-box AP. Not comparable to the paper's numbers.
- REMOVED objects are out of scope; 43 evaluated pairs can only contribute false positives.
- Single machine (RTX 4090 Laptop, `vos_optimized: true`). The Blackwell SceneDiff PC produces
  materially different numbers on identical inputs; these two sets must not be mixed.

### Artifacts

`results/scenediff_test250/SceneDiff/_experiments/scenediff_v10_no_dino_no_refine/summary_all.json` and `summary_all.md` (per-pair IoU/TP/FP/FN for all 250, with a
`seen in dev` column); per-chunk logs and summaries under `results/scenediff_test250/SceneDiff/_experiments/scenediff_v10_no_dino_no_refine/logs/`.

