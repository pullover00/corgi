# SceneDiff overnight ablations (2026-09-10)

Baseline config: `configs/ablate_v10_no_dino.yaml` (DINOv2 disabled throughout, as in every
experiment below). Fixed preregistered 10-pair subset (`data/scenediff_benchmark/diagnostic_subset.txt`),
exactly one T1 query per pair (`diagnostic_subset_queries.json`), no multi-T1 fusion, no threshold
tuning, no post-result modifications. Every experiment is a single-variable change from the baseline;
no ablations are combined. All reported IoU/precision/recall/F1 are pixel-pooled over the 10 pairs
unless labelled "mean" (equal-weight per-pair average). Full per-pair data, decision counts, and case
inspections are saved under `results/scenediff_diagnostic/SceneDiff/_experiments/` (see Reproducibility).

## Experiment 0: baseline

`scenediff_v10_no_dino`, config `configs/ablate_v10_no_dino.yaml`, reused the existing cached
10-frame T0 reconstructions and shared render/refine artifacts already on disk. **10/10 reconstructed,
10/10 evaluated**, no failures. Pooled IoU **0.3010**, mean per-pair IoU **0.3498**, precision
**0.3681**, recall **0.6229**, F1 **0.4627**, mean render coverage **0.755**. This is bit-for-bit
consistent with the earlier `v6_no_dino` diagnostic run (same numbers to 4 decimals), which used an
equivalent config, confirming the two configs behave identically on this subset.

## Table 1: reference-view experiment (Experiment 1)

| N T0 views | Pooled IoU | Mean IoU | Precision | Recall | F1 | Coverage | Recon. time (10 pairs) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.1516 | 0.1865 | 0.2221 | 0.3231 | 0.2633 | 0.430 | 160.7 s |
| 3  | 0.0929 | 0.1118 | 0.1046 | 0.4549 | 0.1700 | 0.568 | 150.9 s |
| 5  | 0.1461 | 0.1820 | 0.1665 | 0.5432 | 0.2549 | 0.730 | 153.7 s |
| 10 (baseline) | 0.3010 | 0.3498 | 0.3681 | 0.6229 | 0.4627 | 0.755 | 185.5 s |

Reference frames were fixed **before** any evaluation: evenly spaced, nested positions in each pair's
existing 10（or 11)-frame baseline T0 list (`nested_reference_subset`, e.g. baseline indices
`[0,38,76,113,151,189,227,264,302,340]` → N=1 keeps only index 189, N=3 keeps `{0,189,340}`, N=5 keeps
`{0,76,189,264,340}` ⊂ N=10). No frame was chosen by reconstruction quality or CORGI accuracy.

FP-by-class (pooled, 10 pairs):

| N | TP | FP | FN | ADDED FP | REMOVED FP | MOVED FP | REPLACED FP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 267,463 | 936,538 | 560,289 | 387,004 | 498,728 | 0 | 50,806 |
| 3 | 376,563 | 3,225,131 | 451,189 | 2,098,961 | 1,049,037 | 0 | 77,133 |
| 5 | 449,660 | 2,250,755 | 378,092 | 1,378,252 | 772,660 | 0 | 99,843 |
| 10 | 515,616 | 885,304 | 312,136 | 254,183 | 509,844 | 0 | 121,277 |

### Per-pair IoU

| Pair | N=1 | N=3 | N=5 | N=10 |
|---|---:|---:|---:|---:|
| P01 184214 0030→0032 | 0.0012 | 0.0002 | 0.0023 | 0.0000 |
| P01 095114 0001→0011 | 0.0039 | 0.0005 | 0.0005 | 0.0000 |
| closet_1→closet_2 | 0.0000 | 0.2691 | 0.4020 | 0.4045 |
| bedroom_28→bedroom_29 | 0.4837 | 0.0003 | 0.0956 | 0.5502 |
| store_57→store_58 | 0.2257 | 0.2229 | 0.2773 | 0.6918 |
| store_39→store_40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bedroom_32→bedroom_33 | 0.0000 | 0.0000 | 0.1915 | 0.5721 |
| table_5→table_6 | 0.1417 | 0.1302 | 0.2216 | 0.2009 |
| gym_3→gym_4 | **0.9474** | 0.0815 | 0.1132 | 0.4417 |
| living_room_49→living_room_50 | 0.0611 | 0.4133 | 0.5158 | 0.6364 |

### Diagnosis

**N=10 decisively beats every sparse subset** (pooled IoU 0.30 vs 0.09–0.15), but the relationship
among N=1/3/5 is **not monotonic**: N=3 (pooled 0.093) is worse than N=1 (0.152), and N=5 (0.146) does
not clearly beat N=1 either, even though mean render coverage rises monotonically and smoothly with N
(0.430 → 0.568 → 0.730 → 0.755). Two distinct, traceable failure modes drive this, not noise:

1. **Intermittent total localization failure at low N.** `closet_1_closet_2` at N=1 and
   `bedroom_32_bedroom_33` at N=1 *and* N=3 render with **zero coverage** (TP=0, FP=0, the whole frame
   comes back empty) — the single-camera-center similarity fit this ablation required (N=1 has no second
   camera center to constrain rotation/scale, so `_fit_single_frame_transform`'s single-frame depth-ratio
   fallback was added specifically for this experiment) or a small-subset fit for N=3 lands so far off
   that the aligned point cloud never intersects the query frustum. `bedroom_32_bedroom_33`'s N=3
   alignment residual is 0.0116 vs baseline's 0.00098 (12x), confirming a genuinely bad fit, not an
   artifact of scoring. **The N=1 alignment-residual diagnostic itself is not trustworthy**: with one
   camera center there is nothing independent to check the fit against, so the reported residual is
   trivially ≈0 even when the fit is badly wrong (both zero-coverage N=1 cases report residual 0.0000).
2. **Non-monotonic background false-positive area, independent of recall.** `gym_3_gym_4`'s recall is
   essentially flat across every N (TP 80,478–81,421, i.e. the real change is found regardless of N),
   but its FP swings from 3,214 (N=1, far *better* than baseline's 102,591) to 912,489 (N=3) to 632,302
   (N=5) back down to 102,591 (N=10). A sparse, partial reference reconstruction can accidentally be
   *cleaner* than a fuller one (N=1 got lucky here) or introduce more depth-conflict/merge noise than
   either fewer or more views (N=3, N=5) — this is not a smooth quality gradient.

**Most important conclusion: performance does not systematically improve as more T0 reference views
become available in the 1–5 range** — small reference sets are unstable (occasional total failure,
occasional lucky success) rather than being on a gradual improving curve. The full N=10 reference set
is unambiguously the best and most reliable operating point tested, but that is a large-N-vs-small-N
effect, not evidence that "more is monotonically better" in general. Reference-reconstruction wall time
did not scale down with fewer frames (150.9–185.5s for 10 pairs; VGGT-Omega's forward pass has largely
fixed overhead here at these small N), so there is no runtime incentive to use a sparse reference set
even setting quality aside.

## Table 2: component ablations (Experiments 2–4)

| Variant | Pooled IoU | Mean IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| v10_no_dino (baseline) | 0.3010 | 0.3498 | 0.3681 | 0.6229 | 0.4627 |
| no_refine | **0.3287** | **0.3704** | **0.4071** | 0.6304 | **0.4947** |
| no_recovery | 0.1264 | 0.1450 | 0.1368 | 0.6265 | 0.2245 |
| no_appearance | 0.3029 | 0.3498 | 0.3709 | 0.6229 | 0.4650 |

TP/FP/FN and FP-by-class:

| Variant | TP | FP | FN | ADDED FP | REMOVED FP | MOVED FP | REPLACED FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 515,616 | 885,304 | 312,136 | 254,183 | 509,844 | 0 | 121,277 |
| no_refine | 521,786 | 759,873 | 305,966 | 274,894 | 369,879 | 0 | 115,100 |
| no_recovery | 518,627 | 3,273,728 | 309,125 | 1,785,841 | 1,385,116 | 0 | 102,771 |
| no_appearance | 515,616 | 874,392 | 312,136 | 254,183 | 498,904 | 0 | 121,305 |

### Experiment 2: DI2FIX refinement ablation

`configs/scenediff_v10_no_dino_no_refine.yaml` sets only `refine.enabled: false`; reconstruction,
localization and raw render are the same cached artifacts as the baseline (confirmed: stage 1-3 timings
match baseline to within noise, e.g. SAM3 stage 99.9s vs 98.6s/query). No detection thresholds were
touched. **Per-pair: 7 better without refine, 2 worse, 1 tied** (`store_39_store_40`, which has no
in-scope GT either way).

| Pair | refine (baseline) IoU | no_refine IoU | ΔFP (refine − no_refine) | ΔTP |
|---|---:|---:|---:|---:|
| P01 184214 0030→0032 | 0.0000 | 0.0019 | +54,720 | −168 |
| P01 095114 0001→0011 | 0.0000 | 0.0029 | +28,631 | −263 |
| closet_1→closet_2 | 0.4045 | 0.4158 | +374 | −4,079 |
| bedroom_28→bedroom_29 | 0.5502 | **0.6855** | +8,520 | +38 |
| store_57→store_58 | **0.6918** | 0.6442 | −9,581 | 0 |
| store_39→store_40 | 0.0000 | 0.0000 | −1,134 | 0 |
| bedroom_32→bedroom_33 | **0.5721** | 0.4573 | −34,684 | −23 |
| table_5→table_6 | 0.2009 | **0.2662** | +67,638 | −1,672 |
| gym_3→gym_4 | 0.4417 | 0.4473 | +2,276 | −3 |
| living_room_49→living_room_50 | 0.6364 | **0.7827** | +8,671 | 0 |

In every pair TP is essentially unchanged either way (≤4,079 px, mostly 0); **the entire effect of
refinement is on FP area**, and on this subset the net effect is to *add* FP more often than it removes
it (5 of 7 "hurts" pairs lose ≥8,500 FP pixels by turning refine off; `table_5_table_6` loses 67,638).

Case inspection (renders + `object_counts` + `overlay.png`/`labels_color.png` compared directly, since
raw and refined renders get *independently* re-segmented by SAM3 and are not simply replayed):

- **Refinement hurts** (`living_room_49_living_room_50`, −0.146 IoU, the largest "hurts" case): visually,
  refined and raw `render_t0` are nearly indistinguishable at the object level — DI2FIX did not
  hallucinate or remove any visible content here. What changed is **SAM3 proposal fragmentation**:
  refinement's smoothing produced **26** proposals on `render_t0` vs the raw render's **16**, seeding
  **3 extra spurious REMOVED regions** (baseline `removed=3` vs no_refine `removed=0`) that appear as
  thin slivers along a render-hole boundary (top-right curtain/wall edge) and a small mark near a media
  console — edge-fragmentation artifacts, not new false objects. TP is identical (29,527) both ways.
- **Refinement helps** (`bedroom_32_bedroom_33`, +0.115 IoU): here refinement *reduces* FP (52,523 vs
  87,207 without it) while TP is again unchanged (78,914 vs 78,937) — the raw render's own noise/holes
  seed a larger REMOVED false-positive region that refinement's smoothing suppresses.
- **Neutral** (`gym_3_gym_4`, `store_39_store_40`): object counts and changed-pixel fraction are nearly
  identical with and without refinement (e.g. gym_3_gym_4: 0.0887 vs 0.0876 changed-pixel fraction).

So refinement's effect on SAM3 segmentation is **not consistently under-segmentation** as its own code
comment assumes — it fragmented `render_t0` *more* in the "hurts" case above (16→26 objects) and
consolidated it in the "helps" case (17→15). Runtime: DI2FIX cost **~27.5s/query** (275.2s / 10 pairs)
that `no_refine` skips entirely, with every other stage's timing unchanged.

**Does DI2FIX earn its place in the final method? On this subset, no** — turning it off improved pooled
IoU, mean IoU, precision and F1, and was tied-or-better on 8/10 pairs, driven entirely by FP differences
whose direction (help vs hurt) tracks unpredictable segmentation-boundary changes rather than genuine
denoising. This is a real, controlled result on a small (10-pair) sample, not a large-scale validation;
it directly contradicts refinement's inclusion rationale and is flagged in Q6 below.

### Experiment 3: SAM2 unmatched-object recovery ablation

`configs/scenediff_v10_no_dino_no_recovery.yaml` sets only `recover_unmatched_via_tracking: false`;
direct correspondence continues using the same SAM2 evidence as baseline (only the secondary
proposal-miss recovery pass is disabled). Replayed from the baseline's stage-1-3 inventory bundle
(identical SAM3 proposals/descriptors/tracks; only stage 4+ reran) — confirmed by identical
`n_t0_objects`/`n_t1_objects`/`unmatched_T0`/`unmatched_T1` counts (124/161) between the two runs.

| | baseline | no_recovery | Δ |
|---|---:|---:|---:|
| Pooled IoU | 0.3010 | 0.1264 | **−0.1746** |
| Recall | 0.6229 | 0.6265 | +0.0036 (flat) |
| Precision | 0.3681 | 0.1368 | −0.2313 |
| FP (total) | 885,304 | 3,273,728 | **+2,388,424** |
| ADDED FP | 254,183 | 1,785,841 | +1,531,658 |
| REMOVED FP | 509,844 | 1,385,116 | +875,272 |
| Recovery events (baseline) | 65 | — | — |
| Unmatched T0 objects (both) | 124 | 124 | 0 |
| Unmatched T1 objects (both) | 161 | 161 | 0 |

**Per-pair: 1 better, 6 worse, 3 tied** (1 pixel-identical: `table_5_table_6`, where recovery apparently
found nothing to rescue). Case inspection of the baseline's 68 recorded recovery events (5 pairs have
0–1; most have 4–19) against ground truth: their recovered masks sit almost entirely on **background**
— mean GT-overlap fraction **0.076%**, median **0**, only 5/68 have *any* GT overlap at all and none
exceed 10%. In other words, essentially every object SAM2 recovery pulls back from the
removed/added bucket in the baseline genuinely was unchanged background that SAM3 merely failed to
propose in both frames — recovery is not trading away recall for precision, it is almost pure precision
gain with negligible recall cost (TP is flat, 515,616 → 518,627, actually marginally higher without
recovery because a few recovered objects' pixels don't perfectly reconstruct the same footprint as their
independent REMOVED+ADDED masks would have).

**Does SAM2 recovery earn its place in CORGI? Decisively yes.** Removing it costs 2.39M FP pixels
(2.7x the baseline's total FP) for a near-zero, slightly negative recall change, and case inspection
confirms the recovered objects are correctly identified as unchanged, not real changes being suppressed.

### Experiment 4: SAM3 appearance-feature ablation

`configs/scenediff_v10_no_dino_no_appearance.yaml` adds `use_sam_features: false` and
`enable_appearance_correspondence: false` on top of baseline's already-`use_dino_features: false` —
the repo's existing "tracking + geometry, no appearance anywhere" mechanism (identical flag set to
`configs/model_ablation_m4_tracking_geometry.yaml`). Pre-registered exactly what this changes in each
code path *before* running (`results/scenediff_diagnostic/SceneDiff/_experiments/_no_appearance_preregistration.md`):
direct identity, the reciprocal-NN fallback (removed, since it is itself appearance-derived), geometric
rescue (loses its appearance-gated identity option but keeps geometry-and-track feasibility), the clean
bridge (falls back to a bare one-way track ≥ 0.20 with no descriptor check), and tracking recovery (falls
back to track-and-area-ratio acceptance with no descriptor confirmation). Nothing was replaced with a
newly tuned matcher — SAM3 continues to generate proposals, only its *appearance descriptors* are
removed as correspondence evidence.

| | baseline | no_appearance | Δ |
|---|---:|---:|---:|
| Pooled IoU | 0.3010 | 0.3029 | +0.0019 |
| Mean IoU | 0.3498 | 0.3498 | 0.0000 (bit-identical) |
| Precision | 0.3681 | 0.3709 | +0.0028 |
| Recall | 0.6229 | 0.6229 | 0.0000 |
| Accepted identities | 83 | 83 | 0 |
| Recovery events | 65 | 66 | +1 |
| Unmatched T0 / T1 | 124 / 161 | 124 / 161 | 0 / 0 |

**Per-pair: 0 better, 0 worse, 10 tied — 9/10 pairs pixel-identical**, the 10th
(`P01-20240203-184214_0030_P01-20240203-184214_0032`, a pair with no in-scope GT either way) differs by
only 10,912 FP pixels (136,324 → 125,412) with TP unchanged at 0. No object-level decision (identity
accepted/rejected, MOVED/ADDED/REMOVED assignment) changed on any of the other 9 pairs. Consequently
**there were no representative cases of appearance preventing a false SAM2 match or vetoing a useful one
to inspect** on this subset — geometry + SAM2 transport reproduced the *same* correspondence decisions
as geometry + SAM2 transport + SAM3 appearance, essentially everywhere.

**Do SAM3 appearance features earn their place in CORGI? Not demonstrably, on this subset.** Removing
them changed nothing measurable on 9/10 pairs and a negligible amount on the 10th. This does not mean
appearance evidence is never useful (10 pairs, particular scenes, particular objects), but it earns no
observed keep here, in contrast to recovery's clearly load-bearing result above.

## Answers

**1. Does increasing T0 reference-view count help?** Going from N=10 down to any sparse subset (1, 3,
or 5) hurts substantially and consistently. Within the sparse range itself, no — the relationship is not
monotonic (N=3 is worse than N=1 despite N=3 having strictly more, nested reference frames); performance
is dominated by whether the sparse subset happens to produce a workable alignment/render for a given
scene, not by view count per se. Render coverage does increase smoothly with N, but detection accuracy
does not track it cleanly.

**2. Does DI2FIX earn its complexity?** No, not on this subset — disabling it improved pooled IoU
(+0.028), mean IoU (+0.021), precision (+0.039) and F1 (+0.032), and was tied-or-better on 8/10 pairs.
Its effect on SAM3 segmentation is unpredictable in direction (it fragmented one scene's proposals and
consolidated another's), rather than reliably cleaning up render defects.

**3. Does SAM2 recovery earn its place?** Yes, decisively. Removing it costs 2.39M FP pixels (pooled
IoU 0.301 → 0.126) for essentially zero recall change, and the baseline's 68 recovery events land on
background 92% of the time (mean GT overlap 0.076%) — it is recovering real proposal misses, not
suppressing real changes.

**4. Do SAM3 appearance features earn their place?** Not demonstrably on this subset — removing them
left 9/10 pairs pixel-identical and the 10th changed by a negligible, GT-irrelevant amount. Geometry +
SAM2 transport alone reproduced the same correspondence decisions here.

**5. Which components should remain in the final CORGI pipeline?** SAM2 unmatched-object recovery
should clearly remain (Experiment 3). On this controlled 10-pair evidence, DI2FIX refinement and SAM3
appearance-feature correspondence are both candidates for removal or reconsideration — refinement
because it measurably hurt more than it helped here, appearance because it changed nothing measurable.
Neither result is large-scale enough to justify removing either from the shipped method outright without
a bigger validation run (see Q6), but both deserve a dedicated larger-sample re-test before the next
release, prioritized over further reference-view tuning (Experiment 1 gives no actionable tuning
target — the sparse range is simply unstable).

**6. Is there any result important enough to change the paper's main story?** The DI2FIX result is the
one that matters most here: if refinement is described as improving downstream change detection by
cleaning up render defects, this controlled ablation directly contradicts that on this subset (net
negative, driven by unpredictable segmentation-fragmentation side effects, not denoising quality) and
should be flagged before the paper repeats that claim. The reference-view non-monotonicity is also worth
a caveat if the paper claims richer references "smoothly" help — the true effect size (N=10 vs sparse)
is real and large, but the smooth-improvement framing is not supported. SAM2 recovery's necessity and
appearance's near-irrelevance are consistent with, not contradictions of, the existing design rationale
documented in `change_detection.py`, so they reinforce rather than change the story.

## Reproducibility

- Baseline config: `configs/ablate_v10_no_dino.yaml`; ablation configs: `configs/scenediff_v10_no_dino_no_refine.yaml`,
  `configs/scenediff_v10_no_dino_no_recovery.yaml`, `configs/scenediff_v10_no_dino_no_appearance.yaml`
  (each a byte-diff of exactly one flag from the baseline).
- Fixed subset: `data/scenediff_benchmark/diagnostic_subset.txt`; fixed queries:
  `data/scenediff_benchmark/diagnostic_subset_queries.json`.
- Runner: `scripts/run_scenediff_diagnostic.py` (extended this run with `--reference-views N`,
  deterministic nested subsetting via `nested_reference_subset`, and `--refine-worker-dir` for a resident
  DI2FIX process); orchestrated end-to-end by `scripts/run_scenediff_overnight_ablations.sh`.
- Aggregation: `scripts/scenediff_ablation_report.py` →
  `results/scenediff_diagnostic/SceneDiff/_experiments/overnight_ablations_report.json`.
- Case inspection: `scripts/scenediff_ablation_cases.py` →
  `results/scenediff_diagnostic/SceneDiff/_experiments/<variant>/case_inspection_vs_baseline.json`.
- Per-experiment outputs: `results/scenediff_diagnostic/SceneDiff/_experiments/scenediff_v10_no_dino*/`
  (`summary.json`, `config.yaml`, `invocation.json`, `reference_sets.json`, per-pair `metrics.json` /
  `inference.json` / `decisions.json` under each pair's own experiment cell).
- Reconstruction fix required for N=1: `src/ocmask_pipeline/reconstruction.py` gained
  `_fit_single_frame_transform` (a depth-ratio + shared-pose alignment for the degenerate single-camera
  case `_fit_similarity_transform` cannot solve); flagged above as the source of the N=1 alignment-residual
  diagnostic's unreliability.
- No thresholds were changed anywhere in this document. No automatic follow-up experiments were run
  after seeing results. The 10-pair subset and T1 query frames were never changed.
