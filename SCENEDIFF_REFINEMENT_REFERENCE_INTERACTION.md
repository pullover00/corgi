# Does DI2FIX refinement compensate for sparse reference reconstructions?

A 2 × 4 paired factorial on the fixed SceneDiff diagnostic subset:
reference views N ∈ {1, 3, 5, 10} × render refinement ∈ {ON, OFF}.

**Convention used throughout: ΔIoU = IoU(refine OFF) − IoU(refine ON).**
A **positive** ΔIoU means the raw render scored higher, i.e. **refinement hurt**.
ΔFP = FP(OFF) − FP(ON), so a negative ΔFP means refinement produced *more* false positives.

## 1. Protocol

Five cells already existed (N=1/3/5/10 with refine ON, and N=10 with refine OFF). Only the
three missing cells were run: `scenediff_v10_no_dino_ref1_no_refine`,
`scenediff_v10_no_dino_ref3_no_refine`, `scenediff_v10_no_dino_ref5_no_refine`.

Held fixed across every cell: the preregistered 10-pair subset
(`data/scenediff_benchmark/diagnostic_subset.txt`), the single fixed T1 query per pair
(`data/scenediff_benchmark/diagnostic_subset_queries.json`), DINOv2 disabled, all thresholds,
SAM3 proposal parameters, SAM2 tracking/recovery, geometric identity, visibility/occlusion
filtering, and replacement detection. No multi-query fusion. No post-result modification.

### Pre-run verification (all four checks passed before any cell was launched)

1. **Config diff.** A parsed-YAML comparison of `configs/scenediff_v10_no_dino_no_refine.yaml`
   against `configs/ablate_v10_no_dino.yaml` (the config every ON cell used) showed exactly one
   semantic difference: `refine.enabled: True -> False`. All three ON runs were confirmed to
   have used `configs/ablate_v10_no_dino.yaml`.
2. **Reference frames reused, not resampled.** The reference-view count is a *runner* flag
   (`--reference-views N`), not a config field, so each OFF cell resolves to the same
   deterministic nested subset as its ON partner and reads the same stored `reference_sets.json`.
   Spot-checked on `store_57_store_58`: N=1 `[189]`, N=3 `[0, 189, 340]`,
   N=5 `[0, 76, 189, 264, 340]` — identical to the existing experiment.
3. **Shared upstream artifacts.** ON and OFF at a given N read the same `shared_ref{N}`
   reconstruction/localization/render cell. 30/30 `reference_reconstruction` and 30/30 `render`
   cells were cache hits — no VGGT-Omega recomputation, so the reconstruction inputs are
   literally the same files. This is what makes the comparison paired.
4. **Detection recomputed from raw renders.** `ArtifactStore.stale_reason` confirmed the OFF
   config makes the `refine` stage stale (hash `81790302560268c0 → e75f812bf69aae3c`), which
   cascades `labels ← resolution ← descriptors ← proposals ← refine`. No SAM3 proposal or
   inference output generated from a DI2FIX-refined render is reused in an OFF cell.

### N=10 control check

The brief's expected values reproduce exactly from the stored summaries:

| | IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| N=10 refine ON — expected | 0.3010 | 0.3681 | 0.6229 | 0.4627 |
| N=10 refine ON — loaded | 0.3010 | 0.3681 | 0.6229 | 0.4627 |
| N=10 refine OFF — expected | 0.3287 | 0.4071 | 0.6304 | 0.4947 |
| N=10 refine OFF — loaded | 0.3287 | 0.4071 | 0.6304 | 0.4947 |

All agree to < 5e-5. Per-pair render coverage is identical between ON and OFF at every N
(control check in the generated tables), as it must be — DI2FIX rewrites RGB only.

### Run provenance (disclosure)

The `ref5_no_refine` cell was interrupted after 4/10 pairs by a machine-wide memory
exhaustion at 2026-09-10 15:13 (`systemd-oomd` killed `org.gnome.Shell@x11.service`; the kernel
OOM-killer took Chrome; `user@1000.service` exited SIGKILL). The launcher shell was collateral
damage, not a pipeline fault — the log ends cleanly with no traceback. The interrupted pair had
written an empty `labels/` directory with zero files and no manifest; it was removed, and the
cell was re-run to completion with the identical config and flags. Reconstruction/localization
stages came back as cache hits; all 10 pairs' detection stages were recomputed. N=1 and N=3
were not touched. No other cell was interrupted.

## 2. Main 2 × 4 comparison

| N | Coverage | IoU refine ON | IoU refine OFF | ΔIoU OFF−ON | F1 ON | F1 OFF | ΔFP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.430 | 0.1516 | **0.1589** | +0.0073 | 0.2633 | **0.2742** | −106,112 |
| 3 | 0.568 | **0.0929** | 0.0678 | −0.0251 | **0.1700** | 0.1270 | −95,367 |
| 5 | 0.730 | **0.1461** | 0.1433 | −0.0028 | **0.2549** | 0.2506 | +127,754 |
| 10 | 0.755 | 0.3010 | **0.3287** | +0.0277 | 0.4627 | **0.4947** | −125,431 |

By pooled IoU the sign of the refinement effect is **not stable**: refinement loses at N=1,
wins at N=3, essentially ties at N=5, and loses at N=10.

### Full per-cell metrics

| N | refine | pooled IoU | mean IoU | P | R | F1 | TP | FP | FN | coverage | n eval | zero-cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ON | 0.1516 | 0.1865 | 0.2221 | 0.3231 | 0.2633 | 267,463 | 936,538 | 560,289 | 0.430 | 10/10 | 3 |
| 1 | OFF | 0.1589 | 0.1577 | 0.2408 | 0.3183 | 0.2742 | 263,458 | 830,426 | 564,294 | 0.430 | 10/10 | 3 |
| 3 | ON | 0.0929 | 0.1118 | 0.1046 | 0.4549 | 0.1700 | 376,563 | 3,225,131 | 451,189 | 0.568 | 10/10 | 1 |
| 3 | OFF | 0.0678 | 0.0869 | 0.0790 | 0.3241 | 0.1270 | 268,307 | 3,129,764 | 559,445 | 0.568 | 10/10 | 1 |
| 5 | ON | 0.1461 | 0.1820 | 0.1665 | 0.5432 | 0.2549 | 449,660 | 2,250,755 | 378,092 | 0.730 | 10/10 | 0 |
| 5 | OFF | 0.1433 | 0.1760 | 0.1619 | 0.5550 | 0.2506 | 459,383 | 2,378,509 | 368,369 | 0.730 | 10/10 | 0 |
| 10 | ON | 0.3010 | 0.3498 | 0.3681 | 0.6229 | 0.4627 | 515,616 | 885,304 | 312,136 | 0.755 | 10/10 | 0 |
| 10 | OFF | 0.3287 | 0.3704 | 0.4071 | 0.6304 | 0.4947 | 521,786 | 759,873 | 305,966 | 0.755 | 10/10 | 0 |

All 8 cells evaluated 10/10 pairs with zero reconstruction or evaluation failures.

**Pooled and mean per-pair IoU disagree at N=1.** Pooled favours OFF (+0.0073) while mean
per-pair favours ON (0.1865 vs 0.1577). Pooled IoU is pixel-area weighted, so it is dominated by
a few large-area scenes; the mean weights each scene equally. Both numbers are reported
everywhere below rather than choosing one.

## 3. Per-pair results

### Per-pair IoU (ON → OFF, Δ = OFF − ON)

| Pair | N=1 ON | N=1 OFF | N=1 Δ | N=3 ON | N=3 OFF | N=3 Δ | N=5 ON | N=5 OFF | N=5 Δ | N=10 ON | N=10 OFF | N=10 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P01 184214 0030-0032 | 0.0012 | 0.0000 | −0.0012 | 0.0002 | 0.0000 | −0.0002 | 0.0023 | 0.0000 | −0.0023 | 0.0000 | 0.0019 | +0.0019 |
| P01 095114 0001-0011 | 0.0039 | 0.0000 | −0.0039 | 0.0005 | 0.0015 | +0.0009 | 0.0005 | 0.0232 | +0.0227 | 0.0000 | 0.0029 | +0.0029 |
| closet_1-closet_2 | 0.0000 | 0.0000 | +0.0000 | 0.2691 | 0.0246 | **−0.2445** | 0.4020 | 0.3823 | −0.0198 | 0.4045 | 0.4158 | +0.0113 |
| bedroom_28-bedroom_29 | 0.4837 | 0.2784 | **−0.2053** | 0.0003 | 0.0000 | −0.0003 | 0.0956 | 0.0989 | +0.0034 | 0.5502 | 0.6855 | +0.1353 |
| store_57-store_58 | 0.2257 | 0.2058 | −0.0199 | 0.2229 | 0.2220 | −0.0009 | 0.2773 | 0.2744 | −0.0029 | 0.6918 | 0.6442 | −0.0476 |
| store_39-store_40 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| bedroom_32-bedroom_33 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.1915 | 0.2692 | +0.0776 | 0.5721 | 0.4573 | **−0.1148** |
| table_5-table_6 | 0.1417 | 0.1303 | −0.0115 | 0.1302 | 0.1356 | +0.0055 | 0.2216 | 0.1241 | **−0.0974** | 0.2009 | 0.2662 | +0.0653 |
| gym_3-gym_4 | 0.9474 | 0.8077 | −0.1398 | 0.0815 | 0.0819 | +0.0005 | 0.1132 | 0.1247 | +0.0115 | 0.4417 | 0.4473 | +0.0055 |
| living_room_49-living_room_50 | 0.0611 | 0.1544 | +0.0933 | 0.4133 | 0.4033 | −0.0100 | 0.5158 | 0.4633 | −0.0525 | 0.6364 | 0.7827 | **+0.1463** |

### Per-pair render coverage by N (identical for ON and OFF)

| Pair | N=1 | N=3 | N=5 | N=10 |
|---|---:|---:|---:|---:|
| P01 184214 0030-0032 | 0.351 | 0.458 | 0.484 | 0.489 |
| P01 095114 0001-0011 | 0.903 | 0.765 | 0.878 | 0.935 |
| closet_1-closet_2 | 0.000 | 0.446 | 0.527 | 0.495 |
| bedroom_28-bedroom_29 | 0.794 | 0.438 | 0.959 | 0.974 |
| store_57-store_58 | 0.436 | 0.695 | 0.716 | 0.729 |
| store_39-store_40 | 0.000 | 0.640 | 0.592 | 0.606 |
| bedroom_32-bedroom_33 | 0.000 | 0.000 | 0.701 | 0.793 |
| table_5-table_6 | 0.499 | 0.820 | 0.883 | 0.907 |
| gym_3-gym_4 | 0.780 | 0.778 | 0.888 | 0.937 |
| living_room_49-living_room_50 | 0.536 | 0.642 | 0.672 | 0.686 |

Coverage is **not monotone in N** per pair, because the nested construction guarantees frame
nesting, not coverage nesting: `bedroom_28-bedroom_29` drops from 0.794 (N=1) to 0.438 (N=3)
before recovering to 0.974 (N=10), and `store_39-store_40` goes 0.000 → 0.640 → 0.592 → 0.606.
This is why the coverage analysis in §6 is done against measured per-cell coverage rather than
against N.

### Pairs that carry no usable signal

Three of the ten pairs contribute essentially nothing at any N, in both arms:

- **`store_39-store_40`** has **TP = 0 and FN = 0 in every cell** — the T1-space ground truth
  contains no positive pixels for this pair (its changes are removed-only, out of scope for
  T1-space evaluation). It can therefore only ever contribute false positives to the pooled
  metric: 1.20M (N=3), 0.73M (N=5), 0.23M (N=10). Its ΔIoU is exactly 0.0000 everywhere.
- **The two P01 pairs** sit below IoU 0.024 in all 8 cells with FP between 23k and 157k.

Effectively 7 of 10 pairs carry signal, and the pooled statistic absorbs a large FP mass from a
pair that cannot score. This is a property of the fixed preregistered subset, identical in both
arms, so it does not bias the paired comparison — but it does mean pooled IoU is a noisy
headline number here.

## 4. False positives by predicted class

| N | refine | ADDED FP | REMOVED FP | MOVED FP | REPLACED FP | total FP |
|---|---|---:|---:|---:|---:|---:|
| 1 | ON | 387,004 | 498,728 | 0 | 50,806 | 936,538 |
| 1 | OFF | 435,066 | 339,322 | 0 | 56,038 | 830,426 |
| 3 | ON | 2,098,961 | 1,049,037 | 0 | 77,133 | 3,225,131 |
| 3 | OFF | 2,009,929 | 1,039,794 | 0 | 80,041 | 3,129,764 |
| 5 | ON | 1,378,252 | 772,660 | 0 | 99,843 | 2,250,755 |
| 5 | OFF | 1,396,097 | 884,244 | 0 | 98,168 | 2,378,509 |
| 10 | ON | 254,183 | 509,844 | 0 | 121,277 | 885,304 |
| 10 | OFF | 274,894 | 369,879 | 0 | 115,100 | 759,873 |

The most consistent single effect in the whole factorial: **refinement increases REMOVED false
positives** at N=1 (+159,406), N=5 (−111,584 — the exception), and N=10 (+139,965). At N=1 and
N=10 refinement simultaneously *reduces* ADDED FP (−48,062 and −20,711). MOVED FP is 0 in every
cell, as expected for this configuration.

Note the N=3 column is a degenerate regime in both arms: ~3.1–3.2M FP against ~0.8–0.9M at N=1
and N=10. This is the `gym_3-gym_4` background blow-up already documented in
`SCENEDIFF_OVERNIGHT_ABLATIONS.md`, and it dominates the pooled N=3 statistic regardless of
refinement.

## 5. Interaction analysis

### Paired per-N statistics (10 per-pair deltas within each N)

| N | helped (Δ<0) | hurt (Δ>0) | tied | mean ΔIoU | median ΔIoU | Wilcoxon p (non-tied) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 3 | −0.0288 | −0.0025 | 0.156 |
| 3 | 5 | 3 | 2 | −0.0249 | −0.0001 | 0.641 |
| 5 | 5 | 4 | 1 | −0.0060 | −0.0011 | 0.910 |
| 10 | 2 | 7 | 1 | +0.0206 | +0.0042 | 0.250 |

Trend of mean ΔIoU against N: **r = +0.988** (refinement becomes more harmful as the reference
grows). **No individual N reaches significance** (all p ≥ 0.156, n = 10 pairs with 1–3 ties).

### Render-side SAM3 proposal counts on `render_t0`

DI2FIX rewrites only the RGB of the rendered T0 image, so the `render_t0` proposal count is the
most direct measurement of what refinement changed downstream.

| N | proposals ON | proposals OFF | Δ | REMOVED ON | REMOVED OFF | ADDED ON | ADDED OFF |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 122 | 87 | **+35** | 25 | 19 | 28 | 30 |
| 3 | 174 | 162 | **+12** | 67 | 69 | 99 | 107 |
| 5 | 215 | 168 | **+47** | 73 | 66 | 83 | 86 |
| 10 | 207 | 167 | **+40** | 65 | 49 | 47 | 53 |

Refinement **increases** the number of SAM3 proposals on the render side at every N, and the
size of that increase tracks the harm: Pearson r(extra render-side proposals, ΔIoU) = **+0.415**
(n = 36 non-zero-coverage cells).

### Answers to the six questions

**1. Is refinement more helpful at low N than at N=10?**
Yes, on the per-pair statistics. Mean ΔIoU moves monotonically from −0.0288 (N=1) through
−0.0249 (N=3) and −0.0060 (N=5) to +0.0206 (N=10), r = +0.988 against N, and the help/hurt
count inverts from 6–1 at N=1 to 2–7 at N=10. On pooled IoU the ordering does not hold: N=1
favours OFF by 0.0073. So the answer depends on the metric, and no per-N difference is
statistically significant at n = 10.

**2. Does the sign of the refinement effect change?**
Yes, on both statistics, but not in the same places. Pooled IoU: OFF better at N=1 and N=10,
ON better at N=3 and N=5 — an unstable, non-monotone sign. Mean per-pair IoU: ON better at
N=1/3/5, OFF better at N=10 — an orderly single flip between N=5 and N=10. The two metrics
agree that refinement is harmful at N=10 and disagree at N=1.

**3. Is any improvement driven by TP/recall, or by fewer FP, REMOVED FP, or ADDED FP?**
Where refinement wins, it wins on **recall**, not on precision:
- N=3 (its clearest pooled win): TP 376,563 vs 268,307 (+108,256), recall 0.4549 vs 0.3241,
  while FP simultaneously *rose* by 95,367. Purely a recall effect.
- N=5 (near-tie): the one cell where refinement reduces FP (−127,754, mostly −111,584 REMOVED
  FP), at the cost of 9,723 TP.
- N=1 and N=10 (losses): refinement adds FP (+106,112 and +125,431), dominated by REMOVED FP
  (+159,406 and +139,965), while changing TP by only +4,005 and −6,170.
There is no cell in which refinement both raises TP and lowers FP.

**4. Does refinement help only when the raw render is incomplete but localization is valid?**
Partially supported, and only as a tendency. Binned by measured coverage (§6), refinement helps
on average in the mid-coverage bands 0.4–0.6 (−0.0175, 7 helped / 3 hurt) and 0.6–0.8 (−0.0249,
8 helped / 4 hurt), and hurts in the near-complete band 0.8–1.0 (+0.0151, 2 helped / 8 hurt).
That is the direction the hypothesis predicts — refinement is most useful when the render has
holes to smooth and least useful when it is already complete. But the effect is weak, the bins
are small, and it does not survive as a linear correlation (§6).

**5. What happens in zero-coverage / failed-localization cases?**
There are 4 such (N, pair) cells: `closet_1-closet_2`, `store_39-store_40` and
`bedroom_32-bedroom_33` at N=1 (alignment residual 0.00000 — the N=1 residual is trivially ≈0
by construction and is not a trustworthy diagnostic), and `bedroom_32-bedroom_33` at N=3
(residual 0.01555). In **all 4**, ON and OFF are identical: IoU 0.0000, TP 0, FP 0,
ΔIoU exactly +0.0000. This confirms explicitly that **refinement cannot recover these cases**.
DI2FIX only rewrites RGB; where no valid render exists there is nothing for it to modify, and
it cannot restore missing geometry or repair a failed localization.

**6. Is refinement benefit correlated with render coverage?**
Essentially not, as a linear relationship. Across all 40 N × pair cells,
Pearson r(coverage, ΔIoU) = **+0.032**; excluding the 4 zero-coverage cells, r = **+0.107**
(n = 36). Both are negligible, and the weak positive sign runs *opposite* to the compensation
hypothesis (higher coverage → refinement slightly more harmful). The binned view in §6 shows the
underlying relationship is non-linear, which is why the correlation is near zero. With 10 scene
pairs this correlation should not be overinterpreted in either direction.

## 6. Coverage-vs-refinement analysis

| coverage bin | n cells | mean ΔIoU (OFF−ON) | refinement helped (Δ<0) | hurt (Δ>0) | tied |
|---|---:|---:|---:|---:|---:|
| 0 (failed render) | 4 | +0.0000 | 0 | 0 | 4 |
| 0–0.4 | 1 | −0.0012 | 1 | 0 | 0 |
| 0.4–0.6 | 11 | −0.0175 | 7 | 3 | 1 |
| 0.6–0.8 | 14 | −0.0249 | 8 | 4 | 2 |
| 0.8–1.0 | 10 | +0.0151 | 2 | 8 | 0 |

Pearson r(coverage, ΔIoU) = +0.032 over all 40 cells; +0.107 over the 36 non-zero-coverage
cells.

The shape is: **exactly zero effect where there is no render, a mild positive effect in the
0.4–0.8 mid-coverage band, and a mild negative effect where coverage is already high.** That is
qualitatively consistent with "refinement has something to fix only when the render is
incomplete," but the magnitudes are small (|mean ΔIoU| ≤ 0.025), the bins hold 1–14 cells, and
the linear correlation is ~0. Ten scene pairs cannot support a stronger claim than a tendency.

## 7. Qualitative strongest-help / strongest-hurt cases

Panels (`raw render_t0 | DI2FIX-refined render_t0 | labels OFF | labels ON`) are in
`results/scenediff_diagnostic/SceneDiff/_experiments/_interaction_cases/`.

| N | case | ΔIoU | coverage | `render_t0` proposals ON→OFF | what the saved outputs show |
|---|---|---:|---:|---|---|
| 1 | **help** `bedroom_28-bedroom_29` | −0.2053 | 0.794 | 10 → 9 | Refinement raises *clean_render* objects 6 → 16 while leaving render-side proposals flat; decisions shift removed 5/replaced 4 (ON) vs removed 4/replaced 5 (OFF). Proposal consolidation on the clean-render side. |
| 1 | **hurt** `living_room_49-living_room_50` | +0.0933 | 0.536 | 33 → 24 | Refinement adds 9 render-side proposals and doubles REMOVED decisions (11 vs 5) with identical ADDED (13). Proposal fragmentation on the render side. |
| 3 | **help** `closet_1-closet_2` | −0.2445 | 0.446 | 6 → 8 | Refinement *reduces* render-side proposals 8 → 6 and halves both ADDED (4 vs 6) and REMOVED (2 vs 4). The single largest help in the factorial. |
| 3 | **hurt** `table_5-table_6` | +0.0055 | 0.820 | 11 → 12 | Negligible: one proposal difference, decisions nearly identical. Effectively no meaningful downstream change. |
| 5 | **help** `table_5-table_6` | −0.0974 | 0.883 | 9 → 13 | Refinement consolidates 13 → 9 render-side proposals; REPLACED rises 11 → 19 and ADDED/REMOVED both fall (3/3 vs 5/6). |
| 5 | **hurt** `bedroom_32-bedroom_33` | +0.0776 | 0.701 | 19 → 8 | Refinement more than doubles render-side proposals (8 → 19), pushing ADDED 4 → 7 and visibility-filter rejections 10 → 16. |
| 10 | **help** `bedroom_32-bedroom_33` | −0.1148 | 0.793 | 17 → 15 | Refinement adds 2 render-side proposals but quadruples tracking recoveries (4 vs 1); REMOVED falls 5 → 4. |
| 10 | **hurt** `living_room_49-living_room_50` | +0.1463 | 0.686 | 26 → 16 | The clearest harm case: refinement adds 10 render-side proposals and creates 3 REMOVED decisions where the raw render produces 0. |

The recurring mechanism across all eight cases is the **render-side proposal count**. Where
refinement *consolidates* proposals (`closet_1` at N=3, `table_5` at N=5) it helps; where it
*fragments* or inflates them (`living_room_49` at N=1 and N=10, `bedroom_32` at N=5) it produces
spurious REMOVED/ADDED objects and hurts. Aggregated over all 36 non-zero-coverage cells this
holds as a positive correlation (r = +0.415) but not as a rule — refinement inflates proposal
counts on net at *every* N (+35/+12/+47/+40), including the N values where it helps on average.

I have deliberately not attributed these differences to hallucinated content, since the saved
outputs record proposal counts, decision counts and masks but do not contain a measurement that
would distinguish hallucinated appearance from ordinary smoothing.

## 8. Can DI2FIX compensate for sparse reference coverage?

**Not reliably.**

The evidence *for* a compensation effect is real but weak: mean per-pair ΔIoU is ordered almost
perfectly against N (r = +0.988), the help/hurt count inverts from 6–1 at N=1 to 2–7 at N=10,
and refinement helps on average in exactly the mid-coverage band where an incomplete render
gives it something to fix.

The evidence *against* relying on it is stronger:

- The effect is **not significant at any N** (all Wilcoxon p ≥ 0.156, n = 10).
- The **sign is metric-dependent**: pooled IoU and mean per-pair IoU disagree at N=1, the
  sparsest condition — the very cell the hypothesis is about.
- The pooled sign is **non-monotone** across N (+, −, −, +).
- Its one clear pooled win, N=3, is **a single scene**: removing `closet_1-closet_2`
  (Δ = −0.2445) leaves an N=3 column where the largest remaining delta in either direction is
  +0.0055. That win also occurs inside a degenerate ~3.2M-FP regime present in both arms.
- Where it wins it wins on **recall while adding false positives**; it never both raises TP and
  lowers FP.
- It **cannot** address the actual failure mode of sparse references. The three worst sparse
  cells are total localization failures with zero coverage, and there ΔIoU is exactly 0.0000.
  Sparse references fail by losing geometry and localization, which is precisely what an
  RGB-space refiner cannot restore.

So refinement's relative standing does improve as the reference gets sparser, but it does not
convert into a dependable gain, and it does not touch the mechanism by which sparse references
actually fail.

## 9. Conclusion and recommendation for the paper

**Conclusion B — refinement helps only isolated cases and has unstable sign; it does not
reliably compensate for sparse reconstruction.**

Conclusion A is not supportable: "consistently" fails on significance, on metric disagreement at
N=1, on the non-monotone pooled sign, and on the fact that the N=3 win is one scene. Conclusion C
is slightly too strong as stated: the effect is not uniformly neutral-or-harmful across N — there
is an orderly directional trend (r = +0.988) that is genuine even though no cell reaches
significance, and refinement is mildly favourable on per-pair means at N=1/3/5.

For the paper, the defensible claims are:

1. DI2FIX is **net-negative at the operating point that matters** (N=10, the configuration CORGI
   actually ships): −0.0277 pooled IoU, −0.0320 F1, +125,431 FP, hurting 7 of 10 pairs. This
   reproduces the finding already in `SCENEDIFF_OVERNIGHT_ABLATIONS.md`.
2. Its relative standing **improves monotonically as the reference becomes sparser**, but never
   becomes a reliable gain, and it is exactly zero where sparse references genuinely fail
   (zero-coverage localization failures).
3. The measurable downstream effect of refinement is on **SAM3 render-side proposal counts**,
   which it inflates at every N; harm tracks the size of that inflation (r = +0.415).

That supports reporting the refinement ablation as a negative result with an honest note that the
sparse-reference case is directionally different but not significant — rather than as a
conditional-use recommendation.

**Per the brief, no pipeline change has been made and no follow-up tuning was run.** This
document reports evidence only.

## 10. Artifacts

- Machine-readable summary (every value cited above):
  `results/scenediff_diagnostic/SceneDiff/_experiments/refinement_reference_interaction.json`
- Generated tables: `.../_experiments/refinement_reference_interaction_tables.md`
- Qualitative panels: `.../_experiments/_interaction_cases/N{1,3,5,10}_<pair>.png`
- Analysis scripts: `scripts/scenediff_refine_interaction_report.py`,
  `scripts/scenediff_refine_interaction_cases.py`
- Launchers: `scripts/run_scenediff_refine_reference_interaction.sh`,
  `scripts/resume_scenediff_ref5_no_refine.sh`
- Config (OFF arm): `configs/scenediff_v10_no_dino_no_refine.yaml`;
  config (ON arm): `configs/ablate_v10_no_dino.yaml`
- Per-cell summaries: `.../_experiments/scenediff_v10_no_dino_ref{1,3,5}[_no_refine]/summary.json`,
  `.../_experiments/scenediff_v10_no_dino[_no_refine]/summary.json`
- Run logs: `.../_experiments/_interaction_logs/`, `.../_experiments/_interaction_launcher.log`
