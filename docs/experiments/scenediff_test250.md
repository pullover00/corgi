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

---

## Results

*(appended after the run; nothing above this line is edited afterwards)*
