# CORGI research diagnosis: PASLCD and SceneDiff

## Bottom line

The dominant error is not a missing model or a slightly wrong threshold. CORGI turns *failure to establish correspondence* into positive evidence of physical change. Every asymmetric proposal, imperfect track, bad render fragment, or occlusion can therefore become an `ADDED` or `REMOVED` object. PASLCD directly exposes the amplification: removing tracking-based recovery adds about **1.08 M false-positive pixels** on 30 queries, while gaining only about **8.2 k true-positive pixels**. Low-location-IoU matches are also deliberately left unconsumed and can reappear as an `ADDED`/`REMOVED` pair.

The three highest-value changes are therefore: (1) joint correspondence and object-state inference with an explicit `UNKNOWN/OCCLUDED` state, (2) a persistent 3D object memory that is projected into each query instead of rediscovering the reference inventory in a rendered image, and (3) ray/depth-based visibility and negative-evidence tests before declaring disappearance or appearance. These are changes to the problem formulation, not tuning.

## Scope and provenance

The conclusions below follow the current worktree and executed artifacts, principally `scripts/run_paslcd_scene.py`, `scripts/run_scenediff_diagnostic.py`, `src/ocmask_pipeline/reconstruction.py`, `src/ocmask_pipeline/change_detection.py`, `configs/ablate_v6_ceiling_sky_suppression.yaml`, `configs/ablate_v8_occlusion_only.yaml`, and `results/paslcd_suppression_ablation/analysis.md`.

There are two important provenance limits:

- The latest PASLCD comparison is the fixed 30-query suppression ablation. `v6` (semantic ceiling/sky removal) is the best measured result; the chronologically later, pre-fix `v8` 2D occlusion rule is a wash. Its post-mortem prompted a code correction that preserves the changed-pixel set by merging a covered removal into `ADDED`; two fast replays match `v6` in binary output, but the corrected path has not had a full benchmark rerun. The larger 375-query overnight analysis is useful for failure localization but is from an earlier configuration.
- The newest SceneDiff diagnostic completed shared reconstruction, localization, rendering, and DI²FIX for all 10 selected pairs, but produced **zero** `inference.json` files and no detection manifest. Thus it supports reconstruction/coverage findings, not a current-pipeline accuracy or model-ablation conclusion. Older complete SceneDiff results used RGB-only manifests, silently disabling geometry, visibility, and corroboration buffers, so their low IoU is qualitative evidence only, not a fair measurement of the current pipeline.

## Actual executed pipeline

1. **Reference reconstruction and query localization.** VGGT-Omega reconstructs the T0 reference set in isolation. Each unposed T1 query is then run jointly with the same T0 frames in a separate VGGT call. A similarity transform, estimated by Umeyama alignment of shared reference camera centers, maps the isolated reference scene into that query call's frame. The old cloud is splatted into the query camera as `render_t0`. A mutually depth-filtered union of old and query points produces `clean_render`. The renderer also exports world-position maps, depth confidence, binary coverage, per-pixel reference corroboration, horizon, and scene scale. DI²FIX refines the two RGB renders only; it does not repair their geometry.

2. **SAM3 proposals.** SAM3 independently generates class-agnostic masks for `render_t0`, `clean_render`, and the real `image_t1`. Proposals are component-filtered, size-filtered, greedily deduplicated, and subject to a feature/containment-based “part” suppression. Dense SAM3 backbone features are retained.

3. **SAM2 correspondence evidence.** SAM2.1 Hiera Large propagates every object mask through all six directed image pairs among T0 render, clean render, and T1. In this code path it is mask-prompted pairwise propagation, not persistent scene tracking.

4. **Appearance and direct assignment.** SAM3 and DINOv2 ViT-B/14 dense features are separately mask-pooled. A direct T0–T1 pair normally requires both appearance gates, a compatible area ratio, and either sufficient bidirectional SAM2 overlap or reciprocal-best appearance. Feasible pairs are selected one-to-one by Hungarian assignment.

5. **Geometric identity and clean bridge.** A second assignment pass considers only leftovers and can admit geometrically compatible pairs using 3D centroids/point overlap plus identity, tracking, or reciprocal evidence. It cannot revise first-pass assignments. A clean-render bridge can connect a pair through SAM2 tracks and clean-render proposal descriptors, but it fires only about 0.13 times per PASLCD query.

6. **State decision.** A matched pair with mask IoU at least the same-location criterion is `UNCHANGED`. With the active `reject_low_confidence_moved` behavior, a lower-IoU pair is recorded as `location_mismatch_rejected` but **neither object is consumed**; `MOVED` is consequently almost absent.

7. **Unmatched-object handling.** Every remaining T0 proposal becomes a `REMOVED` candidate and every remaining T1 proposal becomes an `ADDED` candidate. Coverage/visibility and semantic ceiling/sky filters may suppress candidates. Tracking recovery then pools the tracked mask against the opposite dense SAM3 and DINO maps and can relabel it `UNCHANGED`. With occlusion handling enabled, the current worktree reclassifies a majority-overlapped `REMOVED` footprint as `ADDED` and rebuilds the raster; this preserves change pixels but still infers occlusion from 2D overlap alone. Finally, a covered-region chromatic residual appends `REPLACED` components.

The batch PASLCD and new diagnostic runners pass the geometry buffers. The standalone `run_paslcd_pair.py` currently generates but does not pass those buffer flags to `detect.py`; results from that wrapper do not exercise the nominal geometry/visibility pipeline.

## What is failing, and where

On PASLCD-30, `v6` reaches mIoU **0.1727**, precision **0.3199**, and recall **0.2774**. Its 188,864 false-positive pixels are dominated by `ADDED` (102,978; 54.5%) and `REMOVED` (70,870; 37.5%); `REPLACED` contributes 15,016 (8.0%), and `MOVED` contributes none. The larger overnight run still misses 62.3% of changed pixels; 89% of missed pixels belong to objects at least 20 pixels and 96.5% occur where T0 geometry exists. This is not primarily a tiny-object or render-hole recall problem.

| Stage | Observed failure and evidence | Role in `ADDED`/`REMOVED` errors |
|---|---|---|
| Reconstruction/localization | Camera-center alignment residuals in the new 10-pair SceneDiff diagnostic are small (0.00029–0.00270), so gross pose alignment is usually not the first failure. Rendering is still incomplete or distorted: stored coverage is only about 0.489 and 0.495 for two pairs, and PASLCD Playground produces warped/confetti geometry and near-zero IoU. Five older SceneDiff pairs failed reconstruction entirely. | Bad or absent surfaces change SAM3 mask shape and appearance on only one side. The later unmatched rule turns these render artifacts into object changes. |
| Segmentation/proposals | T1 has 54.2 selected proposals/query versus 37.4 on T0 and 35.9 on clean render. In a traced PASLCD query raw proposals cover 99.4% of changed pixels, selection 98.9%, but part suppression reduces this to 84.1%. | Independent segmentation of a synthetic render and a photograph creates split/merge and proposal-count asymmetry. A proposal missing on one side directly seeds a one-sided candidate. |
| Correspondence | Tracking recovery restores about 18 unchanged objects/query. Disabling it increases FP from 227,241 to 1,303,278 while adding very little TP. The clean bridge is almost inactive. In another trace, final decisions reduce retained GT coverage from 84.1% to 49.7%, mostly by incorrectly accepting changed pixels as direct identity. | Both false non-matches and false matches are important: non-matches explode into `ADDED`/`REMOVED`; false identity suppresses true changes. Pairwise matching cannot represent split/merge proposals or uncertainty. |
| Visibility/occlusion | Disabling visibility adds roughly 256 k FP pixels for only 173 TP pixels, so visibility is essential. Yet only ~1% of residual baseline FP lies in binary render holes: remaining errors occur in *covered but wrong, uncertain, or occluded* areas. The pre-fix `v8` 2D overlap rule slightly raised precision but lowered recall; 96.3% of the pixels erased in its main regression were GT change. The current merge fix preserves those pixels but does not resolve their physical state. | Binary coverage answers “was anything rendered here?”, not “should this particular object be visible along this ray?”. 2D overlap cannot distinguish a removal, an under-segmented addition, and a genuinely occluded old object. |
| Final classification | `location_mismatch_rejected` leaves both endpoints available to the unmatched loops. Of 159 such decisions in PASLCD-30, 48 retain both a final `REMOVED` and `ADDED` object. `MOVED` is effectively zero. | The classifier systematically aliases uncertain motion/correspondence into an appearance plus disappearance, producing exactly the dominant FP classes. |

Ceiling/sky suppression improves total FP by about 38 k with no measured recall loss on PASLCD-30, confirming a structured nuisance source near the upper image band (39% of baseline FP versus 2% of GT). It is a useful guard, but it treats a symptom. A geometric horizon experiment failed catastrophically on 8/30 queries because the dominant plane was often wrong. Reference corroboration and confidence-weighted visibility also hurt difficult scenes in prior targeted tests, so simply enabling those existing gates is not supported.

## Are SAM2, SAM3 features, and DINOv2 independently useful?

**SAM2 is independently useful.** The no-recovery ablation is the strongest causal evidence in the repository: recovery is functioning mainly as a precision repair for proposal/correspondence asymmetry, not as a recall booster.

**Geometry is also independently useful.** The geometry pass can rescue descriptor-disagreement pairs, and the secondary ChangeSim sequence improved from about 18.8 to 23.0 mIoU when geometry was added. Its current use is limited, however, because it only sees leftovers and relies on noisy per-query render geometry rather than persistent objects.

**The two appearance descriptors are not proven independently useful.** Across PASLCD-30 candidate matrices, SAM3/DINO cosine scores are moderately correlated (Pearson about 0.55). Among tracked pairs, 775 pass both gates, only 21 pass SAM3 alone, 36 DINO alone, and 8 neither. For assigned/rejected pairs, 797 pass both and only 57 disagree or fail at least one. DINO consumes about 24% of runtime while SAM3 proposal/feature work consumes about 63%. There is no completed PASLCD or current SceneDiff model-set ablation measuring correctness with either descriptor removed; the planned SceneDiff `m1`–`m5` experiment never reached inference. Therefore the repository supports “possibly redundant hard vetoes,” not a defensible claim that either descriptor should already be deleted. SAM2 supplies geometric transport, whereas SAM3/DINO supply appearance; those categories are complementary, but the two appearance branches need one controlled removal test.

## Is “unmatched object → change” fundamental?

Yes. It is the central structural error amplifier. “No accepted match” is ambiguous among true addition/removal, occlusion, missed proposal, split/merge segmentation, render corruption, bad localization, moved object, and descriptor/tracker failure. The implementation nevertheless maps that ambiguity directly to `ADDED` or `REMOVED`. Recovery's million-pixel FP effect quantifies how much correctness depends on repairing non-matches before that conversion. The rejected-location path provides direct code-and-log evidence that even a known association is discarded and often emitted as two changes. A trustworthy physical state decision needs positive presence/absence and visibility evidence, not merely absence of a match.

## Three mechanistic changes

### 1. Joint correspondence and object-state hypotheses

- **Mechanism:** Replace sequential match-then-label logic with one constrained inference over hypotheses `{same/stationary, moved, added, removed, occluded/unobserved, proposal-miss}`. Allow one-to-many and many-to-one proposal groupings. Combine SAM2 transport, one appearance descriptor selected by a controlled ablation, and geometry as evidence in the same assignment. Preserve a matched low-location-IoU pair as `MOVED` or `UNKNOWN`; never release both endpoints into unmatched change buckets. Require independent visibility/presence evidence before choosing `ADDED` or `REMOVED`.
- **Failure it addresses:** False `ADDED`/`REMOVED` from correspondence uncertainty, split/merge masks, and rejected motion; false `UNCHANGED` from a locally plausible but globally inconsistent pair.
- **Evidence from current experiments:** No-recovery adds ~1.08 M FP pixels for ~8.2 k TP; 48/159 rejected-location matches become both added and removed; `MOVED` is effectively zero; direct identity also absorbs substantial true-change area.
- **Expected impact:** High precision gain in the two dominant FP classes, with better moved-object handling and less recall loss from incorrect identity.
- **Implementation effort:** Medium–high; mostly a resolver/data-model rewrite using already computed evidence, without retraining models.
- **Smallest experiment needed to test it:** Replay the cached PASLCD-30 inventories with a minimal hypothesis resolver supporting `UNCHANGED`, `MOVED`, and `UNKNOWN`, plus grouped split/merge matches; compare against `v6`, and separately score the 159 rejected-location cases.

### 2. Persistent 3D object memory with projection-driven recall

- **Mechanism:** Fuse reference-view masks, points, features, and observation counts into persistent 3D object nodes once per scene. For each query, project every node with a depth-ordered expected mask and use that projection to prompt/associate the query, even when independent SAM3 proposal generation misses or splits it. Associate query proposals to memory in 3D; make clean render an optional observation, not a third independently segmented inventory.
- **Failure it addresses:** Synthetic-render/photo proposal asymmetry, part suppression, repeated rediscovery of reference objects, proposal misses, and brittle clean-bridge matching.
- **Evidence from current experiments:** Proposal counts differ sharply across views; tracking recovery repairs ~18 objects/query; the clean bridge fires only ~0.13/query; part suppression lost 14.8 points of GT-pixel coverage in a trace; most overnight false negatives are neither tiny nor in geometry holes. Geometry already adds complementary value but is restricted to a leftover pass.
- **Expected impact:** High reduction in unmatched candidates and moderate recall improvement, especially for stable objects whose 2D masks split, merge, or disappear between render and photograph.
- **Implementation effort:** High; requires cross-reference mask fusion, persistent IDs, and projection bookkeeping, but uses the existing reconstruction, SAM3 masks, SAM2 prompts, and features.
- **Smallest experiment needed to test it:** On five cached PASLCD queries (two high-recovery, two catastrophic, one good), build memory nodes from the existing reference masks/point maps and test projection-prompted query recovery. Measure unmatched-object count, recovered GT pixels, and new FP pixels; then sanity-check projection quality on the two lowest-coverage SceneDiff diagnostic pairs.

### 3. Object-conditioned ray visibility and negative evidence

- **Mechanism:** For each reference object, compare its projected depth distribution with query depth along the same rays and classify pixels as expected-visible, occluded-by-query, out-of-view, or geometrically uncertain. Declare `REMOVED` only when enough expected-visible object support is replaced by observed free/background surface; declare `ADDED` only when query points are in front of or incompatible with a well-supported reference surface. Use depth ordering—not 2D mask overlap—to relate an added occluder to a hidden reference object, and abstain where reconstruction is unsupported.
- **Failure it addresses:** False removals behind occluders, false changes on covered-but-wrong render geometry, render-boundary hallucinations, and ambiguous `ADDED`/`REMOVED` semantics under 2D overlap.
- **Evidence from current experiments:** Visibility removal adds ~256 k FP for negligible TP, yet binary holes explain only ~1% of remaining FP. Pre-fix `v8` is slightly worse than `v6`; its current merge correction restores binary pixels in two replays but still cannot decide the physical state from overlap. New SceneDiff renders can have only ~49% coverage despite good camera-center alignment, while naive horizon, confidence, and corroboration gates have already failed on difficult scenes.
- **Expected impact:** Medium–high precision gain for `REMOVED` and boundary `ADDED`, with less recall damage than global coverage, confidence, or overlap rules.
- **Implementation effort:** Medium; the necessary world-position, depth-confidence, coverage, and scene-scale buffers already exist, but the decision must be object/ray conditioned.
- **Smallest experiment needed to test it:** Hand-label the physical state of the 27 overlap cases from the 16 PASLCD queries where `v8` fired, then replay those cached position buffers with the depth-order hypotheses. Measure state-decision accuracy and binary FP/FN against `v6`; no sweep is needed.
