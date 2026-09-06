#!/usr/bin/env python3
"""Per-stage pixel-loss analysis: of the ground-truth changed pixels in a
PASLCD query photo, how many survive each stage of the pipeline, and why do
the rest get dropped?

Stages measured (all at the pipeline's own working resolution, i.e.
image_t1's pixel grid -- GT is resized down to it with nearest-neighbor):

  0. ground_truth              -- GT changed-pixel count (the denominator, 100%)
  1. raw_sam3_proposals         -- covered by the union of SAM3's raw
                                    automatic mask proposals on image_t1
                                    (before any filtering) -- loss here means
                                    SAM3's point grid never produced a mask
                                    touching that pixel at all
  2. post_select_object_proposals -- covered by the union of proposals
                                    surviving select_object_proposals
                                    (size/dedup filtering) -- loss here means
                                    a raw proposal existed but was filtered
                                    out (too small / too large a fraction of
                                    the frame) or absorbed as a near-duplicate
                                    of a mask that itself doesn't reach here
  3. post_part_suppression      -- covered by the union of objects surviving
                                    _suppress_feature_matched_parts, the
                                    *within-frame* pass that runs after
                                    proposal selection and drops a small
                                    object judged to be a "part" of a larger
                                    one it sits inside -- either because their
                                    pooled SAM3/DINOv2 features agree, OR
                                    merely because the small object is
                                    spatially adjacent to the larger one
                                    (touches it within a few pixels), with NO
                                    feature agreement required in that second
                                    case. This is the stage most likely to
                                    swallow a small added item sitting on an
                                    otherwise-unchanged surface.
  4. final_decision             -- covered by a pixel the full pipeline
                                    actually labeled added/moved (not
                                    unchanged) in its already-computed
                                    labels.png -- loss here means a surviving,
                                    un-suppressed object existed and was
                                    evaluated, but the identity/tracking
                                    association stage matched it (wrongly,
                                    from GT's perspective) as unchanged

Stage 4's losses are further broken down by which evidence path produced the
wrong "unchanged" call (direct_identity / clean_bridge_identity /
tracking_recovery), using inference.json's already-recorded decisions -- see
change_detection.resolve_three_image_changes's evidence field.

Reuses select_object_proposals, _build_inventory, _suppress_feature_matched_parts
and the SAM3/DINOv2 extractors exactly as change_detection.run_object_state_resolution
does; does not re-run SAM2 tracking/decision resolution -- those are read
back from the target run's already-saved inference.json + labels.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detect-dir", type=Path, required=True, help="a run_object_state_resolution output dir (has render_t0.png, target.png, labels.png, inference.json)")
    parser.add_argument("--gt-mask", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--output", type=Path, required=True, help="where to write the JSON report")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from PIL import Image

    from ocmask_pipeline.adapters.dinov2 import Dinov2FeatureExtractor
    from ocmask_pipeline.change_detection import (
        ThreeImageSettings,
        _build_inventory,
        _proposal_kwargs,
        _suppress_feature_matched_parts,
        select_object_proposals,
    )
    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.metrics import load_paslcd_gt
    from ocmask_pipeline.stages.sam3_proposals import Sam3AutomaticMaskGenerator

    config = load_config(args.config)
    settings = ThreeImageSettings.from_config(config)

    render_t0 = np.asarray(Image.open(args.detect_dir / "render_t0.png").convert("RGB"))
    clean_render = np.asarray(Image.open(args.detect_dir / "clean_render.png").convert("RGB"))
    image_t1 = np.asarray(Image.open(args.detect_dir / "target.png").convert("RGB"))
    labels = np.asarray(Image.open(args.detect_dir / "labels.png"))
    inference = json.loads((args.detect_dir / "inference.json").read_text())
    decisions = inference["decisions"]

    height, width = image_t1.shape[:2]
    gt_native = load_paslcd_gt(args.gt_mask)
    gt = cv2.resize(gt_native, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    gt_total = int(gt.sum())
    if gt_total == 0:
        raise ValueError("GT mask has no changed pixels at this resolution; nothing to analyze")

    def coverage(mask_union: np.ndarray) -> dict:
        covered = gt & mask_union
        return {
            "covered_pixels": int(covered.sum()),
            "covered_fraction_of_gt": float(covered.sum()) / gt_total,
            "lost_pixels": int((gt & ~mask_union).sum()),
        }

    def union_of(objects) -> np.ndarray:
        out = np.zeros((height, width), dtype=bool)
        for obj in objects:
            out |= np.asarray(obj.mask, dtype=bool)
        return out

    # --- Stage 1: raw SAM3 proposals ---
    sam_cfg = config["sam3_proposals"]
    generator = Sam3AutomaticMaskGenerator(sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"], **_proposal_kwargs(config))
    try:
        raw_t0, sam_t0 = generator.generate_with_feature_map(render_t0)
        raw_clean, sam_clean = generator.generate_with_feature_map(clean_render)
        raw_t1, sam_t1 = generator.generate_with_feature_map(image_t1)
    finally:
        generator.release()

    raw_union_t1 = np.zeros((height, width), dtype=bool)
    for p in raw_t1:
        raw_union_t1 |= np.asarray(p.mask, dtype=bool)
    stage1 = coverage(raw_union_t1)
    stage1["num_raw_proposals_t1"] = len(raw_t1)
    stage1["num_raw_proposals_render_t0"] = len(raw_t0)
    stage1["num_raw_proposals_clean_render"] = len(raw_clean)

    # --- Stage 2: post select_object_proposals ---
    objects_t0 = select_object_proposals(raw_t0, settings)
    objects_clean = select_object_proposals(raw_clean, settings)
    objects_t1 = select_object_proposals(raw_t1, settings)
    stage2 = coverage(union_of(objects_t1))
    stage2["num_objects_t1"] = len(objects_t1)
    stage2["num_dropped_by_selection"] = len(raw_t1) - len(objects_t1)

    # --- Stage 3: post _suppress_feature_matched_parts (needs DINOv2) ---
    dino_extractor = Dinov2FeatureExtractor(config["dinov2_features"])
    try:
        dino_t0 = dino_extractor.feature_map(render_t0)
        dino_clean = dino_extractor.feature_map(clean_render)
        dino_t1 = dino_extractor.feature_map(image_t1)
    finally:
        dino_extractor.release()

    inventory_t0 = _suppress_feature_matched_parts(_build_inventory(objects_t0, sam_t0, dino_t0, settings), settings)
    inventory_clean = _suppress_feature_matched_parts(_build_inventory(objects_clean, sam_clean, dino_clean, settings), settings)
    inventory_t1 = _suppress_feature_matched_parts(_build_inventory(objects_t1, sam_t1, dino_t1, settings), settings)
    final_objects_t1 = list(inventory_t1.objects)
    stage3 = coverage(union_of(final_objects_t1))
    stage3["num_objects_t1"] = len(final_objects_t1)
    stage3["num_dropped_by_part_suppression"] = len(objects_t1) - len(final_objects_t1)

    # --- Stage 4: final decision (already-computed labels.png) ---
    changed_mask = labels != 0
    stage4 = coverage(changed_mask)

    # Attribute stage-3->4 losses: which surviving t1 objects overlap
    # GT-changed pixels that are NOT in the final changed mask, and what
    # evidence path decided them "unchanged"?
    lost_at_stage4 = gt & union_of(final_objects_t1) & ~changed_mask
    evidence_by_t1_id = {}
    for row in decisions:
        t1_id = row.get("t1_object_id")
        if t1_id is not None:
            evidence_by_t1_id[t1_id] = row.get("evidence", row["decision"])

    # final_objects_t1's indices, after _suppress_feature_matched_parts, no
    # longer line up 1:1 with the t1_object_id the saved run assigned
    # (that numbering came from *its own* run's post-selection list, before
    # part suppression). Match by mask identity/overlap against objects_t1
    # instead, which does share select_object_proposals's own numbering
    # convention with the saved inference.json.
    attributed = {}
    for obj in final_objects_t1:
        mask = np.asarray(obj.mask, dtype=bool)
        overlap = int((mask & lost_at_stage4).sum())
        if overlap == 0:
            continue
        best_id, best_iou = None, 0.0
        for index, candidate in enumerate(objects_t1, start=1):
            candidate_mask = np.asarray(candidate.mask, dtype=bool)
            inter = np.logical_and(mask, candidate_mask).sum()
            union = np.logical_or(mask, candidate_mask).sum()
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, index
        evidence = evidence_by_t1_id.get(best_id, "unmatched_but_not_in_lost_set") if best_id else "no_matching_selected_object"
        attributed[evidence] = attributed.get(evidence, 0) + overlap

    report = {
        "detect_dir": str(args.detect_dir),
        "gt_mask": str(args.gt_mask),
        "resolution": [height, width],
        "gt_total_pixels_at_pipeline_resolution": gt_total,
        "stages": {
            "0_ground_truth": {"covered_pixels": gt_total, "covered_fraction_of_gt": 1.0, "lost_pixels": 0},
            "1_raw_sam3_proposals": stage1,
            "2_post_select_object_proposals": stage2,
            "3_post_part_suppression": stage3,
            "4_final_decision": stage4,
        },
        "stage4_loss_attribution_by_evidence": attributed,
        "decision_counts": inference.get("decision_counts"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Visual funnel: for each stage, GT pixels still covered are green,
    # GT pixels already lost by this stage are red, overlaid on image_t1.
    vis_dir = args.output.parent / "pixel_loss_visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    stage_masks = {
        "1_raw_sam3_proposals": raw_union_t1,
        "2_post_select_object_proposals": union_of(objects_t1),
        "3_post_part_suppression": union_of(final_objects_t1),
        "4_final_decision": changed_mask,
    }
    base = (0.35 * image_t1.astype(np.float32)).astype(np.uint8)
    for stage_name, mask_union in stage_masks.items():
        vis = base.copy()
        covered = gt & mask_union
        lost = gt & ~mask_union
        vis[covered] = (60, 220, 90)
        vis[lost] = (230, 50, 50)
        Image.fromarray(vis).save(vis_dir / f"{stage_name}.png")
    print(f"wrote stage visualizations to {vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
