#!/usr/bin/env python3
"""Build per-object correspondence crops for the pixel-loss artifact: for
each decision in an already-completed run's inference.json, crop the same
image region from both frames being compared, with that decision's mask(s)
highlighted, plus its recorded feature-similarity and tracking evidence.

Reuses select_object_proposals / _build_inventory / _suppress_feature_matched_parts
exactly as change_detection.run_object_state_resolution does, to regenerate
the same final object masks (same IDs) the saved inference.json refers to --
does not re-run SAM2 tracking (that evidence is read back from inference.json,
which already recorded track_iou/sam_cosine/dino_cosine per decision).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--margin", type=int, default=24)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image, ImageDraw

    from ocmask_pipeline.adapters.dinov2 import Dinov2FeatureExtractor
    from ocmask_pipeline.change_detection import (
        ThreeImageSettings,
        _build_inventory,
        _proposal_kwargs,
        _suppress_feature_matched_parts,
        select_object_proposals,
    )
    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.stages.sam3_proposals import Sam3AutomaticMaskGenerator

    config = load_config(args.config)
    settings = ThreeImageSettings.from_config(config)

    render_t0 = np.asarray(Image.open(args.detect_dir / "render_t0.png").convert("RGB"))
    clean_render = np.asarray(Image.open(args.detect_dir / "clean_render.png").convert("RGB"))
    image_t1 = np.asarray(Image.open(args.detect_dir / "target.png").convert("RGB"))
    inference = json.loads((args.detect_dir / "inference.json").read_text())
    decisions = inference["decisions"]

    sam_cfg = config["sam3_proposals"]
    generator = Sam3AutomaticMaskGenerator(sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"], **_proposal_kwargs(config))
    try:
        raw_t0, sam_t0 = generator.generate_with_feature_map(render_t0)
        raw_clean, sam_clean = generator.generate_with_feature_map(clean_render)
        raw_t1, sam_t1 = generator.generate_with_feature_map(image_t1)
    finally:
        generator.release()

    dino_extractor = Dinov2FeatureExtractor(config["dinov2_features"])
    try:
        dino_t0 = dino_extractor.feature_map(render_t0)
        dino_clean = dino_extractor.feature_map(clean_render)
        dino_t1 = dino_extractor.feature_map(image_t1)
    finally:
        dino_extractor.release()

    objects_t0 = select_object_proposals(raw_t0, settings)
    objects_clean = select_object_proposals(raw_clean, settings)
    objects_t1 = select_object_proposals(raw_t1, settings)
    inventory_t0 = _suppress_feature_matched_parts(_build_inventory(objects_t0, sam_t0, dino_t0, settings), settings)
    inventory_clean = _suppress_feature_matched_parts(_build_inventory(objects_clean, sam_clean, dino_clean, settings), settings)
    inventory_t1 = _suppress_feature_matched_parts(_build_inventory(objects_t1, sam_t1, dino_t1, settings), settings)
    final_t0 = list(inventory_t0.objects)
    final_t1 = list(inventory_t1.objects)

    # Sanity check: recompute spatial IoU for a matched pair and compare
    # against inference.json's own recorded value, to confirm this rerun's
    # object numbering lines up with the saved run's.
    def mask_iou(a, b):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union else 0.0

    checks = []
    for row in decisions:
        if row.get("t0_object_id") and row.get("t1_object_id") and "spatial_iou" in row:
            t0_id, t1_id = row["t0_object_id"] - 1, row["t1_object_id"] - 1
            if t0_id < len(final_t0) and t1_id < len(final_t1):
                recomputed = mask_iou(final_t0[t0_id].mask, final_t1[t1_id].mask)
                checks.append(abs(recomputed - row["spatial_iou"]) < 1e-3)
    match_rate = sum(checks) / len(checks) if checks else 0.0
    print(f"ID-alignment sanity check: {sum(checks)}/{len(checks)} matched ({match_rate:.0%})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def bbox(mask, margin, shape):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None
        h, w = shape
        x0, y0, x1, y1 = xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    def union_bbox(*boxes):
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        return x0, y0, x1, y1

    def crop_with_mask(image, mask, crop_box, tint):
        img = Image.fromarray(image).convert("RGB")
        overlay = np.asarray(img).astype(np.float32).copy()
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            overlay[m] = overlay[m] * 0.45 + np.array(tint, dtype=np.float32) * 0.55
        result = Image.fromarray(overlay.astype(np.uint8))
        if mask is not None:
            draw = ImageDraw.Draw(result)
            ys, xs = np.nonzero(mask)
            if len(xs):
                draw.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=tuple(tint), width=1)
        return result.crop(crop_box) if crop_box else result

    GREEN = (60, 220, 90)
    RED = (230, 60, 60)
    BLUE = (70, 130, 240)

    saved = []
    for row_index, row in enumerate(decisions):
        decision = row["decision"]
        t0_id = row.get("t0_object_id")
        t1_id = row.get("t1_object_id")
        t0_mask = final_t0[t0_id - 1].mask if t0_id else None
        t1_mask = final_t1[t1_id - 1].mask if t1_id else None

        box0 = bbox(t0_mask, args.margin, render_t0.shape[:2]) if t0_mask is not None else None
        box1 = bbox(t1_mask, args.margin, image_t1.shape[:2]) if t1_mask is not None else None
        shared_box = union_bbox(box0, box1)

        tint = GREEN if decision == "unchanged" else BLUE if decision == "moved" else RED
        left = crop_with_mask(render_t0, t0_mask, shared_box, tint)
        right = crop_with_mask(image_t1, t1_mask, shared_box, tint)

        name = f"{row_index:03d}_{decision}_t0-{t0_id}_t1-{t1_id}"
        left.save(args.output_dir / f"{name}_source.png")
        right.save(args.output_dir / f"{name}_target.png")
        saved.append({**row, "crop_source": f"{name}_source.png", "crop_target": f"{name}_target.png"})

    (args.output_dir / "gallery_manifest.json").write_text(json.dumps(saved, indent=2))
    print(f"wrote {len(saved)} correspondence pairs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
