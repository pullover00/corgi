#!/usr/bin/env python3
"""Evaluate one SceneDiff diagnostic query and write metrics.json.

Split out of run_scenediff_diagnostic.py because scenediff_gt_eval needs
pycocotools, which the vggt-omega env the runner lives in does not have;
the runner shells this out to EVAL_CONDA_ENV instead.

Pixel IoU exactly as scripts/scenediff_pxim_iou.py defines it -- any
changed class vs any in-scope GT object, in image_t1 space -- plus the
per-class GT-bucket breakdown and the correspondence-stage counters the
stage-wise diagnosis needs, read straight from inference.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", type=Path, required=True)
    ap.add_argument("--t1-frame-idx", type=int, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True, help="detect output dir with labels.png + inference.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import cv2
    from scenediff_gt_eval import compare_to_prediction, load_query_gt

    gt = load_query_gt(args.pair_dir, args.t1_frame_idx)
    labels_png = args.labels_dir / "labels.png"
    result = compare_to_prediction(gt, labels_png)
    pred = cv2.imread(str(labels_png), cv2.IMREAD_GRAYSCALE)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_mask, gt_mask = pred != 0, gt.label != 0
    tp = int((pred_mask & gt_mask).sum()); fp = int((pred_mask & ~gt_mask).sum()); fn = int((~pred_mask & gt_mask).sum())
    result.update({
        "tp": tp, "fp": fp, "fn": fn,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "pred_changed_fraction": float(pred_mask.mean()),
        "t1_frame_idx": args.t1_frame_idx,
    })
    inference = json.loads((args.labels_dir / "inference.json").read_text())
    decisions = inference.get("decisions") or []
    result["decision_counts"] = inference.get("decision_counts")
    result["changed_pixel_fraction"] = inference.get("changed_pixel_fraction")
    result["visibility_filter_rejected"] = inference.get("visibility_filter_rejected")
    result["horizon_suppressed"] = inference.get("horizon_suppressed")
    result["corroboration_rejected"] = inference.get("corroboration_rejected")
    result["tracking_recoveries"] = (inference.get("tracking_recovery") or {}).get("tracking_recoveries")
    result["object_counts"] = inference.get("object_counts")
    result["timings"] = inference.get("timings")
    result["n_decisions_by_kind"] = {k: sum(1 for d in decisions if d.get("decision") == k)
                                     for k in ("unchanged", "moved", "removed", "added", "replaced",
                                               "visibility_filtered", "horizon_suppressed")}
    by_evidence: dict[str, int] = {}
    for d in decisions:
        if d.get("decision") in ("unchanged", "moved"):
            ev = d.get("evidence") or "none"
            by_evidence[ev] = by_evidence.get(ev, 0) + 1
    result["confirmed_identity_by_evidence"] = by_evidence
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"iou={result['iou']:.4f} P={result['precision']:.4f} R={result['recall']:.4f} "
          f"added_recall={result['_recall'].get('added_recall')} moved_bucket_recall={result['_recall'].get('moved_bucket_recall')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
