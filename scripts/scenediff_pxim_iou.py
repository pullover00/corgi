#!/usr/bin/env python3
"""Compute px/im IoU matching SceneDiff's own definition
(scene_diff/scripts/evaluate_multiview.py: global_iou = TP/(TP+FP+FN),
pooled across all frames -- see compute_final_metrics/lines ~680-716),
RESTRICTED to image_t1's own pixel space (single representative frame per
scene, video2 side only) -- NOT the full multi-frame video1+video2
protocol their official number uses. This is a deliberately modified,
smaller-scope metric; do not present it as directly comparable to the
paper's px/im IoU without this caveat.

pred_mask = our labels.png != 0 (any changed class), in image_t1 space.
gt_mask = union of GT ADDED + moved-bucket object masks in image_t1's
exact frame (see scenediff_gt_eval.load_query_gt). REMOVED objects are
structurally invisible in image_t1, so any REMOVED-labeled prediction here
counts as FP if it doesn't overlap something the GT calls added/moved-
bucket -- an inherent, documented limitation of the restricted scope, not
a bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from scenediff_gt_eval import load_query_gt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    bench_root = args.benchmark_root
    out_root = args.output_root
    query_meta = json.loads((out_root / "query_meta.json").read_text())

    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    per_pair = {}
    failed = []

    for pair_id, meta in query_meta.items():
        pair_dir = bench_root / "data" / pair_id
        labels_path = REPO / meta["output_dir"] / "labels.png"
        try:
            gt = load_query_gt(pair_dir, meta["t1_frame_idx"])
            pred = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE)
            if pred is None:
                raise FileNotFoundError(labels_path)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"[{pair_id}] FAILED: {e}")
            failed.append(pair_id)
            continue

        pred_mask = pred != 0
        gt_mask = gt.label != 0

        tp = float(np.logical_and(pred_mask, gt_mask).sum())
        fp = float(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
        fn = float(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())

        global_tp += tp
        global_fp += fp
        global_fn += fn
        pair_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        per_pair[pair_id] = {"tp": tp, "fp": fp, "fn": fn, "iou": pair_iou}
        print(f"[{pair_id}] tp={tp:.0f} fp={fp:.0f} fn={fn:.0f} iou={pair_iou:.4f}")

    union = global_tp + global_fp + global_fn
    global_iou = global_tp / union if union > 0 else 0.0

    print("=" * 70)
    print(f"n_pairs succeeded: {len(per_pair)}/{len(query_meta)} (failed: {failed})")
    print(f"global TP={global_tp:.0f} FP={global_fp:.0f} FN={global_fn:.0f}")
    print(f"px/im IoU (t1-frame-only, modified protocol) = {global_iou:.4f}")

    out = {
        "per_pair": per_pair, "failed": failed,
        "global_tp": global_tp, "global_fp": global_fp, "global_fn": global_fn,
        "global_iou_t1_only": global_iou,
    }
    out_path = out_root / "pxim_iou_t1only.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
