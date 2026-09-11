#!/usr/bin/env python3
"""Aggregate the SceneDiff-paired single-query run into per-query CSV + summary.

Sources (read only):
  manifest                      frozen queries (split, subset, frames, co-visibility, GT ids)
  <root>/SceneDiff/<pair>/t1_<annot>/<exp>/metrics/metrics.json   restricted t1-space
                                pixel metrics (scenediff_diag_eval.py: iou/tp/fp/fn)
  .../labels/inference.json     decision counts, FP-by-class inputs
  <official_eval>/official_eval_summary.json   official-evaluator numbers (if present)

Aggregates: ALL queries; evaluable-visible (non-empty T1 GT); SD-V; SD-K -- each
with pooled px/im IoU, mean per-query IoU, P/R/F1, TP/FP/FN, decision counts,
FP pixels by predicted class, no-prediction and failure counts. Empty-GT
queries are kept in ALL (TP=0, predictions are FP). REPLACED is counted as
changed for pixel metrics and reported separately as a diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LABEL_NAME = {1: "added", 2: "removed", 3: "moved", 5: "replaced"}


def pooled(rows):
    tp = sum(r["tp"] or 0 for r in rows); fp = sum(r["fp"] or 0 for r in rows); fn = sum(r["fn"] or 0 for r in rows)
    p = tp / (tp + fp) if tp + fp else None; rc = tp / (tp + fn) if tp + fn else None
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "pooled_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": p, "recall": rc, "f1": (2 * p * rc / (p + rc)) if p and rc else None,
            "mean_iou": float(np.mean(ious)) if ious else None,
            "n_zero_iou": sum(1 for v in ious if v == 0), "n_empty_gt": sum(1 for r in rows if r["gt_empty"]),
            "n_no_prediction": sum(1 for r in rows if r["no_prediction"]),
            "decisions": {k: sum((r["decisions"] or {}).get(k, 0) for r in rows) for k in ("added", "removed", "moved", "replaced", "unchanged")},
            "fp_pixels_by_class": {k: sum((r["fp_by_class"] or {}).get(k, 0) for r in rows) for k in ("added", "removed", "moved", "replaced")}}


def fp_by_class(qdir: Path, pair_dir: Path, t1_annot: int):
    """FP pixels per predicted class: labels.npy vs the T1-space GT (same evaluator GT)."""
    try:
        import sys
        sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
        from scenediff_gt_eval import load_query_gt
        import cv2
        lab = np.load(qdir / "labels.npy")
        try:
            gt = load_query_gt(pair_dir, t1_annot).label
        except ValueError:
            gt = np.zeros(lab.shape, np.uint8)
        if gt.shape != lab.shape:
            gt = cv2.resize(gt.astype(np.uint8), (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = {}
        for code, name in LABEL_NAME.items():
            out[name] = int(((lab == code) & (gt == 0)).sum())
        return out
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--official-eval", type=Path, default=None, help="dir holding official_eval_summary.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    bench = REPO / "data/scenediff_benchmark/data"
    rows = []
    for q in man["queries"]:
        qdir = args.results_root / "SceneDiff" / q["pair"] / f"t1_{q['t1_annotation_idx']:04d}" / args.experiment
        m = json.loads((qdir / "metrics/metrics.json").read_text()) if (qdir / "metrics/metrics.json").exists() else None
        inf = json.loads((qdir / "labels/inference.json").read_text()) if (qdir / "labels/inference.json").exists() else None
        dec = (inf or {}).get("decision_counts")
        rows.append({**{k: q[k] for k in ("split", "subset", "pair", "t1_annotation_idx", "t1_original_idx", "covisibility",
                                          "above_threshold", "gt_empty", "t1_decode")},
                     "gt_object_ids": ";".join(map(str, q["gt_object_ids"])),
                     "evaluated": m is not None, "iou": (m or {}).get("iou"), "tp": (m or {}).get("tp"), "fp": (m or {}).get("fp"), "fn": (m or {}).get("fn"),
                     "no_prediction": (inf is not None) and sum((dec or {}).get(k, 0) for k in ("added", "removed", "moved", "replaced")) == 0,
                     "decisions": dec, "fp_by_class": fp_by_class(qdir, bench / q["pair"], q["t1_annotation_idx"]) if m else None,
                     "n_render_t0_objects": ((inf or {}).get("object_counts") or {}).get("render_t0"),
                     "n_image_t1_objects": ((inf or {}).get("object_counts") or {}).get("image_t1")})
    ev = [r for r in rows if r["evaluated"]]
    agg = {"ALL": pooled(ev), "evaluable_visible": pooled([r for r in ev if not r["gt_empty"]]),
           "SD-V": pooled([r for r in ev if r["subset"] == "SD-V"]), "SD-K": pooled([r for r in ev if r["subset"] == "SD-K"]),
           "SD-V_evaluable": pooled([r for r in ev if r["subset"] == "SD-V" and not r["gt_empty"]]),
           "SD-K_evaluable": pooled([r for r in ev if r["subset"] == "SD-K" and not r["gt_empty"]])}
    official = json.loads((args.official_eval / "official_eval_summary.json").read_text()) if args.official_eval and (args.official_eval / "official_eval_summary.json").exists() else None
    out = {"manifest": str(args.manifest), "manifest_meta": man["_meta"], "n_queries": len(rows), "n_evaluated": len(ev),
           "n_not_evaluated": len(rows) - len(ev), "not_evaluated": [r["pair"] for r in rows if not r["evaluated"]],
           "aggregates": agg, "official_evaluator": official and {k: official[k] for k in ("pixel", "object_iou05", "n_queries_no_prediction", "predicted_objects_by_label")}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scenediff_single_query_covis_summary.json").write_text(json.dumps(out, indent=1, default=str))
    with open(args.out_dir / "scenediff_single_query_covis_per_query.csv", "w", newline="") as f:
        cols = ["split", "subset", "pair", "t1_annotation_idx", "t1_original_idx", "covisibility", "above_threshold", "gt_empty", "gt_object_ids",
                "t1_decode", "evaluated", "iou", "tp", "fp", "fn", "no_prediction", "n_render_t0_objects", "n_image_t1_objects",
                "dec_added", "dec_removed", "dec_moved", "dec_replaced", "dec_unchanged", "fp_px_added", "fp_px_removed", "fp_px_moved", "fp_px_replaced"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            d = r["decisions"] or {}; fb = r["fp_by_class"] or {}
            w.writerow({**{k: r.get(k) for k in cols if k in r}, **{f"dec_{k}": d.get(k) for k in ("added", "removed", "moved", "replaced", "unchanged")},
                        **{f"fp_px_{k}": fb.get(k) for k in ("added", "removed", "moved", "replaced")}})
    f4 = lambda x: "—" if x is None else f"{x:.4f}"
    print(f"queries {len(rows)}  evaluated {len(ev)}  not evaluated {len(rows)-len(ev)}")
    print(f"{'aggregate':<20}{'n':>5}{'pooledIoU':>11}{'meanIoU':>9}{'P':>8}{'R':>8}{'F1':>8}{'emptyGT':>9}{'noPred':>8}")
    for k, a in agg.items():
        print(f"{k:<20}{a['n']:>5}{f4(a['pooled_iou']):>11}{f4(a['mean_iou']):>9}{f4(a['precision']):>8}{f4(a['recall']):>8}{f4(a['f1']):>8}{a['n_empty_gt']:>9}{a['n_no_prediction']:>8}")
    a = agg["ALL"]; print(f"decisions {a['decisions']}  FP px by class {a['fp_pixels_by_class']}")
    if official: print(f"official: pixel IoU {f4(official['pixel']['pooled_iou'])}  object P/R/F1 @0.5 {f4(official['object_iou05']['precision'])}/{f4(official['object_iou05']['recall'])}/{f4(official['object_iou05']['f1'])}")
    print(f"wrote {args.out_dir}/scenediff_single_query_covis_{{summary.json,per_query.csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
