#!/usr/bin/env python3
"""Case inspection for the SceneDiff overnight ablations: which object-level
decisions differ between a variant and the baseline, and where each
differing mask lands on the image_t1-space ground truth.

For every pair it lists
  * identity associations (t0,t1) accepted by only one of the two runs,
    with the union mask's GT-change fraction -- a high fraction means the
    association hides a real change (accepting it costs recall), a low one
    means it covers static content (accepting it removes false positives);
  * baseline SAM2 recovery events (t0/t1 ids) with the recovered object's
    proposal-mask GT fraction -- low = the recovery removed a false
    ADDED/REMOVED, high = it removed a true one.
Needs pycocotools (run in the goldilocs env).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from scenediff_gt_eval import load_query_gt  # noqa: E402

ROOT = REPO / "results/scenediff_diagnostic/SceneDiff"
BENCH = REPO / "data/scenediff_benchmark/data"
PAIRS = [l.strip() for l in (REPO / "data/scenediff_benchmark/diagnostic_subset.txt").read_text().split()]
IDENTITY = ("direct_identity", "clean_bridge_identity")


def unpack(npz) -> np.ndarray:
    shape = tuple(int(v) for v in npz["shape"])
    return np.unpackbits(npz["packed"], axis=-1)[..., : shape[-1]].astype(bool).reshape(shape)


def query_dir(pair):
    return sorted(p for p in (ROOT / pair).iterdir() if p.name.startswith("t1_"))[0]


def gt_on_grid(pair, t1_idx, shape):
    gt = load_query_gt(BENCH / pair, t1_idx)
    return cv2.resize((gt.label != 0).astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def frac(mask, gt):
    a = int(mask.sum())
    return (float((mask & gt).sum()) / a) if a else 0.0, a


def identity_pairs(decisions):
    return {(d["t0_object_id"], d["t1_object_id"]): d for d in decisions
            if d.get("decision") in ("unchanged", "moved") and d.get("evidence") in IDENTITY}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="scenediff_v10_no_dino")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {}
    for pair in PAIRS:
        q = query_dir(pair)
        t1_idx = int(q.name.split("_")[1])
        b_dir, v_dir = q / args.baseline, q / args.variant
        if not (v_dir / "labels/inference.json").exists():
            continue
        b_inf = json.loads((b_dir / "labels/inference.json").read_text())
        v_inf = json.loads((v_dir / "labels/inference.json").read_text())
        t0 = unpack(np.load(b_dir / "proposals/t0_selected.npz")); t1 = unpack(np.load(b_dir / "proposals/t1_selected.npz"))
        # the variant's own proposals (identical for replays, different for no_refine)
        v_t0 = unpack(np.load(v_dir / "proposals/t0_selected.npz")); v_t1 = unpack(np.load(v_dir / "proposals/t1_selected.npz"))
        gt = gt_on_grid(pair, t1_idx, t0.shape[1:])
        b_ids, v_ids = identity_pairs(b_inf["decisions"]), identity_pairs(v_inf["decisions"])
        rows = {"only_baseline": [], "only_variant": [], "baseline_recoveries": [], "n_gt_pixels_on_grid": int(gt.sum()),
                "n_objects": {"baseline": b_inf["object_counts"], "variant": v_inf["object_counts"]}}
        same_inventory = t0.shape == v_t0.shape and t1.shape == v_t1.shape
        for key in sorted(set(b_ids) - set(v_ids)):
            d = b_ids[key]; m = t0[key[0] - 1] | t1[key[1] - 1]
            f, a = frac(m, gt)
            rows["only_baseline"].append({"t0": key[0], "t1": key[1], "evidence": d["evidence"], "sam_cosine": d.get("sam_cosine"),
                                          "track_iou": d.get("track_iou"), "spatial_iou": d.get("spatial_iou"), "gt_fraction": f, "area": a})
        for key in sorted(set(v_ids) - set(b_ids)):
            d = v_ids[key]
            if same_inventory:
                m = v_t0[key[0] - 1] | v_t1[key[1] - 1]; f, a = frac(m, gt)
            else:
                m = v_t0[key[0] - 1] | v_t1[key[1] - 1]; f, a = frac(m, gt)
            rows["only_variant"].append({"t0": key[0], "t1": key[1], "evidence": d["evidence"], "sam_cosine": d.get("sam_cosine"),
                                         "track_iou": d.get("track_iou"), "spatial_iou": d.get("spatial_iou"), "gt_fraction": f, "area": a})
        rec = b_inf.get("tracking_recovery") or {}
        for i in rec.get("tracking_recovered_t0_ids", []):
            f, a = frac(t0[i - 1], gt); rows["baseline_recoveries"].append({"side": "t0", "id": i, "gt_fraction": f, "area": a})
        for i in rec.get("tracking_recovered_t1_ids", []):
            f, a = frac(t1[i - 1], gt); rows["baseline_recoveries"].append({"side": "t1", "id": i, "gt_fraction": f, "area": a})
        report[pair] = rows
        nb, nv = len(rows["only_baseline"]), len(rows["only_variant"])
        print(f"{pair[:34]:36s} identities only-baseline={nb} only-variant={nv} recoveries={len(rows['baseline_recoveries'])} "
              f"| only-variant on-GT>0.5: {sum(1 for r in rows['only_variant'] if r['gt_fraction'] > 0.5)} "
              f"| recoveries on-GT>0.5: {sum(1 for r in rows['baseline_recoveries'] if r['gt_fraction'] > 0.5)}")
    out = args.out or ROOT / "_experiments" / args.variant / "case_inspection_vs_baseline.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
