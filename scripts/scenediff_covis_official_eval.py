#!/usr/bin/env python3
"""Score CORGI's single-query SceneDiff outputs with the OFFICIAL evaluator.

Two steps, both restricted to the evaluated (after-video, frame) set from the
frozen manifest:

1. Export. Each query's final objects (resolution/final_objects.npz: per-object
   masks in image_t1 space + CORGI label codes) are written as
   <pred_root>/<pair>/object_masks.pkl in the exact format the official
   pipeline writes (modules/scenediff.py:782): {'H','W', obj_id: {'video_2':
   {frame_idx // 30: {'mask': RLE, 'cost': c}}}}. frame_idx is the annotation
   index (30 fps space), so frame_idx // 30 is the evaluator's resampled index.
   CORGI has no confidence, so cost = 1.0 for every object (see the report:
   AP is therefore not meaningful; object precision/recall at IoU 0.5 is).
   REPLACED objects are exported as changed (the benchmark has no such class)
   and counted separately.

2. Restricted GT. A copy of each pair's segments.pkl keeps only video2 masks at
   the evaluated frame(s); video1 masks are dropped. This is the documented
   scope restriction (removed objects are annotated in the before video, which
   a single-query method never scores), not a change to any metric.

Metrics come from the official functions (extract_ground_truth,
extract_detections, match_detections_to_gt, compute_final_metrics) called in
the same order as evaluate_all_scenes; that orchestration is reproduced here
only to expose the per-detection match records for precision/recall. The
official CLI is also run end-to-end on the same files as a cross-check.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCENE_DIFF = Path("/home/tessa/scene_diff")
BENCH = REPO / "data/scenediff_benchmark"
LABEL_NAME = {1: "added", 2: "removed", 3: "moved", 5: "replaced"}


def load_official():
    for n in ("faiss", "open3d", "torch_scatter"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.path.insert(0, str(SCENE_DIFF / "scripts")); sys.path.insert(0, str(SCENE_DIFF))
    import inspect
    import evaluate_multiview as E
    assert str(SCENE_DIFF) in inspect.getsourcefile(E)
    return E


def unpack(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    n, h, w = [int(x) for x in z["shape"]]
    masks = np.unpackbits(z["packed"], axis=-1)[..., :w].astype(bool).reshape(n, h, w) if n else np.zeros((0, h, w), bool)
    return masks, z["labels"].astype(int) if "labels" in z else np.zeros(n, int), (h, w)


def export_query(results_root: Path, experiment: str, q: dict, pred_root: Path) -> dict:
    from pycocotools import mask as mask_utils
    pair, t1 = q["pair"], q["t1_annotation_idx"]
    qdir = results_root / "SceneDiff" / pair / f"t1_{t1:04d}" / experiment
    npz = qdir / "resolution" / "final_objects.npz"
    out = {"pair": pair, "t1_annotation_idx": t1, "resampled_idx": t1 // 30, "exported": False,
           "n_objects": 0, "by_label": {}, "no_prediction": True}
    if not npz.exists():
        return out
    masks, labels, (h, w) = unpack(npz)
    obj = {"H": h, "W": w}
    for i, (m, lab) in enumerate(zip(masks, labels)):
        if int(lab) not in LABEL_NAME or not m.any():
            continue
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
        key = f"corgi_{pair}_{t1}_{i}_{LABEL_NAME[int(lab)]}"
        obj[key] = {"video_2": {t1 // 30: {"mask": rle, "cost": 1.0}}}
        out["by_label"][LABEL_NAME[int(lab)]] = out["by_label"].get(LABEL_NAME[int(lab)], 0) + 1
        out["n_objects"] += 1
    d = pred_root / pair; d.mkdir(parents=True, exist_ok=True)
    (d / "object_masks.pkl").write_bytes(pickle.dumps(obj))
    out.update({"exported": True, "no_prediction": out["n_objects"] == 0})
    return out


def restrict_gt(pair: str, frames_annot: set[int], gt_root: Path) -> None:
    src = BENCH / "data" / pair / "segments.pkl"
    seg = pickle.loads(src.read_bytes())
    seg["video1_objects"] = {}
    v2 = {}
    for oid, frames in seg.get("video2_objects", {}).items():
        kept = {k: v for k, v in frames.items() if int(k) in frames_annot}
        if kept:
            v2[oid] = kept
    seg["video2_objects"] = v2
    d = gt_root / pair; d.mkdir(parents=True, exist_ok=True)
    (d / "segments.pkl").write_bytes(pickle.dumps(seg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True, help="runner --root")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--split", choices=("val", "test"), default=None, help="for the official CLI's --splits; default: manifest's split")
    args = ap.parse_args()

    E = load_official()
    man = json.loads(args.manifest.read_text())
    queries = man["queries"]
    pred_root, gt_root = args.out_dir / "official_pred", args.out_dir / "official_gt_restricted"
    for d in (pred_root, gt_root):
        if d.exists():
            shutil.rmtree(d)

    exports = [export_query(args.results_root, args.experiment, q, pred_root) for q in queries]
    frames_by_pair = {}
    for q in queries:
        frames_by_pair.setdefault(q["pair"], set()).add(q["t1_annotation_idx"])
    for pair, frames in frames_by_pair.items():
        restrict_gt(pair, frames, gt_root)

    scenes = sorted(p for p in frames_by_pair if (pred_root / p / "object_masks.pkl").exists())
    eargs = types.SimpleNamespace(pred_dir=str(pred_root), gt_dir=str(gt_root), video_dir=str(BENCH / "data"),
                                  resample_rate=30, max_length=args.max_length, iou_threshold=args.iou_threshold,
                                  duplicate_match_threshold=1, per_frame_duplicate_match_threshold=1,
                                  visualize=False, mask_background=False, crop=False)

    # ---- official functions, evaluate_all_scenes order ----
    total_gt_regions = 0; total_gt_objects = {}
    region_info, obj_info, obj_info_by_label = [], [], []
    tp = fp = fn = 0.0
    per_scene = {}
    import torch
    for scene in scenes:
        pred_data, gt_data = E.load_scene_data(pred_root / scene, gt_root / scene)
        target_hw = E.get_target_hw_from_gt(gt_data, args.max_length)
        if target_hw is not None:
            H, W = target_hw
        else:
            src_h, src_w = pred_data.get("H", 1024), pred_data.get("W", 576)
            s = float(args.max_length) / float(max(src_h, src_w)); H, W = max(1, round(src_h * s)), max(1, round(src_w * s))
        gt_objs, gt_labels, gt_by_label = E.extract_ground_truth(gt_data, H, W, 30)
        total_gt_regions += len(gt_objs); total_gt_objects[scene] = {0: gt_by_label[0], 1: gt_by_label[1]}
        dets = E.extract_detections(pred_data, H, W)
        pm, gm = {}, {}
        for d in dets:
            if d["video"] != 2: continue
            pm[d["frame_idx"]] = np.logical_or(pm.get(d["frame_idx"], np.zeros((H, W), bool)), d["mask"].cpu().numpy().astype(bool))
        for (_, fi, vid), m in gt_objs.items():
            if vid != 2: continue
            gm[fi] = np.logical_or(gm.get(fi, np.zeros((H, W), bool)), (m.numpy() if isinstance(m, torch.Tensor) else np.asarray(m)).astype(bool))
        s_tp = s_fp = s_fn = 0.0
        for fi in set(pm) | set(gm):
            p, g = pm.get(fi, np.zeros((H, W), bool)), gm.get(fi, np.zeros((H, W), bool))
            s_tp += float((p & g).sum()); s_fp += float((p & ~g).sum()); s_fn += float((~p & g).sum())
        tp += s_tp; fp += s_fp; fn += s_fn
        per_scene[scene] = {"tp": s_tp, "fp": s_fp, "fn": s_fn, "iou": s_tp / (s_tp + s_fp + s_fn) if s_tp + s_fp + s_fn else None,
                            "n_gt_regions": len(gt_objs), "n_detections": len(dets), "target_hw": [int(H), int(W)]}
        pd = {k: v for k, v in pred_data.items() if k not in ("H", "W")}
        for k, v in pd.items():
            v["label"] = 1 if ("video_1" in v and "video_2" in v) else 0
            v["confidence"] = E.compute_object_confidence(v)
        matched = {k: 0 for k in gt_objs}
        n_before = len(region_info)
        E.match_detections_to_gt(dets, gt_objs, gt_labels, matched, region_info, scene, eargs)
        E.evaluate_object_level(pd, gt_objs, gt_labels, obj_info, obj_info_by_label, scene, H, W, eargs)
        per_scene[scene]["region_tp"] = sum(1 for r in region_info[n_before:] if r["gt_matched"])
        per_scene[scene]["region_fp"] = sum(1 for r in region_info[n_before:] if not r["gt_matched"])
    metrics = E.compute_final_metrics(region_info, total_gt_regions, obj_info, obj_info_by_label, total_gt_objects, scenes, tp, fp, fn)

    r_tp = sum(1 for r in region_info if r["gt_matched"]); r_fp = len(region_info) - r_tp
    obj_p = r_tp / (r_tp + r_fp) if r_tp + r_fp else None
    obj_r = r_tp / total_gt_regions if total_gt_regions else None
    summary = {
        "n_scenes_scored": len(scenes), "n_queries": len(queries),
        "n_queries_no_prediction": sum(1 for e in exports if e["no_prediction"]),
        "n_queries_not_exported": sum(1 for e in exports if not e["exported"]),
        "predicted_objects_by_label": {k: sum(e["by_label"].get(k, 0) for e in exports) for k in LABEL_NAME.values()},
        "pixel": {"pooled_iou": metrics["global_iou"], "tp": tp, "fp": fp, "fn": fn,
                  "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None},
        "object_iou05": {"tp": r_tp, "fp": r_fp, "n_gt_regions": total_gt_regions,
                         "precision": obj_p, "recall": obj_r,
                         "f1": (2 * obj_p * obj_r / (obj_p + obj_r)) if obj_p and obj_r else None,
                         "official_per_frame_ap_constant_confidence": metrics["per_frame_ap"]},
        "official_metrics_raw": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in metrics.items()},
        "per_scene": per_scene, "exports": exports,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "official_eval_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # ---- cross-check: the official CLI on the same files ----
    cli_out = args.out_dir / "official_cli_result.txt"
    cmd = [sys.executable, str(SCENE_DIFF / "scripts/evaluate_multiview.py"), "--pred_dir", str(pred_root.resolve()),
           "--gt_dir", str(gt_root.resolve()), "--video_dir", str(BENCH / "data"), "--output_path", str(cli_out.resolve()),
           "--resample_rate", "30", "--iou_threshold", str(args.iou_threshold), "--visualize", "False",
           "--splits", args.split or man["_meta"].get("split", "all"), "--sets", "all"]
    # main() opens data/scenediff_benchmark/splits/*.json RELATIVE to the cwd; that layout exists in this repo
    stub = "import sys,types\nfor n in ('faiss','open3d','torch_scatter'): sys.modules.setdefault(n, types.ModuleType(n))\n"
    runner = args.out_dir / "_run_official_cli.py"
    runner.write_text(stub + f"sys.path.insert(0, {str(SCENE_DIFF / 'scripts')!r}); sys.path.insert(0, {str(SCENE_DIFF)!r})\n"
                      f"sys.argv = {cmd[1:]!r}\nimport runpy; runpy.run_path({str(SCENE_DIFF / 'scripts/evaluate_multiview.py')!r}, run_name='__main__')\n")
    res = subprocess.run([sys.executable, str(runner.resolve())], capture_output=True, text=True, cwd=str(REPO))
    (args.out_dir / "official_cli_stdout.txt").write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
    cli_lines = [l for l in res.stdout.splitlines() if "Metric" in l]

    px, ob = summary["pixel"], summary["object_iou05"]
    f = lambda x: "—" if x is None else f"{x:.4f}"
    print(f"scenes scored {len(scenes)}/{len(frames_by_pair)}; queries {len(queries)}; no-prediction {summary['n_queries_no_prediction']}")
    print(f"pixel   : pooled IoU {f(px['pooled_iou'])}  P {f(px['precision'])}  R {f(px['recall'])}   (TP {px['tp']:.0f} FP {px['fp']:.0f} FN {px['fn']:.0f})")
    print(f"object  : P {f(ob['precision'])}  R {f(ob['recall'])}  F1 {f(ob['f1'])}  @ mask-IoU {args.iou_threshold}  (TP {ob['tp']} FP {ob['fp']} of {ob['n_gt_regions']} GT regions)")
    print(f"official CLI cross-check: " + (" | ".join(cli_lines) if cli_lines else f"(no metric lines; see official_cli_stdout.txt, rc={res.returncode})"))
    print(f"predicted objects by label: {summary['predicted_objects_by_label']}")
    print(f"wrote {args.out_dir / 'official_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
