#!/usr/bin/env python3
"""Workaround B: score CORGI's REMOVED predictions in the BEFORE frames.

SceneDiff annotates a removed object only in the before video (it is absent
from the after video), while CORGI localizes it in the query view as the
rendered silhouette of the reference reconstruction. This tool carries that
prediction back to where the ground truth lives:

  1. Every pixel of a REMOVED (or REPLACED) mask in the query view has a 3D
     point (render/render_t0_positions.npy) -- the exact reference-scene point
     that won the z-buffer, expressed in the query-aligned frame
     align(p) = s * p @ R^T + t.
  2. (s, R, t) were not saved by the run. They are recovered EXACTLY from the
     saved buffers: render_t0_confidence.npy stores each winning point's own
     normalized confidence, a continuous per-point scalar untouched by align(),
     so rendered pixels are matched to their source reference points by that
     value and the pipeline's own _fit_similarity_transform is refitted. The
     fit is verified by the nearest-neighbour residual of ALL rendered points
     mapped back into the reference cloud; a query whose residual is not tiny
     is rejected, never silently used.
  3. The object's points are projected into each of the T0 reference cameras
     (reference_scene.pkl extrinsic/intrinsic, camera-from-world) with a
     depth test against that frame's own surface, rasterized, and exported as
     object_masks.pkl 'video_1' entries keyed by the frame's annotation index
     (round(original_idx * 30 / fps)).
  4. Scored with the official evaluator functions against the official
     before-video masks, GT restricted to those reference frames, resample 1.

The ground truth is untouched; the prediction is what moves, using the
reconstruction that produced it. If that reconstruction is wrong the
transferred prediction misses and CORGI is penalized, not helped.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
# reference_scene.pkl is written by the runner under Python 3.13, where PosixPath is
# defined in pathlib._local; alias it so the 3.11 evaluation env can unpickle it.
import pathlib as _pathlib
sys.modules.setdefault("pathlib._local", _pathlib)
SCENE_DIFF = Path("/home/tessa/scene_diff")
BENCH = REPO / "data/scenediff_benchmark"
PROJECT_LABELS = {2: "removed", 5: "replaced", 3: "moved"}   # ADDED never: its render points are background


def load_official():
    for n in ("faiss", "open3d", "torch_scatter"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.path.insert(0, str(SCENE_DIFF / "scripts")); sys.path.insert(0, str(SCENE_DIFF))
    import evaluate_multiview as E
    return E


def unpack(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    n, h, w = [int(x) for x in z["shape"]]
    masks = np.unpackbits(z["packed"], axis=-1)[..., :w].astype(bool).reshape(n, h, w) if n else np.zeros((0, h, w), bool)
    return masks, z["labels"].astype(int)


def recover_transform(scene, positions, coverage, confidence, conf_pct, cleaned_pct, rng):
    """Refit (rotation, translation, scale) with align(p) = s p R^T + t; verify on all rendered points."""
    from scipy.spatial import cKDTree
    from ocmask_pipeline.reconstruction import _fit_similarity_transform

    W, dc = scene.world_points, scene.depth_conf
    thr = np.percentile(dc, conf_pct); cleaned = np.percentile(dc, cleaned_pct)
    scale_c = max(float(cleaned - thr), 1e-6)
    ref_mask = np.isfinite(W).all(axis=-1) & (dc >= thr)
    pts_ref = W[ref_mask].astype(np.float64)
    conf_ref = np.clip((dc[ref_mask].astype(np.float64) - thr) / scale_c, 0.0, 1.0).astype(np.float32)

    fin = coverage & np.isfinite(positions).all(axis=-1) & np.isfinite(confidence)
    usable = fin & (confidence > 0.02) & (confidence < 0.98)
    idx = np.flatnonzero(usable.ravel())
    if len(idx) < 50:
        return None, {"reason": f"only {len(idx)} unclipped rendered pixels"}
    rng.shuffle(idx)
    order = np.argsort(conf_ref); sorted_conf = conf_ref[order]
    # The run stored the winning point's confidence as float32; recomputing it here can
    # differ by a rounding ulp (bathroom_11 in the smoke test: every value within 1e-6,
    # none bit-equal). Match within 2 ulp and require the window to hold ONE reference value.
    tol = np.float32(2.5e-7)
    src, tgt = [], []
    for k in idx[:3000]:
        c = confidence.ravel()[k]
        lo, hi = np.searchsorted(sorted_conf, c - tol, "left"), np.searchsorted(sorted_conf, c + tol, "right")
        if hi - lo == 1:                       # exactly one reference point within tolerance
            src.append(pts_ref[order[lo]]); tgt.append(positions.reshape(-1, 3)[k])
        if len(src) >= 200:
            break
    if len(src) < 8:
        return None, {"reason": f"only {len(src)} unique-confidence correspondences"}
    src, tgt = np.asarray(src), np.asarray(tgt, dtype=np.float64)
    R, t, s = _fit_similarity_transform(src, tgt)
    fit_res = np.linalg.norm(s * src @ R.T + t - tgt, axis=1)
    # robust refit on inliers (a few confidence collisions are possible)
    inl = fit_res < np.percentile(fit_res, 80) + 1e-9
    R, t, s = _fit_similarity_transform(src[inl], tgt[inl])
    # global verification: every rendered point, mapped back, must sit on a reference point
    back = ((positions[fin] - t) @ R) / s
    tree = cKDTree(pts_ref if len(pts_ref) <= 3_000_000 else pts_ref[rng.choice(len(pts_ref), 3_000_000, replace=False)])
    d, _ = tree.query(back[rng.choice(len(back), min(20000, len(back)), replace=False)], k=1)
    extent = float(np.percentile(np.linalg.norm(pts_ref - pts_ref.mean(0), axis=1), 90))
    stats = {"n_corr": int(inl.sum()), "scale": float(s), "fit_residual_med": float(np.median(fit_res[inl])),
             "nn_residual_med_rel": float(np.median(d) / extent), "nn_residual_p95_rel": float(np.percentile(d, 95) / extent),
             "extent": extent}
    ok = stats["nn_residual_p95_rel"] < 2e-3
    return ((R, t, s) if ok else None), {**stats, "accepted": ok}


def project_to_frames(points_ref, scene, frame_depths, tol=0.03):
    """points_ref (M,3) -> per reference frame k: (boolean mask HxW) after depth test."""
    N, H, W_ = scene.depth_conf.shape
    out = []
    for k in range(N):
        E = scene.extrinsic[k].astype(np.float64); K = scene.intrinsic[k].astype(np.float64)
        cam = points_ref @ E[:3, :3].T + E[:3, 3]
        z = cam[:, 2]; ok = z > 1e-6
        uv = (cam[ok] @ K.T); u = uv[:, 0] / uv[:, 2]; v = uv[:, 1] / uv[:, 2]
        ui = np.round(u).astype(int); vi = np.round(v).astype(int)
        inb = (ui >= 0) & (ui < W_) & (vi >= 0) & (vi < H)
        ui, vi, zz = ui[inb], vi[inb], z[ok][inb]
        surf = frame_depths[k][vi, ui]
        keep = np.isfinite(surf) & (zz <= surf * (1 + tol))       # not behind that frame's own surface
        m = np.zeros((H, W_), bool); m[vi[keep], ui[keep]] = True
        out.append(m)
    return out


def tidy(mask, min_px=32):
    from scipy import ndimage
    m = ndimage.binary_dilation(mask, iterations=1)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    return m if m.sum() >= min_px else None


def main() -> int:
    from pycocotools import mask as mask_utils
    from run_scenediff_batch import video_meta

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs/scenediff_v10_no_dino_no_refine.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--audit-overlays", type=int, default=0, help="write this many (query, frame) overlay PNGs")
    args = ap.parse_args()

    import yaml
    recon = yaml.safe_load(args.config.read_text())["reconstruction"]
    conf_pct, cleaned_pct = float(recon["depth_confidence_percentile"]), float(recon["cleaned_depth_confidence_percentile"])
    E = load_official(); man = json.loads(args.manifest.read_text())
    pred_root, gt_root = args.out_dir / "official_pred_before", args.out_dir / "official_gt_before_restricted"
    for d in (pred_root, gt_root):
        if d.exists(): shutil.rmtree(d)
    (args.out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    records, n_overlay = [], 0
    for q in man["queries"]:
        pair, t1 = q["pair"], q["t1_annotation_idx"]
        base = args.results_root / "SceneDiff" / pair
        qdir = base / f"t1_{t1:04d}"; exp = qdir / args.experiment
        rec = {"pair": pair, "t1_annotation_idx": t1, "status": "skipped"}
        try:
            if not (exp / "resolution/final_objects.npz").exists():
                rec["status"] = "no_output"; records.append(rec); continue
            scene = pickle.loads((base / "shared/reference_reconstruction/reference_scene.pkl").read_bytes())
            pos = np.load(qdir / "shared/render/render_t0_positions.npy"); cov = np.load(qdir / "shared/render/render_t0_coverage.npy")
            conf = np.load(qdir / "shared/render/render_t0_confidence.npy")
            masks, labels = unpack(exp / "resolution/final_objects.npz")
            sel = [i for i, l in enumerate(labels) if int(l) in PROJECT_LABELS]
            rec["n_objects_projectable"] = len(sel)
            transform, tstats = recover_transform(scene, pos, cov, conf, conf_pct, cleaned_pct, rng)
            rec["transform"] = tstats
            if transform is None:
                rec["status"] = "transform_rejected"; records.append(rec); continue
            R, t, s = transform
            # frame k -> annotation index
            t0 = json.loads((base / "shared/reference_reconstruction/t0_frames.json").read_text())["indices"]
            _, fps1 = video_meta(q["video1_path"])
            annot = [int(round(i * 30.0 / fps1)) for i in t0]
            assert len(annot) == scene.depth_conf.shape[0], "reference frame count mismatch"
            frame_depths = [(scene.world_points[k] @ scene.extrinsic[k][:3, :3].T + scene.extrinsic[k][:3, 3])[..., 2] for k in range(len(annot))]
            obj = {"H": int(scene.depth_conf.shape[1]), "W": int(scene.depth_conf.shape[2])}
            n_masks = 0; overlays_this_pair = 0
            for i in sel:
                m = masks[i] & cov & np.isfinite(pos).all(axis=-1)
                if m.sum() < 8: continue
                pts_ref = ((pos[m] - t) @ R) / s
                per_frame = project_to_frames(pts_ref, scene, frame_depths)
                entry = {}
                for k, fm in enumerate(per_frame):
                    fm = tidy(fm)
                    if fm is None: continue
                    rle = mask_utils.encode(np.asfortranarray(fm.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
                    entry[annot[k]] = {"mask": rle, "cost": 1.0}
                    n_masks += 1
                    if n_overlay < args.audit_overlays and overlays_this_pair < 2:
                        from PIL import Image
                        img = np.asarray(Image.open(scene.image_paths[k]).convert("RGB")).copy()
                        if img.shape[:2] != fm.shape:
                            img = np.asarray(Image.fromarray(img).resize((fm.shape[1], fm.shape[0]))).copy()
                        # GT (video1 masks at this annotation frame) in green, projected prediction in red, overlap yellow
                        gtm = np.zeros(fm.shape, bool)
                        seg_full = pickle.loads((BENCH / "data" / pair / "segments.pkl").read_bytes())
                        for oid, fr in seg_full.get("video1_objects", {}).items():
                            key = next((kk for kk in fr if int(kk) == annot[k]), None)
                            if key is not None:
                                g = mask_utils.decode(fr[key]).astype(bool)
                                if g.shape != fm.shape:
                                    g = np.asarray(Image.fromarray(g.astype(np.uint8) * 255).resize((fm.shape[1], fm.shape[0]), Image.NEAREST)) > 0
                                gtm |= g
                        img[gtm & ~fm] = (0.4 * img[gtm & ~fm] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
                        img[fm & ~gtm] = (0.4 * img[fm & ~gtm] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
                        img[fm & gtm] = (0.3 * img[fm & gtm] + 0.7 * np.array([255, 255, 0])).astype(np.uint8)
                        Image.fromarray(img).save(args.out_dir / "overlays" / f"{pair[:30]}_t1{t1}_obj{i}_{PROJECT_LABELS[int(labels[i])]}_T0annot{annot[k]}.png")
                        n_overlay += 1; overlays_this_pair += 1
                if entry:
                    obj[f"corgi_{pair}_{t1}_{i}_{PROJECT_LABELS[int(labels[i])]}"] = {"video_1": entry}
            (pred_root / pair).mkdir(parents=True, exist_ok=True)
            (pred_root / pair / "object_masks.pkl").write_bytes(pickle.dumps(obj))
            # restricted GT: video1 masks at the reference annotation frames only
            seg = pickle.loads((BENCH / "data" / pair / "segments.pkl").read_bytes())
            keep = set(annot); seg["video2_objects"] = {}
            seg["video1_objects"] = {oid: {k: v for k, v in fr.items() if int(k) in keep} for oid, fr in seg.get("video1_objects", {}).items()}
            seg["video1_objects"] = {oid: fr for oid, fr in seg["video1_objects"].items() if fr}
            (gt_root / pair).mkdir(parents=True, exist_ok=True); (gt_root / pair / "segments.pkl").write_bytes(pickle.dumps(seg))
            rec.update({"status": "ok", "n_frame_masks": n_masks, "reference_annotation_frames": annot,
                        "n_gt_video1_masks_at_those_frames": sum(len(fr) for fr in seg["video1_objects"].values())})
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"error: {type(e).__name__}: {str(e)[:120]}"
        records.append(rec)

    # ---- official scoring on before frames, resample 1 ----
    scenes = sorted(r["pair"] for r in records if r["status"] == "ok")
    eargs = types.SimpleNamespace(iou_threshold=args.iou_threshold, duplicate_match_threshold=1, per_frame_duplicate_match_threshold=1)
    import torch
    tp = fp = fn = 0.0; region_info, obj_info, obj_lab = [], [], []; total_gt = 0; total_gt_objects = {}; per_scene = {}
    for scene_name in scenes:
        pred_data, gt_data = E.load_scene_data(pred_root / scene_name, gt_root / scene_name)
        hw = E.get_target_hw_from_gt(gt_data, 1024)
        if hw is None:
            s_ = 1024 / max(pred_data["H"], pred_data["W"]); hw = (max(1, round(pred_data["H"] * s_)), max(1, round(pred_data["W"] * s_)))
        H, W_ = hw
        gt_objs, gt_labels, by_label = E.extract_ground_truth(gt_data, H, W_, 1)
        total_gt += len(gt_objs); total_gt_objects[scene_name] = {0: by_label[0], 1: by_label[1]}
        dets = E.extract_detections(pred_data, H, W_)
        pm, gm = {}, {}
        for d in dets:
            pm[d["frame_idx"]] = np.logical_or(pm.get(d["frame_idx"], np.zeros((H, W_), bool)), d["mask"].cpu().numpy().astype(bool))
        for (_, fi, vid), m in gt_objs.items():
            gm[fi] = np.logical_or(gm.get(fi, np.zeros((H, W_), bool)), (m.numpy() if isinstance(m, torch.Tensor) else np.asarray(m)).astype(bool))
        s_tp = s_fp = s_fn = 0.0
        for fi in set(pm) | set(gm):
            p, g = pm.get(fi, np.zeros((H, W_), bool)), gm.get(fi, np.zeros((H, W_), bool))
            s_tp += float((p & g).sum()); s_fp += float((p & ~g).sum()); s_fn += float((~p & g).sum())
        tp += s_tp; fp += s_fp; fn += s_fn
        matched = {k: 0 for k in gt_objs}; n0 = len(region_info)
        E.match_detections_to_gt(dets, gt_objs, gt_labels, matched, region_info, scene_name, eargs)
        per_scene[scene_name] = {"tp": s_tp, "fp": s_fp, "fn": s_fn, "iou": s_tp / (s_tp + s_fp + s_fn) if s_tp + s_fp + s_fn else None,
                                 "n_gt_regions": len(gt_objs), "n_detections": len(dets),
                                 "region_tp": sum(1 for r in region_info[n0:] if r["gt_matched"]), "region_fp": sum(1 for r in region_info[n0:] if not r["gt_matched"])}
    r_tp = sum(1 for r in region_info if r["gt_matched"]); r_fp = len(region_info) - r_tp
    P = r_tp / (r_tp + r_fp) if r_tp + r_fp else None; Rc = r_tp / total_gt if total_gt else None
    summary = {"n_queries": len(records), "status_counts": {s: sum(1 for r in records if r["status"] == s) for s in sorted({r["status"] for r in records})},
               "transform_verification": {"accepted": sum(1 for r in records if r.get("transform", {}).get("accepted")),
                                          "nn_residual_p95_rel_median": float(np.median([r["transform"]["nn_residual_p95_rel"] for r in records if "transform" in r and "nn_residual_p95_rel" in r["transform"]])) if any("transform" in r and "nn_residual_p95_rel" in r["transform"] for r in records) else None},
               "pixel_before_frames": {"pooled_iou": tp / (tp + fp + fn) if tp + fp + fn else None, "tp": tp, "fp": fp, "fn": fn,
                                       "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None},
               "object_before_frames_iou05": {"tp": r_tp, "fp": r_fp, "n_gt_regions": total_gt, "precision": P, "recall": Rc,
                                              "f1": (2 * P * Rc / (P + Rc)) if P and Rc else None},
               "per_scene": per_scene, "records": records}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "removed_projection_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    f4 = lambda x: "—" if x is None else f"{x:.4f}"
    print(f"queries {len(records)}: {summary['status_counts']}")
    print(f"transform: accepted {summary['transform_verification']['accepted']}, median p95 NN residual (rel. extent) {f4(summary['transform_verification']['nn_residual_p95_rel_median'])}")
    px, ob = summary["pixel_before_frames"], summary["object_before_frames_iou05"]
    print(f"BEFORE-frame pixel : pooled IoU {f4(px['pooled_iou'])}  P {f4(px['precision'])}  R {f4(px['recall'])}  (TP {px['tp']:.0f} FP {px['fp']:.0f} FN {px['fn']:.0f})")
    print(f"BEFORE-frame object: P {f4(ob['precision'])}  R {f4(ob['recall'])}  F1 {f4(ob['f1'])} @ IoU {args.iou_threshold}  (TP {ob['tp']} FP {ob['fp']} of {ob['n_gt_regions']} GT regions)")
    print(f"wrote {args.out_dir / 'removed_projection_summary.json'}; overlays: {n_overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
