#!/usr/bin/env python3
"""Pre-inference validation audits for the SceneDiff-paired single-query protocol.

7A  frame-rate mapping: for one pair each at ~10, ~30, 60 and 120 fps, the
    review-video frame at annotation index i beside the ORIGINAL frame at the
    mapped index, with the image correlation of both (and of the unmapped
    original[i] where it exists) -- the mapping must win clearly.
7B  ground-truth alignment: for >=10 random non-empty queries, the original RGB
    query frame with the decoded SceneDiff GT masks overlaid -- masks must sit
    on the objects.
7C  pairing: for >=10 queries, the before-frame that gave the highest
    co-visibility beside the selected after-frame, with the score printed.

Reads only; nothing here influences selection.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
BENCH = REPO / "data/scenediff_benchmark"


def grab_review(path, idx):
    import cv2
    c = cv2.VideoCapture(str(path)); c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = c.read(); c.release()
    return f if ok else None


def corr(a, b):
    import cv2
    if a is None or b is None: return None
    if (a.shape[0] < a.shape[1]) != (b.shape[0] < b.shape[1]): b = cv2.rotate(b, cv2.ROTATE_90_CLOCKWISE)
    a = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (160, 120)).astype(float); b = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (160, 120)).astype(float)
    a -= a.mean(); b -= b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d else None


def side_by_side(panels, labels, out, height=480):
    import cv2
    from PIL import Image, ImageDraw
    tiles = []
    for p, lab in zip(panels, labels):
        if p is None: p = np.zeros((height, int(height * 0.56), 3), np.uint8)
        if p.shape[0] < p.shape[1]: p = cv2.rotate(p, cv2.ROTATE_90_CLOCKWISE)
        s = height / p.shape[0]; p = cv2.resize(p, (int(p.shape[1] * s), height))
        img = Image.fromarray(p[:, :, ::-1]); ImageDraw.Draw(img).rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        ImageDraw.Draw(img).text((4, 4), lab, fill=(255, 255, 255)); tiles.append(np.asarray(img))
    gap = np.full((height, 6, 3), 255, np.uint8)
    row = []
    for i, t in enumerate(tiles):
        if i: row.append(gap)
        row.append(t)
    Image.fromarray(np.concatenate(row, axis=1)).save(out)


def audit_7a(out_dir: Path, report: dict):
    from run_scenediff_batch import annotation_to_original_index, read_frame_robust, resolve_original_video, video_meta
    q = json.loads((BENCH / "test250_queries_v2.json").read_text())
    picks = {}
    for pair, rec in q.items():
        if rec["t1_idx"] <= 0: continue
        r = rec["fps_ratio"]; cls = "10fps" if r < 0.5 else ("30fps" if abs(r - 1) < 0.05 else ("60fps" if r < 3 else "120fps"))
        picks.setdefault(cls, []).append(pair)
    rows = []
    for cls in ("10fps", "30fps", "60fps", "120fps"):
        for pair in picks.get(cls, [])[:2]:
            pd = BENCH / "data" / pair; v2 = resolve_original_video(pd, 2); i = q[pair]["t1_idx"]
            m = annotation_to_original_index(i, v2); n, fps = video_meta(v2)
            review = grab_review(pd / "video2.mp4", i)
            mapped, how = read_frame_robust(v2, m["original_idx"]); raw = read_frame_robust(v2, i)[0] if i < n else None
            c_map, c_raw = corr(review, mapped), corr(review, raw)
            rows.append({"class": cls, "pair": pair, "annotation_idx": i, "original_fps": round(fps, 2), "mapped_idx": m["original_idx"],
                         "corr_mapped": c_map, "corr_unmapped_original_i": c_raw, "decode": how})
            side_by_side([review, mapped, raw], [f"review video2.mp4 @ {i}", f"original @ mapped {m['original_idx']} (r={c_map:.3f})" if c_map else "mapped",
                          f"original @ {i} unmapped (r={c_raw:.3f})" if c_raw else f"original @ {i}: out of range"],
                         out_dir / f"7A_{cls}_{pair[:36]}.png")
    report["7A_frame_mapping"] = rows
    print("7A frame mapping:")
    for r in rows: print(f"   {r['class']:<7} {r['pair'][:40]:<40} i={r['annotation_idx']:>4} -> {r['mapped_idx']:>4} @ {r['original_fps']:>6} fps  corr mapped {r['corr_mapped']:.3f}  unmapped {('%.3f' % r['corr_unmapped_original_i']) if r['corr_unmapped_original_i'] is not None else 'n/a'}")


def audit_7b(manifest: dict, out_dir: Path, report: dict, n: int, seed: int):
    from run_scenediff_batch import read_frame_robust
    from scenediff_gt_eval import load_query_gt
    import cv2
    qs = [q for q in manifest["queries"] if not q["gt_empty"]]
    random.Random(seed).shuffle(qs)
    rows = []
    for q in qs[:n]:
        pd = BENCH / "data" / q["pair"]
        frame, how = read_frame_robust(q["video2_path"], q["t1_original_idx"])
        gt = load_query_gt(pd, q["t1_annotation_idx"])
        lab = gt.label
        if frame.shape[:2] != lab.shape:
            lab = cv2.resize(lab.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        over = frame.copy()
        over[lab == 1] = (0.45 * over[lab == 1] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)   # added: green
        over[lab == 2] = (0.45 * over[lab == 2] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)   # moved bucket: blue (BGR)
        objs = ", ".join(f"{o['obj_id']}:{o['status']}" for o in gt.objects)
        side_by_side([frame, over], [f"{q['pair'][:30]} t1={q['t1_annotation_idx']} (orig {q['t1_original_idx']}, {how})", f"GT overlay: {objs[:60]}"],
                     out_dir / f"7B_{q['pair'][:36]}_t1_{q['t1_annotation_idx']}.png")
        rows.append({"pair": q["pair"], "t1_annotation_idx": q["t1_annotation_idx"], "t1_original_idx": q["t1_original_idx"],
                     "gt_objects": [(o["obj_id"], o["status"], o["frame_used"]) for o in gt.objects], "gt_pixels": int((lab > 0).sum()),
                     "frame_shape": list(frame.shape[:2]), "gt_shape": list(gt.label.shape), "decode": how})
    report["7B_gt_alignment"] = rows
    print(f"7B GT alignment: {len(rows)} overlays written")


def audit_7c(manifest: dict, pairing: dict, out_dir: Path, report: dict, n: int, seed: int):
    from run_scenediff_batch import read_frame_robust
    qs = list(manifest["queries"]); random.Random(seed + 1).shuffle(qs)
    rows = []
    for q in qs[:n]:
        pair = q["pair"]; rec = pairing[pair]
        t0 = next(f for f in rec["t0_frames"] if f["annotation_idx"] == q["t0_source_annotation_idx"])
        t1 = next(f for f in rec["t1_frames"] if f["annotation_idx"] == q["t1_annotation_idx"])
        a = read_frame_robust(q["video1_path"], t0["original_idx"])[0]; b = read_frame_robust(q["video2_path"], t1["original_idx"])[0]
        side_by_side([a, b], [f"BEFORE {pair[:26]} annot {t0['annotation_idx']} (orig {t0['original_idx']})",
                              f"AFTER annot {t1['annotation_idx']} (orig {t1['original_idx']})  co-vis {q['covisibility']:.3f} {'>0.5' if q['above_threshold'] else 'argmax'}"],
                     out_dir / f"7C_{pair[:36]}_{t0['annotation_idx']}_{t1['annotation_idx']}.png")
        rows.append({"pair": pair, "before_annotation_idx": t0["annotation_idx"], "after_annotation_idx": t1["annotation_idx"],
                     "covisibility": q["covisibility"], "above_threshold": q["above_threshold"], "n_selected_pairs_in_sequence": len(rec["selections"])})
    report["7C_pairing"] = rows
    print(f"7C pairing: {len(rows)} side-by-sides written; co-vis range {min(r['covisibility'] for r in rows):.3f}-{max(r['covisibility'] for r in rows):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pairing", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text()); pairing = json.loads(args.pairing.read_text())
    report = {}
    audit_7a(args.out_dir, report)
    audit_7b(manifest, args.out_dir, report, args.n, args.seed)
    audit_7c(manifest, pairing, args.out_dir, report, args.n, args.seed)
    (args.out_dir / "audit_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out_dir / 'audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
