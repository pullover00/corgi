#!/usr/bin/env python3
"""Fix the single t1 query frame per diagnostic pair, BEFORE any run.

Rule (data availability only, never model output):
  1. candidates = every video2 frame index an in_video2 annotation names
  2. choose the candidate with the most in-scope GT objects whose masks are
     actually decodable at that frame (load_query_gt), ties broken by most
     GT pixels, then lowest index
  3. removed-only pairs have no in-scope GT at any frame: use the frame at
     the same relative position through video2 as the annotators'
     representative t0 frame is through video1, so the query plausibly
     views the vacated location

By default writes data/scenediff_benchmark/diagnostic_subset_queries.json for
the diagnostic subset, which run_scenediff_diagnostic.py reads. Pass
--pair-ids-file / --out to fix queries for another pair list under the SAME
rule (added 2026-09-10 for the 250-pair held-out test-split run, so every
pair's query is chosen by the pre-registered data-availability rule instead
of the runner's most-common-frame fallback). Needs pycocotools (goldilocs env).
"""
import argparse, json, pickle, sys
from pathlib import Path
import cv2
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from scenediff_gt_eval import load_query_gt
from run_scenediff_batch import (VIS_FPS, annotation_to_original_index, representative_frame_index,
                                 resolve_original_video, video_meta)

BENCH = Path("data/scenediff_benchmark")
def nframes(p):
    c = cv2.VideoCapture(str(p)); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release(); return n


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pair-ids-file", type=Path, default=BENCH / "diagnostic_subset.txt")
ap.add_argument("--out", type=Path, default=BENCH / "diagnostic_subset_queries.json")
args = ap.parse_args()

out = {}
for pair in args.pair_ids_file.read_text().split():
    pd = BENCH / "data" / pair
    objs = pickle.loads((pd / "segments.pkl").read_bytes())["objects"]
    cands = sorted({int(o["video2_frame_idx"]) for o in objs if o.get("in_video2") and int(o.get("video2_frame_idx", -1)) >= 0})
    best = None
    for t1 in cands:
        try:
            gt = load_query_gt(pd, t1)
        except Exception:
            continue
        key = (len(gt.objects), int((gt.label > 0).sum()), -t1)
        if best is None or key > best[0]:
            best = (key, t1, int((gt.label == 1).sum()), int((gt.label == 2).sum()), len(gt.objects))
    if best:
        out[pair] = {"t1_idx": best[1], "rule": "max_usable_gt_objects", "n_gt_objects": best[4],
                     "added_px": best[2], "moved_bucket_px": best[3], "candidates": cands}
        m = annotation_to_original_index(best[1], resolve_original_video(pd, 2))
        out[pair].update({"t1_idx_original": m["original_idx"], "video2_original_n": m["original_n"],
                          "video2_original_fps": m["original_fps"], "fps_ratio": m["ratio"]})
    else:
        t0_rep = representative_frame_index(objs, "in_video1", "video1_frame_idx")
        n1, n2 = nframes(resolve_original_video(pd, 1)), nframes(resolve_original_video(pd, 2))
        t1 = max(0, min(n2 - 1, round(t0_rep / max(n1 - 1, 1) * (n2 - 1))))
        # t1 here is already an original_video2 frame index (computed from original frame counts)
        n2m, fps2 = video_meta(resolve_original_video(pd, 2))
        out[pair] = {"t1_idx": t1, "rule": "relative_position_of_t0_rep (no in-scope GT at any frame)",
                     "n_gt_objects": 0, "added_px": 0, "moved_bucket_px": 0, "t0_rep": t0_rep, "n1": n1, "n2": n2,
                     "t1_idx_original": t1, "video2_original_n": n2m, "video2_original_fps": round(fps2, 3),
                     "fps_ratio": round(fps2 / VIS_FPS, 4)}
    r = out[pair]
    print(f"{pair:52s} t1={r['t1_idx']:4d}  gt_objs={r['n_gt_objects']}  added_px={r['added_px']:7d}  moved_px={r['moved_bucket_px']:7d}  [{r['rule']}]")
args.out.write_text(json.dumps(out, indent=2))
print(f"\nwrote {args.out} ({sum(1 for r in out.values() if r['n_gt_objects']>0)}/{len(out)} evaluable)")
