#!/usr/bin/env python3
"""Build and freeze the query manifest for the SceneDiff-paired single-query protocol.

Reads the official co-visibility pairing (scenediff_covis_pairing.py output) and,
for each sequence pair, ranks the selected after-frames by their highest
bidirectional co-visibility with any before-frame -- SceneDiff's rule (2),
"the after frame with the highest co-visibility", applied per after-frame --
and keeps the top K (K=1 for protocol P1). Selection is purely geometric; the
annotations are consulted only to RECORD which GT objects are visible, never
to choose.

Writes two files:
  <out>.json                 the frozen manifest, one record per query, with
                             every field of the protocol brief's section 6
  <out>.runner_queries.json  the same queries in run_scenediff_diagnostic.py's
                             --queries-file format (t1_idx = annotation index,
                             t1_idx_original = the real frame to extract)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
BENCH = REPO / "data/scenediff_benchmark"


def subset_of(pair: str, split_json: dict) -> str:
    for cat, members in split_json.items():
        if pair in members:
            return {"varied": "SD-V", "kitchen": "SD-K"}.get(cat, cat)
    return "?"


def main() -> int:
    from run_scenediff_batch import (annotation_to_original_index, read_frame_robust,
                                     representative_frame_index, resolve_original_video,
                                     sample_frame_indices, video_meta)
    from scenediff_gt_eval import load_query_gt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairing", type=Path, required=True, help="covis_pairing_{val,test}.json")
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--frames-per-video", type=int, default=10, help="CORGI T0 reference count (config value)")
    ap.add_argument("--pair-ids-file", type=Path, default=None, help="restrict to these pairs (e.g. a smoke subset)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairing = json.loads(args.pairing.read_text())
    meta = pairing.get("_meta", {})
    split_json = json.loads((BENCH / "splits" / f"{args.split}_split.json").read_text())
    pairs = [k for k in pairing if not k.startswith("_")]
    if args.pair_ids_file:
        keep = {l.strip() for l in args.pair_ids_file.read_text().splitlines() if l.strip()}
        pairs = [p for p in pairs if p in keep]

    manifest, runner_queries = [], {}
    for pair in sorted(pairs):
        rec = pairing[pair]
        pair_dir = BENCH / "data" / pair
        video1, video2 = resolve_original_video(pair_dir, 1), resolve_original_video(pair_dir, 2)
        n1, fps1 = video_meta(video1)
        n2, fps2 = video_meta(video2)

        # rank after-frames by their best co-visibility with any before-frame
        best = {}
        for s in rec["selections"]:
            t1 = s["t1_annotation_idx"]
            if t1 not in best or s["covisibility"] > best[t1]["covisibility"]:
                best[t1] = s
        ranked = sorted(best.values(), key=lambda s: -s["covisibility"])[: args.top_k]

        # CORGI's T0 reference: uniform frames of the ORIGINAL before video plus the
        # annotators' representative before frame (runner behaviour, unchanged)
        objects = pickle.loads((pair_dir / "segments.pkl").read_bytes())["objects"]
        t0_rep_annot = representative_frame_index(objects, "in_video1", "video1_frame_idx")
        t0_rep_orig = annotation_to_original_index(t0_rep_annot, video1)["original_idx"]
        t0_orig = sample_frame_indices(n1, args.frames_per_video)
        if t0_rep_orig not in t0_orig:
            t0_orig = sorted(t0_orig + [t0_rep_orig])
        t0_decode = []
        for idx in t0_orig:
            frame, how = read_frame_robust(video1, idx)
            t0_decode.append(how if frame is not None else "failed")

        for rank, s in enumerate(ranked, 1):
            t1_annot, t1_orig = s["t1_annotation_idx"], s["t1_original_idx"]
            frame, t1_how = read_frame_robust(video2, t1_orig)
            if frame is None:
                t1_how = "failed"
            try:
                gt = load_query_gt(pair_dir, t1_annot)
                gt_ids = [o["obj_id"] for o in gt.objects]
                gt_status = {o["obj_id"]: o["status"] for o in gt.objects}
            except ValueError:
                gt_ids, gt_status = [], {}
            manifest.append({
                "split": args.split, "subset": subset_of(pair, split_json), "pair": pair,
                "protocol": f"covis_top{args.top_k}", "rank": rank,
                "t0_source_annotation_idx": s["t0_annotation_idx"],
                "t0_source_original_idx": s["t0_original_idx"],
                "t1_annotation_idx": t1_annot, "t1_original_idx": t1_orig,
                "video2_original_fps": round(fps2, 4), "video2_true_frame_count": n2,
                "covisibility": s["covisibility"], "above_threshold": s["above_threshold"],
                "selection_rule": s["selection_rule"],
                "t0_reference_original_indices": t0_orig,
                "t0_reference_annotation_indices": [int(round(i * 30.0 / fps1)) for i in t0_orig],
                "video1_original_fps": round(fps1, 4), "video1_true_frame_count": n1,
                "t0_reference_decode": t0_decode,
                "gt_object_ids": gt_ids, "gt_object_status": gt_status, "gt_empty": len(gt_ids) == 0,
                "t1_decode": t1_how,
                "video1_path": str(video1), "video2_path": str(video2),
            })
            if rank == 1:
                runner_queries[pair] = {"t1_idx": t1_annot, "t1_idx_original": t1_orig,
                                        "rule": f"covis_top{args.top_k}_rank1", "covisibility": s["covisibility"],
                                        "n_gt_objects": len(gt_ids)}

    out = {"_meta": {**meta, "split": args.split, "top_k": args.top_k, "frames_per_video": args.frames_per_video,
                     "n_sequence_pairs": len(pairs),
                     "n_raw_selected_pairs": sum(len(pairing[p]["selections"]) for p in pairs),
                     "n_unique_t1_all_density": sum(len(pairing[p]["unique_t1"]) for p in pairs),
                     "n_queries": len(manifest),
                     "n_nonempty_gt": sum(1 for m in manifest if not m["gt_empty"]),
                     "n_empty_gt": sum(1 for m in manifest if m["gt_empty"]),
                     "n_decode_failures": sum(1 for m in manifest if m["t1_decode"] == "failed" or "failed" in m["t0_reference_decode"]),
                     "n_sequential_fallback": sum(1 for m in manifest if m["t1_decode"] == "sequential_fallback" or "sequential_fallback" in m["t0_reference_decode"])},
           "queries": manifest}
    args.out.write_text(json.dumps(out, indent=1))
    Path(str(args.out).replace(".json", ".runner_queries.json")).write_text(json.dumps(runner_queries, indent=2))
    m = out["_meta"]
    print(f"wrote {args.out}")
    print(f"  sequence pairs {m['n_sequence_pairs']} | raw selected pairs {m['n_raw_selected_pairs']} | unique T1 at full density {m['n_unique_t1_all_density']}")
    print(f"  queries (top-{args.top_k}) {m['n_queries']} | non-empty GT {m['n_nonempty_gt']} | empty GT {m['n_empty_gt']}")
    print(f"  decode failures {m['n_decode_failures']} | sequential fallbacks {m['n_sequential_fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
