#!/usr/bin/env python3
"""Select the SceneDiff diagnostic subset by criteria fixed BEFORE any run.

Nothing here reads a prediction from the variant under test. The only
performance-informed input is a small, explicitly declared quota drawn from
the *previous* shipped30 run (worst-2 / best-2 by its pixel IoU), which the
experiment brief asks for so known failure and success cases are revisited;
that quota is applied by a fixed rule and labelled as such in the output.

Criteria, applied in this order (deterministic, seed 0):

  pool      every annotated pair with both original videos on disk
  quota     2 lowest-IoU and 2 highest-IoU pairs of the prior shipped30
            evaluation (pxim_iou_t1only.json), ties broken by name
  coverage  fill the remaining slots so that, across the whole subset:
              - every change type occurs: added-only, removed-only,
                moved-bucket-only, and mixed pairs
              - no scene type holds more than 2 slots (kitchen is 50% of
                the pool and would otherwise dominate)
              - both difficulty bands occur: easy = exactly 1 annotated
                object, hard = >= 3 annotated objects
              - both rigid and deformable objects occur
  size      10 pairs total

"Difficulty" is defined by annotation count only, never by score. Viewpoint
spread is not annotated in SceneDiff, so it is not a criterion here; the
frame counts of each video are recorded per pair for the report instead.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data/scenediff_benchmark"
PRIOR = REPO / "results/scenediff_shipped30_refined/pxim_iou_t1only.json"


def change_kind(objects) -> str:
    kinds = set()
    for o in objects:
        a, b = bool(o["in_video1"]), bool(o["in_video2"])
        kinds.add("removed" if a and not b else "added" if b and not a else "moved_bucket")
    return next(iter(kinds)) + "_only" if len(kinds) == 1 else "mixed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--max-per-scene-type", type=int, default=2)
    ap.add_argument("--out", type=Path, default=BENCH / "diagnostic_subset.txt")
    args = ap.parse_args()

    pool = {}
    for pair_dir in sorted(p for p in (BENCH / "data").iterdir() if p.is_dir()):
        if not (pair_dir / "inference_data.pkl").exists():
            continue
        videos = [n for n in pair_dir.iterdir() if n.stem in ("original_video1", "original_video2")]
        if len(videos) < 2:
            continue
        d = pickle.loads((pair_dir / "inference_data.pkl").read_bytes())
        objs = d["objects"]
        if not objs:
            continue
        pool[pair_dir.name] = {
            "scene_type": (d.get("scene_type") or "unknown").lower().replace(" ", "_"),
            "change_kind": change_kind(objs),
            "n_objects": len(objs),
            "difficulty": "easy" if len(objs) == 1 else "hard" if len(objs) >= 3 else "medium",
            "deformability": Counter(o.get("deformability", "unknown") for o in objs).most_common(1)[0][0],
            "frames": [len(d["video1"]["frame_names"]), len(d["video2"]["frame_names"])],
        }
    print(f"pool: {len(pool)} pairs")

    chosen: dict[str, str] = {}  # pair -> reason

    prior = json.loads(PRIOR.read_text())["per_pair"] if PRIOR.exists() else {}
    scored = sorted(((v if isinstance(v, (int, float)) else v.get("iou")), k) for k, v in prior.items()
                    if k in pool and (v if isinstance(v, (int, float)) else v.get("iou")) is not None)
    for iou, k in scored[:2]:
        chosen[k] = f"quota: prior worst (IoU {iou:.3f})"
    for iou, k in scored[-2:]:
        chosen.setdefault(k, f"quota: prior best (IoU {iou:.3f})")

    def covered(key):
        return Counter(pool[k][key] for k in chosen)

    rng = random.Random(0)
    remaining = [k for k in pool if k not in chosen]
    rng.shuffle(remaining)

    def take(predicate, reason):
        for k in remaining:
            if k in chosen or not predicate(pool[k]):
                continue
            if covered("scene_type")[pool[k]["scene_type"]] >= args.max_per_scene_type:
                continue
            chosen[k] = reason
            return True
        return False

    # coverage passes, each only fires if the subset is still missing that stratum
    for kind in ("added_only", "removed_only", "moved_bucket_only", "mixed"):
        if covered("change_kind")[kind] == 0 and len(chosen) < args.size:
            take(lambda m, kind=kind: m["change_kind"] == kind, f"coverage: change kind {kind}")
    for band in ("easy", "hard"):
        if covered("difficulty")[band] == 0 and len(chosen) < args.size:
            take(lambda m, band=band: m["difficulty"] == band, f"coverage: difficulty {band}")
    for deform in ("rigid", "deformable"):
        if covered("deformability")[deform] == 0 and len(chosen) < args.size:
            take(lambda m, d=deform: m["deformability"] == d, f"coverage: {deform} object")
    # top up with unseen scene types first, then anything under the per-type cap
    while len(chosen) < args.size:
        seen = covered("scene_type")
        if not take(lambda m: seen[m["scene_type"]] == 0, "coverage: new scene type"):
            if not take(lambda m: True, "fill"):
                break

    rows = [{"pair": k, "reason": chosen[k], **pool[k]} for k in chosen]
    args.out.write_text("\n".join(r["pair"] for r in rows) + "\n")
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print(f"\nselected {len(rows)} pairs -> {args.out}\n")
    print(f"{'pair':52s} {'scene':14s} {'kind':18s} {'n':>2s} {'diff':6s} {'deform':10s} reason")
    for r in rows:
        print(f"{r['pair']:52s} {r['scene_type']:14s} {r['change_kind']:18s} {r['n_objects']:2d} "
              f"{r['difficulty']:6s} {r['deformability']:10s} {r['reason']}")
    print("\ncoverage:", {k: dict(Counter(r[k] for r in rows)) for k in ("change_kind", "scene_type", "difficulty", "deformability")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
