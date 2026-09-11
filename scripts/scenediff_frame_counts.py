#!/usr/bin/env python3
"""Cache the TRUE decodable frame count of every SceneDiff original video.

cv2's CAP_PROP_FRAME_COUNT overreports on 7 of the 250 test-split videos --
one claims 194 frames and decodes 73 -- so indices derived from it can name
frames that do not exist. The true count is corroborated by the review video:
true_count * 30 / fps matches video{1,2}.mp4's length within a few frames,
while the metadata count does not. Counting requires a full sequential decode,
so results are cached here and reused by the manifest builder and the runner.

    python scripts/scenediff_frame_counts.py --pair-ids-file <list> [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
BENCH = REPO / "data/scenediff_benchmark"
DEFAULT_OUT = BENCH / "frame_counts.json"


def true_frame_count(path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    meta = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    n = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        n += 1
    capture.release()
    return {"meta_frame_count": meta, "true_frame_count": n, "fps": round(fps, 4),
            "metadata_overreports": n < meta}


def main() -> int:
    from run_scenediff_batch import resolve_original_video

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-ids-file", type=Path, action="append", required=True,
                    help="may be repeated (e.g. val and test lists)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    pairs = []
    for f in args.pair_ids_file:
        pairs += [l.strip() for l in f.read_text().splitlines() if l.strip()]
    pairs = sorted(set(pairs))

    cache = json.loads(args.out.read_text()) if args.out.exists() else {}
    n_new = 0
    for i, pair in enumerate(pairs, 1):
        if pair in cache:
            continue
        pair_dir = BENCH / "data" / pair
        rec = {}
        for k in (1, 2):
            video = resolve_original_video(pair_dir, k)
            rec[f"video{k}"] = {"path": str(video), **true_frame_count(video)}
        cache[pair] = rec
        n_new += 1
        if n_new % 25 == 0:
            print(f"  [{i}/{len(pairs)}] counted {n_new} new", flush=True)
            args.out.write_text(json.dumps(cache, indent=1, sort_keys=True))
    args.out.write_text(json.dumps(cache, indent=1, sort_keys=True))

    bad = [(p, k) for p, r in cache.items() for k in ("video1", "video2") if r[k]["metadata_overreports"]]
    print(f"wrote {args.out}: {len(cache)} pairs ({n_new} newly counted)")
    print(f"videos where metadata overreports the true count: {len(bad)}")
    for p, k in bad:
        r = cache[p][k]
        print(f"   {p[:44]:<44} {k} meta={r['meta_frame_count']:>5} true={r['true_frame_count']:>5} fps={r['fps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
