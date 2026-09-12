#!/usr/bin/env python3
"""Per-query SAM3 grounded-text MOVABLE-object masks for a SceneDiff run
(port of scripts/paslcd_surface_masks.py to the SceneDiff layout, 2026-09-12).

Walks <root>/<pair>/t1_XXXX/shared/render/{image_t1,render_t0}.png -- the RAW
render_t0, which is the right frame for the refine-OFF baseline (no refined/
fallback here, deliberately) -- runs Sam3TextPromptDetector for every prompt in
--prompts on both frames, and writes, per query:

  <out>/<pair>/t1_XXXX/<prompt>.npy              union of detections on image_t1
  <out>/<pair>/t1_XXXX/<prompt>__render_t0.npy   same on render_t0
  <out>/<pair>/t1_XXXX/movable_union.npy         union over ALL prompts x both
                                                 frames -- the file the gate reads
                                                 (manifest key movable_object_mask)

plus coverage.csv (per query/prompt/frame: detection count, covered fraction)
and union_coverage.csv (per query: fraction of the frame the whitelist covers).
The per-query union coverage is what to check FIRST when a pair collapses under
the gate: a whitelist covering < ~2% of the frame rejects nearly everything, and
on PASLCD that -- not the gate logic -- was the entire Playground failure.

Both frames matter: REMOVED objects exist only in render_t0, so an image_t1-only
whitelist rejects every REMOVED decision.

Runs in the detection env (goldilocs) with SAM3_SOURCE/SAM3_IMAGE_CHECKPOINT
exported, same as detect_batch.py. Resumable: existing .npy files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_PROMPTS = ["item", "household item", "small object", "object",
                   "door", "drawer", "cabinet door", "cabinet drawer"]


def safe_name(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="<results>/.../SceneDiff directory holding <pair>/t1_XXXX/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--pairs-file", type=Path, default=None, help="restrict to these pair ids (one per line)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N queries (smoke tests)")
    args = ap.parse_args()

    from ocmask_pipeline.stages.sam3_proposals import Sam3TextPromptDetector

    detector = Sam3TextPromptDetector(
        os.environ["SAM3_IMAGE_CHECKPOINT"], source=os.environ["SAM3_SOURCE"],
        confidence_threshold=args.confidence_threshold,
    )

    keep = set(args.pairs_file.read_text().split()) if args.pairs_file else None
    pair_dirs = sorted(p for p in args.root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if keep is not None:
        pair_dirs = [p for p in pair_dirs if p.name in keep]

    args.out.mkdir(parents=True, exist_ok=True)
    cov_path, ucov_path = args.out / "coverage.csv", args.out / "union_coverage.csv"
    cov = open(cov_path, "a", newline="")
    ucov = open(ucov_path, "a", newline="")
    w, uw = csv.writer(cov), csv.writer(ucov)
    if cov_path.stat().st_size == 0:
        w.writerow(["pair", "query", "prompt", "frame", "n_detections", "coverage_fraction", "seconds"])
    if ucov_path.stat().st_size == 0:
        uw.writerow(["pair", "query", "union_coverage_fraction", "H", "W"])

    frames = {"image_t1": "", "render_t0": "__render_t0"}
    n_done = 0
    t_all = time.perf_counter()
    try:
        for pair_dir in pair_dirs:
            for qdir in sorted(p for p in pair_dir.glob("t1_*") if p.is_dir()):
                if args.limit is not None and n_done >= args.limit:
                    break
                render_dir = qdir / "shared" / "render"
                if not (render_dir / "image_t1.png").exists() or not (render_dir / "render_t0.png").exists():
                    continue
                out_dir = args.out / pair_dir.name / qdir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                union = None
                for frame, suffix in frames.items():
                    image = np.asarray(Image.open(render_dir / f"{frame}.png").convert("RGB"))
                    H, W = image.shape[:2]
                    if union is None:
                        union = np.zeros((H, W), dtype=bool)
                    for prompt in args.prompts:
                        target = out_dir / f"{safe_name(prompt)}{suffix}.npy"
                        if target.exists():
                            mask = np.load(target)
                        else:
                            t0 = time.perf_counter()
                            mask = np.zeros((H, W), dtype=bool)
                            dets = detector.detect(image, prompt)
                            for m, _score in dets:
                                m = np.asarray(m, dtype=bool)
                                if m.shape != (H, W):
                                    m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
                                mask |= m
                            np.save(target, mask)
                            w.writerow([pair_dir.name, qdir.name, prompt, frame, len(dets), f"{mask.mean():.5f}", f"{time.perf_counter() - t0:.2f}"])
                            cov.flush()
                        if mask.shape != union.shape:  # render_t0 and image_t1 share one grid; guard anyway
                            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((union.shape[1], union.shape[0]), Image.NEAREST)) > 0
                        union |= mask
                np.save(out_dir / "movable_union.npy", union)
                uw.writerow([pair_dir.name, qdir.name, f"{union.mean():.5f}", union.shape[0], union.shape[1]])
                ucov.flush()
                n_done += 1
                if n_done % 25 == 0:
                    print(f"[{n_done} queries] {time.perf_counter() - t_all:.0f}s", flush=True)
            if args.limit is not None and n_done >= args.limit:
                break
    finally:
        detector.release()
        cov.close()
        ucov.close()
    print(f"done: {n_done} queries, {len(args.prompts)} prompts x 2 frames -> {args.out}")
    print("MOVABLE_MASKS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
