#!/usr/bin/env python3
"""Per-prompt SAM3 grounded-text surface masks for PASLCD queries (2026-09-12).

For every query in a completed PASLCD run and every prompt in --prompts, runs
Sam3TextPromptDetector (the pipeline's own class, unchanged) on the query's
working-resolution image_t1 and saves the union of all detections scoring
>= --confidence-threshold as <out>/<Dataset>_<Instance>/<stem>/<prompt>.npy
(bool). One model load for the whole batch. Also writes coverage.csv with the
per-(query, prompt) fraction of the image covered and the detection count, so
each prompt can be inspected on its own before any prompt SET is evaluated.

Prompt sets are then just unions of these per-prompt files (see
paslcd_prompt_ablation_replay.py) -- computed once, combined freely.

Runs in the detection env (pl_detect) with SAM3_SOURCE/SAM3_IMAGE_CHECKPOINT
exported, same as detect_batch.py.
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


def safe_name(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--scenes", nargs="*", default=None, help="restrict to these <Dataset>_<Instance> dirs")
    ap.add_argument("--frames", nargs="+", default=["image_t1"], choices=["image_t1", "render_t0"],
                    help="frame(s) to run the detector on; render_t0 uses refined/ when present")
    args = ap.parse_args()

    from ocmask_pipeline.stages.sam3_proposals import Sam3TextPromptDetector

    detector = Sam3TextPromptDetector(
        os.environ["SAM3_IMAGE_CHECKPOINT"], source=os.environ["SAM3_SOURCE"],
        confidence_threshold=args.confidence_threshold,
    )

    scene_dirs = sorted(p for p in args.baseline_root.glob("*_Instance_*") if p.is_dir())
    if args.scenes:
        keep = set(args.scenes)
        scene_dirs = [p for p in scene_dirs if p.name in keep]

    args.out.mkdir(parents=True, exist_ok=True)
    cov_path = args.out / "coverage.csv"
    write_header = not cov_path.exists()
    cov = open(cov_path, "a", newline="")
    w = csv.writer(cov)
    if write_header:
        w.writerow(["scene", "stem", "prompt", "n_detections", "coverage_fraction", "seconds"])

    n_done = 0
    t_all = time.perf_counter()
    for scene_dir in scene_dirs:
        for qdir in sorted((scene_dir / "intermediate").iterdir()):
            out_dir = args.out / scene_dir.name / qdir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for frame in args.frames:
                if frame == "image_t1":
                    img_path = qdir / "reconstruction" / "image_t1.png"
                else:
                    img_path = qdir / "refined" / "render_t0.png"
                    if not img_path.exists():
                        img_path = qdir / "reconstruction" / "render_t0.png"
                if not img_path.exists():
                    continue
                image = np.asarray(Image.open(img_path).convert("RGB"))
                H, W = image.shape[:2]
                suffix = "" if frame == "image_t1" else f"__{frame}"
                for prompt in args.prompts:
                    target = out_dir / f"{safe_name(prompt)}{suffix}.npy"
                    if target.exists():
                        continue
                    t0 = time.perf_counter()
                    mask = np.zeros((H, W), dtype=bool)
                    dets = detector.detect(image, prompt)
                    for m, _score in dets:
                        m = np.asarray(m, dtype=bool)
                        if m.shape != (H, W):
                            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
                        mask |= m
                    np.save(target, mask)
                    w.writerow([scene_dir.name, qdir.name, f"{prompt}{suffix}", len(dets), f"{mask.mean():.5f}", f"{time.perf_counter() - t0:.2f}"])
                    cov.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"[{n_done} queries] {time.perf_counter() - t_all:.0f}s", flush=True)

    detector.release()
    cov.close()
    print(f"done: {n_done} queries, {len(args.prompts)} prompts -> {args.out}")
    print("SURFACE_MASKS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
