#!/usr/bin/env python3
"""Representative examples for a PASLCD variant comparison: the queries
whose IoU moved most between two variants, rendered as TP/FP/FN overlays
side by side (left = variant A, right = variant B) so the mechanism behind
an aggregate delta can be checked by eye rather than assumed.

Usage: paslcd_variant_examples.py v0_baseline v3_no_visibility_filter [--k 4]
Writes results/paslcd_suppression_ablation/examples_<A>_vs_<B>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ocmask_pipeline.metrics import load_paslcd_gt  # noqa: E402

ROOT = REPO / "results/paslcd_suppression_ablation"
CACHE = REPO / "results/paslcd_ablation_cache"
DATA = REPO / "data/PASLCD"
TILE = (400, 300)


def overlay(variant: str, key: str) -> tuple[Image.Image, dict]:
    scene, stem = key.split("/")
    ds, inst = scene.rsplit("_Instance_", 1)
    gt = load_paslcd_gt(DATA / ds / f"Instance_{inst}" / "gt_mask" / f"{stem}.png")
    lab = cv2.imread(str(ROOT / variant / scene / stem / "detect" / "labels.png"), cv2.IMREAD_GRAYSCALE)
    pred = cv2.resize(lab, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) != 0
    q = np.array(Image.open(CACHE / scene / "intermediate" / stem / "reconstruction" / "image_t1.png").convert("RGB"))
    q = cv2.resize(q, (gt.shape[1], gt.shape[0])).astype(float)
    g = gt > 0
    for m, c in (((g & pred), (0, 255, 0)), ((~g & pred), (255, 40, 40)), ((g & ~pred), (60, 120, 255))):
        q[m] = 0.35 * q[m] + 0.65 * np.array(c)
    info = json.loads((ROOT / variant / scene / stem / "detect" / "inference.json").read_text())
    return Image.fromarray(q.astype(np.uint8)).resize(TILE), info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--k", type=int, default=4, help="examples per direction (largest gain, largest loss)")
    args = ap.parse_args()
    ea = json.loads((ROOT / args.a / "eval.json").read_text())["per_query"]
    eb = json.loads((ROOT / args.b / "eval.json").read_text())["per_query"]
    deltas = sorted(((eb[k]["iou"] - ea[k]["iou"], k) for k in eb if k in ea))
    picks = [("B worse", d, k) for d, k in deltas[: args.k]] + [("B better", d, k) for d, k in deltas[-args.k:][::-1]]

    rows = []
    for tag, d, k in picks:
        ia, infa = overlay(args.a, k); ib, infb = overlay(args.b, k)
        row = Image.new("RGB", (TILE[0] * 2 + 10, TILE[1] + 34), (16, 16, 16))
        row.paste(ia, (0, 34)); row.paste(ib, (TILE[0] + 10, 34))
        dr = ImageDraw.Draw(row)
        ca, cb = infa.get("decision_counts", {}), infb.get("decision_counts", {})
        dr.text((4, 2), f"{tag} ΔIoU {d:+.3f}  {k}", fill=(255, 255, 0))
        dr.text((4, 18), f"{args.a}: IoU {ea[k]['iou']:.3f} P {ea[k]['precision']:.2f} R {ea[k]['recall']:.2f} "
                         f"add {ca.get('added', 0)} rem {ca.get('removed', 0)} rep {ca.get('replaced', 0)} "
                         f"visrej {infa.get('visibility_filter_rejected', 0)} hor {infa.get('horizon_suppressed', 0)}", fill=(200, 200, 200))
        dr.text((TILE[0] + 14, 18), f"{args.b}: IoU {eb[k]['iou']:.3f} P {eb[k]['precision']:.2f} R {eb[k]['recall']:.2f} "
                                    f"add {cb.get('added', 0)} rem {cb.get('removed', 0)} rep {cb.get('replaced', 0)} "
                                    f"visrej {infb.get('visibility_filter_rejected', 0)} hor {infb.get('horizon_suppressed', 0)}", fill=(200, 200, 200))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 16), (16, 16, 16))
    ImageDraw.Draw(sheet).text((4, 2), "green = correct   red = false positive   blue = missed", fill=(255, 255, 255))
    y = 16
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    out = ROOT / f"examples_{args.a}_vs_{args.b}.png"
    sheet.save(out)
    print(f"wrote {out}")
    for tag, d, k in picks:
        print(f"  {tag:9s} {d:+.3f} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
