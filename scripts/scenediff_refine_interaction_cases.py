#!/usr/bin/env python3
"""Build the qualitative panels for the refinement x reference-count factorial:
for a given (pair, N), lay out

    raw render_t0 | DI2FIX-refined render_t0 | labels OFF (raw) | labels ON (refined)

so the render-level difference and its downstream label consequence can be read
side by side. Reads only artifacts already on disk; renders nothing new.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/scenediff_diagnostic/SceneDiff"
CELLS = {1: ("scenediff_v10_no_dino_ref1", "scenediff_v10_no_dino_ref1_no_refine", "shared_ref1"),
         3: ("scenediff_v10_no_dino_ref3", "scenediff_v10_no_dino_ref3_no_refine", "shared_ref3"),
         5: ("scenediff_v10_no_dino_ref5", "scenediff_v10_no_dino_ref5_no_refine", "shared_ref5"),
         10: ("scenediff_v10_no_dino", "scenediff_v10_no_dino_no_refine", "shared")}


def query_dir(pair: str) -> Path:
    return sorted(p for p in (ROOT / pair).iterdir() if p.name.startswith("t1_"))[0]


def bar(text: str, width: int, height: int = 26) -> np.ndarray:
    img = Image.new("RGB", (width, height), (25, 25, 25))
    ImageDraw.Draw(img).text((6, 6), text, fill=(255, 255, 255))
    return np.array(img)


def build(pair: str, n: int, out_dir: Path) -> Path | None:
    exp_on, exp_off, shared = CELLS[n]
    q = query_dir(pair)
    panels, labels = [], []
    for path, name in (
        (q / shared / "render" / "render_t0.png", "raw render_t0"),
        (q / shared / "refine" / "render_t0.png", "DI2FIX-refined render_t0"),
        (q / exp_off / "labels" / "labels_color.png", f"labels refine=OFF (N={n})"),
        (q / exp_on / "labels" / "labels_color.png", f"labels refine=ON (N={n})"),
    ):
        if not path.exists():
            print(f"  missing {path}")
            continue
        panels.append(np.asarray(Image.open(path).convert("RGB")))
        labels.append(name)
    if not panels:
        return None
    h = min(p.shape[0] for p in panels)
    panels = [p[:h] for p in panels]
    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    row, bars = [], []
    for i, (p, name) in enumerate(zip(panels, labels)):
        if i:
            row.append(gap); bars.append(np.full((26, 8, 3), 255, dtype=np.uint8))
        row.append(p); bars.append(bar(name, p.shape[1]))
    combined = np.concatenate([np.concatenate(bars, axis=1), np.concatenate(row, axis=1)], axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"N{n}_{pair}.png"
    Image.fromarray(combined).save(out)

    for exp, state in ((exp_on, "ON"), (exp_off, "OFF")):
        inf = q / exp / "labels" / "inference.json"
        if inf.exists():
            d = json.loads(inf.read_text())
            print(f"  N={n} {pair} refine={state}: object_counts={d.get('object_counts')} "
                  f"decisions={d.get('decision_counts')} vis_filtered={d.get('visibility_filter_rejected')} "
                  f"recoveries={(d.get('tracking_recovery') or {}).get('tracking_recoveries')}")
    print(f"  -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", required=True, help="pair:N entries, e.g. gym_3_gym_4:1")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "_experiments" / "_interaction_cases")
    args = ap.parse_args()
    for case in args.cases:
        pair, n = case.rsplit(":", 1)
        print(f"== {pair} N={n}")
        build(pair, int(n), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
