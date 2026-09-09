#!/usr/bin/env python3
"""Report, per scene instance and per query, which pipeline stages actually
ran: reconstruction (render_t0/clean_render/image_t1 + position buffers),
refine (DI2FIX), and detect (labels.png). Written for the 2026-09-08
refine-completeness audit, kept as a reusable check -- run it against any
results/paslcd-style output tree after a run to confirm what was and wasn't
skipped, instead of trusting metrics.csv alone (which says nothing about
refine)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    scene_dirs = sorted(p for p in args.output_root.iterdir() if p.is_dir())
    total = {"reconstruction": 0, "refined": 0, "detect": 0, "queries": 0}

    for scene_dir in scene_dirs:
        intermediate = scene_dir / "intermediate"
        if not intermediate.exists():
            continue
        recon_n = refined_n = detect_n = n = 0
        for stem_dir in sorted(intermediate.iterdir()):
            n += 1
            if (stem_dir / "reconstruction" / "render_t0.png").exists():
                recon_n += 1
            if (stem_dir / "refined" / "render_t0.png").exists():
                refined_n += 1
            if (stem_dir / "detect" / "labels.png").exists():
                detect_n += 1
        total["queries"] += n
        total["reconstruction"] += recon_n
        total["refined"] += refined_n
        total["detect"] += detect_n
        flag = "" if refined_n == n else "  <-- refine incomplete/missing"
        print(f"{scene_dir.name:28s} queries={n:4d}  reconstruction={recon_n:4d}  refined={refined_n:4d}  detect={detect_n:4d}{flag}")

    print("-" * 90)
    print(f"{'TOTAL':28s} queries={total['queries']:4d}  reconstruction={total['reconstruction']:4d}  "
          f"refined={total['refined']:4d}  detect={total['detect']:4d}")
    if total["refined"] < total["queries"]:
        print(f"\n{total['queries'] - total['refined']} / {total['queries']} queries are missing the refine stage.")


if __name__ == "__main__":
    main()
