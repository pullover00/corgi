#!/usr/bin/env python3
"""Stage 1: VGGT-Omega reconstruction -> render_t0.png / clean_render.png /
image_t1.png. Run in the vggt-omega conda env."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0-frames", nargs="+", required=True, help="T0 (before) frame image paths")
    parser.add_argument("--t1-frames", nargs="+", required=True, help="T1 (after) frame image paths")
    parser.add_argument("--t0-reference-index", type=int, default=0, help="index into --t0-frames used for the opposing-view depth check")
    parser.add_argument("--t1-reference-index", type=int, default=0, help="index into --t1-frames rendered into / used as image_t1")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from PIL import Image

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import reconstruct_and_render

    config = load_config(args.config)
    result = reconstruct_and_render(args.t0_frames, args.t1_frames, args.t0_reference_index, args.t1_reference_index, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.render_t0).save(args.output_dir / "render_t0.png")
    Image.fromarray(result.clean_render).save(args.output_dir / "clean_render.png")
    Image.fromarray(result.image_t1).save(args.output_dir / "image_t1.png")

    summary = {
        "t0_frames": [str(p) for p in args.t0_frames],
        "t1_frames": [str(p) for p in args.t1_frames],
        "t0_reference_index": args.t0_reference_index,
        "t1_reference_index": args.t1_reference_index,
        "input_point_count": int(len(result.input_points)),
        "target_point_count": int(len(result.target_points)),
        "cleaned_point_count": int(len(result.cleaned_points)),
        "render_t0_coverage_fraction": result.render_t0_coverage_fraction,
        "clean_render_coverage_fraction": result.clean_render_coverage_fraction,
    }
    (args.output_dir / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
