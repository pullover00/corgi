#!/usr/bin/env python3
"""Stage 2 (optional): DI2FIX render refinement. Run in the difix3d conda env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-t0", type=Path, required=True)
    parser.add_argument("--clean-render", type=Path, required=True)
    parser.add_argument("--image-t1", type=Path, required=True, help="the real target photo, used as the reference image")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.refine import refine_renders

    config = load_config(args.config)
    render_t0 = np.asarray(Image.open(args.render_t0).convert("RGB"))
    clean_render = np.asarray(Image.open(args.clean_render).convert("RGB"))
    image_t1 = np.asarray(Image.open(args.image_t1).convert("RGB"))

    fixed_render_t0, fixed_clean_render = refine_renders(render_t0, clean_render, image_t1, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fixed_render_t0).save(args.output_dir / "render_t0.png")
    Image.fromarray(fixed_clean_render).save(args.output_dir / "clean_render.png")
    print(f"wrote {args.output_dir / 'render_t0.png'}")
    print(f"wrote {args.output_dir / 'clean_render.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
