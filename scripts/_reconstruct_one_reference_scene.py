#!/usr/bin/env python3
"""Worker: reconstruct one PASLCD instance's reference scene with one method
and save it. Invoked as a subprocess (with a timeout) by
build_paslcd_reference_scenes_{vggt,mast3r}.py so one pathological instance
cannot hang the overnight batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["vggt_omega", "mast3r"], required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from ocmask_pipeline.config import load_config

    config = load_config(args.config)

    if args.method == "vggt_omega":
        from ocmask_pipeline.reconstruction import reconstruct_reference_scene
        scene = reconstruct_reference_scene(args.images, config)
    else:
        from ocmask_pipeline.reconstruction_mast3r import reconstruct_reference_scene_mast3r
        scene = reconstruct_reference_scene_mast3r(args.images, config)

    scene.save(args.output)
    print(f"OK -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
