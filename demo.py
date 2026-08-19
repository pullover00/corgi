#!/usr/bin/env python3
"""Run the object-consistent-masks method on one image pair.

Usage:

    python demo.py --before before.png --after after.png --output out/

``--before`` is a photo of a place at an earlier time ("source"/"image0" in
the code and config); ``--after`` is a photo of the same place, from
approximately the same viewpoint, at a later time ("target"/"image1") --
the pixel grid the output change map is aligned to. The method needs GPU
compute and the model checkpoints/environment variables described in
README.md's "Setup" section (MASt3R, SAM2, DINOv2, and an external SAM3/
SAM3.1 checkout); run ``ocmask doctor`` first if you are not sure your
environment is ready.

Writes, under ``--output``:

  labels.png              the headline result: the object-consistent
                           full-mask prediction (one of Label's IDs per
                           pixel: 0 unchanged, 1 added, 2 removed, 3 moved,
                           5 replaced -- 4/warped never appears in a final
                           prediction)
  labels_guarded.png      the same decision at a more conservative
                           rasterization footprint (see README.md)
  labels_color.png        labels.png rendered with a fixed color palette
  overlay.png             labels_color.png alpha-blended over the "after"
                           image, for quick visual inspection
  target.png              the "after" image at the resolution labels.png
                           is aligned to
  inference.json           per-stage diagnostics and timings
  <hash>/                  stage 1's own artifacts (reconstruction.npz,
                           the rendered "before"-into-"after" view, etc.),
                           named by ocmask.io.pair_key
  03_tracking/             stage 3's re-tracked baseline raster and
                           tracking diagnostics
  11_object_consistent_masks/  the final refinement's own decision trail
                           (inference.json) and intermediate rasters
  These per-stage subdirectories are useful for debugging one pair in
  depth; only the top-level files above are needed for normal use.

Exit status is nonzero if inference raised (e.g. missing checkpoints/
environment variables, or a corrupt input image).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True, metavar="IMAGE", help="earlier-time photo ('source'/image0)")
    parser.add_argument("--after", required=True, metavar="IMAGE", help="later-time photo ('target'/image1); output is aligned to this image")
    parser.add_argument("--output", required=True, metavar="DIR", help="directory to write predictions and diagnostics into")
    parser.add_argument("--config", default=str(REPOSITORY / "configs/pipeline.yaml"), help="pipeline config (default: configs/pipeline.yaml)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ocmask.config import load_config
    from ocmask.model_paths import configure_mast3r_paths

    configure_mast3r_paths()
    config = load_config(args.config)

    from ocmask.inference import run_pair

    result = run_pair(args.before, args.after, args.output, config)

    changed_fraction = float((result.labels != 0).mean())
    print(
        json.dumps(
            {
                "output": str(result.artifacts_dir.resolve()),
                "changed_pixel_fraction": changed_fraction,
                "timings_seconds": result.timings,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
