#!/usr/bin/env python3
"""Stage 3: SAM3 + DINOv2 + SAM2-tracking object-state resolution over
render_t0 / clean_render / image_t1. Run in the detection conda env (SAM3,
SAM2, DINOv2 -- see README.md).

Needs SAM3_SOURCE and SAM3_IMAGE_CHECKPOINT set in the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-t0", type=Path, required=True)
    parser.add_argument("--clean-render", type=Path, required=True)
    parser.add_argument("--image-t1", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--render-t0-positions", type=Path, default=None,
        help="optional render_t0_positions.npy (world-position buffer) written by run_paslcd_pair.py; "
             "supply all three *-positions flags plus --scene-scale-path to activate the geometric-identity test",
    )
    parser.add_argument("--clean-render-positions", type=Path, default=None)
    parser.add_argument("--image-t1-positions", type=Path, default=None)
    parser.add_argument("--scene-scale-path", type=Path, default=None, help="optional scene_scale.json written by run_paslcd_pair.py")
    parser.add_argument(
        "--render-t0-coverage", type=Path, default=None,
        help="optional render_t0_coverage.npy written by run_paslcd_pair.py, to activate the binary visibility filter",
    )
    parser.add_argument(
        "--render-t0-confidence", type=Path, default=None,
        help="optional render_t0_confidence.npy written by run_paslcd_pair.py, to activate the confidence-weighted "
             "visibility filter (needs use_confidence_weighted_visibility: true in config too)",
    )
    parser.add_argument(
        "--render-t0-corroboration", type=Path, default=None,
        help="optional render_t0_corroboration.npy written by run_paslcd_pair.py, to activate the cross-reference-"
             "view corroboration filter (needs enable_reference_corroboration: true in config too)",
    )
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from ocmask_pipeline.change_detection import run_object_state_resolution
    from ocmask_pipeline.config import load_config

    config = load_config(args.config)
    render_t0 = np.asarray(Image.open(args.render_t0).convert("RGB"))
    clean_render = np.asarray(Image.open(args.clean_render).convert("RGB"))
    image_t1 = np.asarray(Image.open(args.image_t1).convert("RGB"))

    geometry_kwargs = {}
    if args.render_t0_positions is not None:
        geometry_kwargs["render_t0_positions"] = np.load(args.render_t0_positions)
        geometry_kwargs["clean_render_positions"] = np.load(args.clean_render_positions)
        geometry_kwargs["image_t1_positions"] = np.load(args.image_t1_positions)
        geometry_kwargs["scene_scale"] = json.loads(args.scene_scale_path.read_text())["scene_scale"]
    if args.render_t0_coverage is not None:
        geometry_kwargs["render_t0_coverage"] = np.load(args.render_t0_coverage)
    if args.render_t0_confidence is not None:
        geometry_kwargs["render_t0_confidence"] = np.load(args.render_t0_confidence)
    if args.render_t0_corroboration is not None:
        geometry_kwargs["render_t0_corroboration"] = np.load(args.render_t0_corroboration)

    result = run_object_state_resolution(render_t0, clean_render, image_t1, args.output_dir, config, **geometry_kwargs)
    print(
        json.dumps(
            {
                "output": str(result.artifacts_dir.resolve()),
                "changed_pixel_fraction": float((result.labels != 0).mean()),
                "decision_counts": result.diagnostics["decision_counts"],
                "tracking_recovery": result.diagnostics.get("tracking_recovery"),
                "timings_seconds": result.timings,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
