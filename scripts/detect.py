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
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from ocmask_pipeline.change_detection import run_object_state_resolution
    from ocmask_pipeline.config import load_config

    config = load_config(args.config)
    render_t0 = np.asarray(Image.open(args.render_t0).convert("RGB"))
    clean_render = np.asarray(Image.open(args.clean_render).convert("RGB"))
    image_t1 = np.asarray(Image.open(args.image_t1).convert("RGB"))

    result = run_object_state_resolution(render_t0, clean_render, image_t1, args.output_dir, config)
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
