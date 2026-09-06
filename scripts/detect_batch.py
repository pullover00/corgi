#!/usr/bin/env python3
"""Stage 3, batched: SAM3 + DINOv2 + SAM2-tracking object-state resolution
over many (render_t0, clean_render, image_t1) triples in one process,
loading each model once instead of once per triple.

Cold model load + first-forward-pass compilation (SAM3's image encoder,
SAM2's VOS components) measured at ~150s combined per detect.py invocation
-- negligible for one query, but that repeated 500 times across a PASLCD
benchmark run is ~20+ hours of pure reload overhead. This script pays that
cost once per process and reuses the same generator/dino_extractor/tracker
for every triple in --manifest.

Run in the detection conda env (SAM3, SAM2, DINOv2 -- see README.md). Needs
SAM3_SOURCE and SAM3_IMAGE_CHECKPOINT set in the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help='JSON list of {"render_t0", "clean_render", "image_t1", "output_dir", and optionally '
             '"render_t0_positions", "clean_render_positions", "image_t1_positions" (.npy paths) + '
             '"scene_scale_path" (.json path) to activate the geometric-identity test, and/or '
             '"render_t0_coverage" (.npy path) to activate the visibility filter}',
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--results-output", type=Path, default=None, help="write the per-item result summaries here as a JSON list")
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from ocmask_pipeline.adapters.dinov2 import Dinov2FeatureExtractor
    from ocmask_pipeline.change_detection import _proposal_kwargs, run_object_state_resolution
    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.stages.sam2_tracking_backend import Sam2MaskTracker
    from ocmask_pipeline.stages.sam3_proposals import Sam3AutomaticMaskGenerator

    config = load_config(args.config)
    items = json.loads(args.manifest.read_text())

    sam_cfg = config["sam3_proposals"]
    generator = Sam3AutomaticMaskGenerator(sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"], **_proposal_kwargs(config))
    dino_extractor = Dinov2FeatureExtractor(config["dinov2_features"])
    tracker = Sam2MaskTracker(config["sam2_tracking"])

    results = []
    try:
        for index, item in enumerate(items):
            started = time.perf_counter()
            render_t0 = np.asarray(Image.open(item["render_t0"]).convert("RGB"))
            clean_render = np.asarray(Image.open(item["clean_render"]).convert("RGB"))
            image_t1 = np.asarray(Image.open(item["image_t1"]).convert("RGB"))

            geometry_kwargs = {}
            if "render_t0_positions" in item:
                geometry_kwargs["render_t0_positions"] = np.load(item["render_t0_positions"])
                geometry_kwargs["clean_render_positions"] = np.load(item["clean_render_positions"])
                geometry_kwargs["image_t1_positions"] = np.load(item["image_t1_positions"])
                geometry_kwargs["scene_scale"] = json.loads(Path(item["scene_scale_path"]).read_text())["scene_scale"]
            if "render_t0_coverage" in item:
                geometry_kwargs["render_t0_coverage"] = np.load(item["render_t0_coverage"])
            if "render_t0_confidence" in item:
                geometry_kwargs["render_t0_confidence"] = np.load(item["render_t0_confidence"])
            if "render_t0_corroboration" in item:
                geometry_kwargs["render_t0_corroboration"] = np.load(item["render_t0_corroboration"])

            result = run_object_state_resolution(
                render_t0, clean_render, image_t1, item["output_dir"], config,
                generator=generator, dino_extractor=dino_extractor, tracker=tracker,
                **geometry_kwargs,
            )
            summary = {
                "output": str(result.artifacts_dir.resolve()),
                "changed_pixel_fraction": float((result.labels != 0).mean()),
                "decision_counts": result.diagnostics["decision_counts"],
                "tracking_recovery": result.diagnostics.get("tracking_recovery"),
                "timings_seconds": result.timings,
                "wall_seconds": time.perf_counter() - started,
            }
            results.append(summary)
            print(f"[{index + 1}/{len(items)}] {item['output_dir']} -- {summary['wall_seconds']:.1f}s", flush=True)
    finally:
        generator.release()
        dino_extractor.release()
        tracker.release()

    if args.results_output:
        args.results_output.write_text(json.dumps(results, indent=2))
    print(json.dumps({"num_processed": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
