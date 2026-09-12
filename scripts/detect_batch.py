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
import os
import sys
import time
from pathlib import Path

# Newer torch/inductor builds (needed for sm_120/Blackwell GPUs -- see
# SETUP.md) hit a CUDA-graphs tensor-aliasing bug in SAM2's compiled VOS
# path ("accessing tensor output of CUDAGraphs that has been overwritten
# by a subsequent run"), not reproduced with the torch version this repo
# was originally validated against. Disabling inductor's cudagraph capture
# sidesteps it -- pure speed tradeoff, not a correctness one. setdefault
# so an explicit environment choice (e.g. re-enabling it once fixed
# upstream) still wins.
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

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
    parser.add_argument(
        "--sequential-model-lifecycle", action="store_true",
        help="load/release proposal, tracking, and text models inside each item to lower peak VRAM; slower but identical",
    )
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from ocmask_pipeline.adapters.dinov2 import Dinov2FeatureExtractor
    from ocmask_pipeline.change_detection import _proposal_kwargs, run_object_state_resolution
    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.stages.sam2_tracking_backend import Sam2MaskTracker
    from ocmask_pipeline.change_detection import ThreeImageSettings
    from ocmask_pipeline.stages.sam3_proposals import Sam3AutomaticMaskGenerator, Sam3TextPromptDetector

    config = load_config(args.config)
    items = json.loads(args.manifest.read_text())

    sam_cfg = config["sam3_proposals"]
    ceiling_sky_settings = ThreeImageSettings.from_config(config)
    # If every item in this batch is a fast replay (--load-inventory-from
    # equivalent: item["load_inventory_from"] set), stages 1-3 never run for
    # any of them, so SAM3/DINOv2/SAM2 are never touched -- skip loading
    # them at all rather than pay their load time for nothing. A mixed
    # batch (some replay, some not) still needs all three, since at least
    # one item takes the slow path.
    all_replay = items and all(item.get("load_inventory_from") for item in items)
    reuse_models = not args.sequential_model_lifecycle
    generator = (
        Sam3AutomaticMaskGenerator(sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"], **_proposal_kwargs(config))
        if reuse_models and not all_replay else None
    )
    dino_extractor = (
        None
        if all_replay or not ceiling_sky_settings.use_dino_features or not reuse_models
        else Dinov2FeatureExtractor(config["dinov2_features"])
    )
    tracker = Sam2MaskTracker(config["sam2_tracking"]) if reuse_models and not all_replay else None
    # A SEPARATE SAM3 model from `generator` (see Sam3TextPromptDetector's
    # docstring for why they cannot share one) -- built once and reused
    # across the whole batch, same as the other three, but only when this
    # run's config actually needs it, so a batch with the flag off pays no
    # extra VRAM/load cost.
    text_detector = (
        Sam3TextPromptDetector(
            sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"],
            confidence_threshold=ceiling_sky_settings.ceiling_sky_confidence_threshold,
        )
        if ceiling_sky_settings.enable_ceiling_sky_suppression and not all_replay and reuse_models else None
    )

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
            if "above_horizon" in item:
                geometry_kwargs["above_horizon"] = np.load(item["above_horizon"])
            if "dump_stages" in item:
                geometry_kwargs["dump_stages"] = item["dump_stages"]
            if item.get("dump_inventory_to"):
                geometry_kwargs["dump_inventory_to"] = item["dump_inventory_to"]
            if item.get("load_inventory_from"):
                geometry_kwargs["load_inventory_from"] = item["load_inventory_from"]
            if item.get("ceiling_sky_mask"):
                # Optional externally computed structural-surface mask (bool .npy at the
                # working resolution). When present it takes precedence over both the
                # bundle-cached mask and in-resolver recomputation -- plumbing only,
                # added 2026-09-12 for the surface-prompt ablation.
                geometry_kwargs["ceiling_sky_mask"] = np.load(item["ceiling_sky_mask"]).astype(bool)
            if item.get("movable_object_mask"):
                geometry_kwargs["movable_object_mask"] = np.load(item["movable_object_mask"]).astype(bool)
            if "sam_render_t0" in item:
                geometry_kwargs["sam_render_t0"] = np.asarray(Image.open(item["sam_render_t0"]).convert("RGB"))
                geometry_kwargs["sam_clean_render"] = np.asarray(Image.open(item["sam_clean_render"]).convert("RGB"))
                geometry_kwargs["sam_image_t1"] = np.asarray(Image.open(item["sam_image_t1"]).convert("RGB"))

            result = run_object_state_resolution(
                render_t0, clean_render, image_t1, item["output_dir"], config,
                generator=generator, dino_extractor=dino_extractor, tracker=tracker,
                text_detector=text_detector,
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
        if generator is not None:
            generator.release()
        if dino_extractor is not None:
            dino_extractor.release()
        if tracker is not None:
            tracker.release()
        if text_detector is not None:
            text_detector.release()

    if args.results_output:
        args.results_output.write_text(json.dumps(results, indent=2))
    print(json.dumps({"num_processed": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
