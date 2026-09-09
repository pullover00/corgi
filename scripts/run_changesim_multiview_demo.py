#!/usr/bin/env python3
"""One-off demo: run the reference-set-vs-single-query methodology (same
pattern as PASLCD/SceneDiff) on a hand-picked ChangeSim sequence instead of
ChangeSim's own native single-image-vs-single-image protocol.

Warehouse_9/Seq_0 was chosen (see conversation/artifact) because its t0
camera revisits ~9-12 distinct viewpoints across query indices 0-40 (not one
frozen image), all four change classes (ADDED/REMOVED/MOVED/REPLACED) are
present with non-trivial area throughout that range, and query idx=14 has
the largest, most balanced REPLACED region (barrels + forklift) of any frame
scanned -- a good demonstration case for the color-replacement detector,
which ChangeSim's default 20-pair Warehouse_6 ablation subset never
exercises (REPLACED = 0.0 in every ablation row there).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DETECT_ENV = os.environ.get("DETECTION_CONDA_ENV", "goldilocs")
DIFIX_ENV = os.environ.get("DIFIX3D_CONDA_ENV", "difix3d")

CHANGESIM_ROOT = Path("/home/tessa/change_detect/data/changesim")
SEQ_DIR = CHANGESIM_ROOT / "Warehouse_9" / "Seq_0"
T0_INDICES = [0, 5, 14, 18, 19, 25, 26, 28, 35, 40]
QUERY_INDEX = 14


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REPO / "results/changesim_wh9_seq0_multiview_demo")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--skip-refine", action="store_true")
    args = parser.parse_args()

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import reconstruct_reference_scene, localize_and_render_query
    from PIL import Image

    args.output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    t0_paths = [SEQ_DIR / "t0" / "rgb" / f"{i}.png" for i in T0_INDICES]
    query_path = SEQ_DIR / "rgb" / f"{QUERY_INDEX}.png"
    for p in t0_paths + [query_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"reconstructing reference scene from {len(t0_paths)} T0 views: {T0_INDICES}", flush=True)
    reference_scene = reconstruct_reference_scene(t0_paths, config)
    result = localize_and_render_query(t0_paths, query_path, reference_scene, 0, config)
    print(f"alignment residual: {result.alignment_residual}", flush=True)

    recon_dir = args.output_root / "reconstruction"
    recon_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.render_t0).save(recon_dir / "render_t0.png")
    Image.fromarray(result.clean_render).save(recon_dir / "clean_render.png")
    Image.fromarray(result.image_t1).save(recon_dir / "image_t1.png")
    np.save(recon_dir / "render_t0_positions.npy", result.render_t0_positions)
    np.save(recon_dir / "clean_render_positions.npy", result.clean_render_positions)
    np.save(recon_dir / "image_t1_positions.npy", result.image_t1_positions)
    np.save(recon_dir / "render_t0_coverage.npy", result.render_t0_coverage)
    np.save(recon_dir / "render_t0_confidence.npy", result.render_t0_confidence)
    (recon_dir / "scene_scale.json").write_text(json.dumps({"scene_scale": result.scene_scale}))

    render_t0_path, clean_render_path = recon_dir / "render_t0.png", recon_dir / "clean_render.png"
    if not args.skip_refine and config.get("refine", {}).get("enabled", False):
        refined_dir = args.output_root / "refined"
        subprocess.run(
            ["conda", "run", "-n", DIFIX_ENV, "python", str(REPO / "scripts/refine.py"),
             "--render-t0", str(render_t0_path), "--clean-render", str(clean_render_path),
             "--image-t1", str(recon_dir / "image_t1.png"), "--config", str(args.config),
             "--output-dir", str(refined_dir)],
            check=True,
        )
        render_t0_path, clean_render_path = refined_dir / "render_t0.png", refined_dir / "clean_render.png"

    manifest = [{
        "render_t0": str(render_t0_path),
        "clean_render": str(clean_render_path),
        "image_t1": str(recon_dir / "image_t1.png"),
        "render_t0_positions": str(recon_dir / "render_t0_positions.npy"),
        "clean_render_positions": str(recon_dir / "clean_render_positions.npy"),
        "image_t1_positions": str(recon_dir / "image_t1_positions.npy"),
        "scene_scale_path": str(recon_dir / "scene_scale.json"),
        "render_t0_coverage": str(recon_dir / "render_t0_coverage.npy"),
        "render_t0_confidence": str(recon_dir / "render_t0_confidence.npy"),
        "output_dir": str(args.output_root / "detect"),
    }]
    manifest_path = args.output_root / "detect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}", flush=True)

    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", DETECT_ENV, "python", str(REPO / "scripts/detect_batch.py"),
         "--manifest", str(manifest_path), "--config", str(args.config)],
        check=True,
    )

    # Save GT + query for evaluation/visualization
    gt_path = SEQ_DIR / "change_segmentation" / f"{QUERY_INDEX}.png"
    Image.open(gt_path).convert("RGB").save(args.output_root / "gt_change_segmentation.png")
    Image.open(query_path).convert("RGB").save(args.output_root / "query_t1_raw.png")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
