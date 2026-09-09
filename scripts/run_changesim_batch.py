#!/usr/bin/env python3
"""Run change_pipeline's own current pipeline on ChangeSim pairs (image0/
image1, single image each side -- no multi-view reference set, unlike
PASLCD/SceneDiff). GT decoding logic (raw AirAim segmentation IDs -> our
Label enum) ported from change_detect's src/ocmask/changesim.py, verified
there against the official ChangeSim idx2color.txt palette; not re-derived
here.

Reconstruction quality from a single image per side will be worse than
PASLCD's 24-reference-image setup -- this is a known, expected limitation
of ChangeSim's own 2-image protocol, not a bug.

Refines with DI2FIX by default (a real pipeline stage, not optional -- was
silently missing from every ChangeSim run until 2026-09-08). DI2FIX only
touches RGB appearance, not pixel alignment, so the geometry arrays saved
from the original (unrefined) reconstruction stay valid against the
refined render_t0/clean_render -- same convention as run_paslcd_pair.py.
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

CHANGESIM_RGB_TO_RAW = {
    (0, 0, 0): 0,
    (81, 38, 0): 1,
    (41, 36, 132): 2,
    (25, 48, 16): 3,
    (131, 192, 13): 4,
}


def load_manifest(path: Path) -> list[dict]:
    base = path.resolve().parent
    pairs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        resolve = lambda v: (base / v).resolve() if not Path(v).is_absolute() else Path(v)
        pairs.append({
            "id": row["id"],
            "image0": resolve(row["image0"]),
            "image1": resolve(row["image1"]),
            "target": resolve(row["target"]),
        })
    return pairs


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-refine", action="store_true",
                         help="explicit opt-out only -- refine is a real pipeline stage, on by default")
    args = parser.parse_args()

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import reconstruct_and_render
    from PIL import Image

    args.output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    pairs = load_manifest(args.manifest)
    if args.limit:
        pairs = pairs[: args.limit]

    manifest = []
    query_meta = {}
    for pair in pairs:
        pair_id = pair["id"]
        out_dir = args.output_root / pair_id
        try:
            result = reconstruct_and_render([pair["image0"]], [pair["image1"]], 0, 0, config)

            recon_dir = out_dir / "reconstruction"
            recon_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.render_t0).save(recon_dir / "render_t0.png")
            Image.fromarray(result.clean_render).save(recon_dir / "clean_render.png")
            Image.fromarray(result.image_t1).save(recon_dir / "image_t1.png")

            # Full pipeline needs these too: geometric identity (positions +
            # scene_scale) and the visibility filter (coverage, confidence)
            # -- omitted in the first pass, which silently ran without them.
            # Reference corroboration is skipped: off by default
            # (enable_reference_corroboration: false) and not meaningful with
            # a single t0 image anyway (no multiple reference views to
            # cross-check against).
            np.save(recon_dir / "render_t0_positions.npy", result.render_t0_positions)
            np.save(recon_dir / "clean_render_positions.npy", result.clean_render_positions)
            np.save(recon_dir / "image_t1_positions.npy", result.image_t1_positions)
            np.save(recon_dir / "render_t0_coverage.npy", result.render_t0_coverage)
            np.save(recon_dir / "render_t0_confidence.npy", result.render_t0_confidence)
            (recon_dir / "scene_scale.json").write_text(json.dumps({"scene_scale": result.scene_scale}))

            render_t0_path, clean_render_path = recon_dir / "render_t0.png", recon_dir / "clean_render.png"
            if not args.skip_refine and config.get("refine", {}).get("enabled", False):
                refined_dir = out_dir / "refined"
                subprocess.run(
                    ["conda", "run", "-n", DIFIX_ENV, "python", str(REPO / "scripts/refine.py"),
                     "--render-t0", str(render_t0_path), "--clean-render", str(clean_render_path),
                     "--image-t1", str(recon_dir / "image_t1.png"), "--config", str(args.config),
                     "--output-dir", str(refined_dir)],
                    check=True,
                )
                render_t0_path, clean_render_path = refined_dir / "render_t0.png", refined_dir / "clean_render.png"

            manifest.append({
                "render_t0": str(render_t0_path),
                "clean_render": str(clean_render_path),
                "image_t1": str(recon_dir / "image_t1.png"),
                "render_t0_positions": str(recon_dir / "render_t0_positions.npy"),
                "clean_render_positions": str(recon_dir / "clean_render_positions.npy"),
                "image_t1_positions": str(recon_dir / "image_t1_positions.npy"),
                "scene_scale_path": str(recon_dir / "scene_scale.json"),
                "render_t0_coverage": str(recon_dir / "render_t0_coverage.npy"),
                "render_t0_confidence": str(recon_dir / "render_t0_confidence.npy"),
                "output_dir": str(out_dir / "detect"),
            })
            query_meta[pair_id] = {"target": str(pair["target"]), "output_dir": str(out_dir / "detect"),
                                    "image_t1_shape": list(result.image_t1.shape[:2])}
            print(f"[{pair_id}] reconstruction OK", flush=True)
        except Exception as e:
            print(f"[{pair_id}] RECONSTRUCTION FAILED: {e}", flush=True)

    manifest_path = args.output_root / "detect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    meta_path = args.output_root / "query_meta.json"
    meta_path.write_text(json.dumps(query_meta, indent=2))
    print(f"wrote {manifest_path} ({len(manifest)} entries), {meta_path}")

    print(f"=== running detect_batch.py once on {len(manifest)} pairs ===", flush=True)
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", DETECT_ENV, "python", str(REPO / "scripts/detect_batch.py"),
         "--manifest", str(manifest_path), "--config", str(args.config)],
        check=True,
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
