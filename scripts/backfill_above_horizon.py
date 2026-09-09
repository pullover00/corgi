#!/usr/bin/env python3
"""Generate above_horizon.npy for queries whose cached reconstruction predates
the above-horizon suppression feature.

The map is a function of the query camera's pose and intrinsics, which the
cache does not store -- so this re-runs the VGGT-Omega localization to
recover them. Everything else in the cache (renders, positions, coverage,
confidence, corroboration) is left untouched, so the detect-only ablation
pattern still holds afterwards: only the new buffer is added.

The reference scene is reconstructed once per PASLCD instance and reused
across that instance's queries, exactly as run_paslcd_scene.py does.

Run in the vggt-omega conda env.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "results/paslcd_ablation_cache"
DATA = REPO / "data/PASLCD"


def instance_images(dataset: str, instance: str) -> tuple[list[Path], dict[str, Path]]:
    images_dir = DATA / dataset / instance / "images"
    all_images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    reference = [p for p in all_images if "_test_" not in p.stem]
    tests = {p.stem: p for p in all_images if "_test_" in p.stem}
    return reference, tests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    ap.add_argument("--max-reference-images", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import numpy as np

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import localize_and_render_query, reconstruct_reference_scene

    config = load_config(args.config)

    todo: dict[str, list[tuple[str, Path]]] = {}
    for scene_dir in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        for stem_dir in sorted((scene_dir / "intermediate").iterdir()):
            out = stem_dir / "reconstruction" / "above_horizon.npy"
            if out.exists() and not args.overwrite:
                continue
            todo.setdefault(scene_dir.name, []).append((stem_dir.name, stem_dir))
    total = sum(len(v) for v in todo.values())
    if not total:
        print("nothing to do -- every cached query already has above_horizon.npy")
        return 0
    print(f"{total} queries across {len(todo)} instances need above_horizon.npy", flush=True)

    done = 0
    for scene, entries in todo.items():
        dataset, inst = scene.rsplit("_Instance_", 1)
        reference, tests = instance_images(dataset, f"Instance_{inst}")
        if args.max_reference_images:
            n = args.max_reference_images
            step = (len(reference) - 1) / (n - 1) if n > 1 and n < len(reference) else 1
            reference = [reference[i] for i in sorted({round(i * step) for i in range(n)})] if n < len(reference) else reference
        print(f"\n[{scene}] reconstructing reference scene from {len(reference)} images", flush=True)
        started = time.perf_counter()
        reference_scene = reconstruct_reference_scene(reference, config)
        print(f"[{scene}] reference scene in {time.perf_counter() - started:.0f}s", flush=True)

        for stem, stem_dir in entries:
            if stem not in tests:
                print(f"  !! no source query image for {stem}, skipping", flush=True)
                continue
            started = time.perf_counter()
            result = localize_and_render_query(
                t0_image_paths=reference, query_image_path=tests[stem],
                reference_scene=reference_scene, t0_reference_index=0, config=config,
            )
            if result.above_horizon is None:
                print(f"  !! no dominant plane for {stem}; suppression will stay off here", flush=True)
                continue
            out = stem_dir / "reconstruction" / "above_horizon.npy"
            np.save(out, result.above_horizon)
            done += 1
            print(f"  [{done}/{total}] {scene}/{stem}: {100 * result.above_horizon.mean():.1f}% of pixels "
                  f"above horizon, residual {result.alignment_residual:.5f} ({time.perf_counter() - started:.0f}s)",
                  flush=True)

    print(f"\nwrote {done}/{total} above_horizon maps")
    print("BACKFILL_ABOVE_HORIZON_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
