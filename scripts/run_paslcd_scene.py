#!/usr/bin/env python3
"""Evaluate every query photo in one PASLCD scene instance (~25 images) and
write results/paslcd/<Dataset>_<Instance>/{metrics.csv, summary.json}.

Thin loop over run_paslcd_pair.run_one_pair -- no orchestration logic is
duplicated. Run in the vggt-omega conda env, same as run_paslcd_pair.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_paslcd_pair as pair  # noqa: E402


def _already_evaluated(scene_dir: Path, stem: str) -> bool:
    metrics_path = scene_dir / "metrics.csv"
    if not metrics_path.exists():
        return False
    import csv

    with metrics_path.open() as handle:
        return any(row["test_image"] == stem for row in csv.DictReader(handle))


def run_scene(
    data_root: Path,
    dataset: str,
    instance: str,
    output_root: Path,
    config_path: Path,
    max_reference_images: int | None,
    skip_refine: bool,
    limit: int | None = None,
    reference_scene_cache_dir: Path | None = None,
    resume: bool = False,
    detect_lock_file: Path | None = None,
) -> dict:
    instance_dir = data_root / dataset / instance
    reference_images, test_images = pair.discover_instance_images(instance_dir)
    if max_reference_images:
        reference_images = pair.even_subsample(reference_images, max_reference_images)
    if limit:
        test_images = test_images[:limit]

    scene_dir = output_root / pair.scene_name(dataset, instance)

    if resume:
        test_images = [t for t in test_images if not _already_evaluated(scene_dir, t.stem)]
        if not test_images:
            print(f"[{dataset}/{instance}] all queries already evaluated, skipping", flush=True)
            return pair.write_scene_summary(scene_dir, pair.scene_name(dataset, instance), dataset, instance)

    # Built once and reused for every query below -- the reference scene
    # does not depend on which query photo is being evaluated. See
    # reconstruction.reconstruct_reference_scene's docstring. Reused from a
    # prior overnight batch's cache when available instead of rebuilding.
    reference_scene = None
    if reference_scene_cache_dir:
        from ocmask_pipeline.reconstruction import ReferenceScene

        cache_path = reference_scene_cache_dir / f"{pair.scene_name(dataset, instance)}.npz"
        if cache_path.exists():
            print(f"[{dataset}/{instance}] loading cached reference scene from {cache_path}", flush=True)
            reference_scene = ReferenceScene.load(cache_path)
            if [p.resolve() for p in reference_scene.image_paths] != [p.resolve() for p in reference_images]:
                print(f"[{dataset}/{instance}] cached reference scene's images don't match current selection, rebuilding", flush=True)
                reference_scene = None
    if reference_scene is None:
        print(f"[{dataset}/{instance}] building reference scene from {len(reference_images)} images", flush=True)
        reference_scene = pair.build_reference_scene(reference_images, config_path)

    # Stages 1-2 (reconstruction + optional refine) still run per-query --
    # they're comparatively cheap (~1 min) and each needs its own VGGT-Omega
    # localization. Stage 3 (SAM3/DINOv2/SAM2) is batched into one
    # detect_batch.py subprocess for the whole instance instead of one
    # detect.py subprocess per query, so its ~150s model-load/compile cost
    # is paid once per instance (20x) instead of once per query (500x).
    all_paths = {}
    for test_image in test_images:
        print(f"[{dataset}/{instance}] reconstructing {test_image.stem}", flush=True)
        all_paths[test_image.stem] = pair.reconstruct_and_refine_one_pair(
            dataset, instance, reference_images, test_image, instance_dir, scene_dir, config_path, skip_refine,
            reference_scene=reference_scene,
        )

    if all_paths:
        import subprocess

        manifest = [
            {
                "render_t0": str(p["render_t0"]), "clean_render": str(p["clean_render"]),
                "image_t1": str(p["image_t1"]), "output_dir": str(p["detect_dir"]),
                "render_t0_positions": str(p["render_t0_positions"]),
                "clean_render_positions": str(p["clean_render_positions"]),
                "image_t1_positions": str(p["image_t1_positions"]),
                "scene_scale_path": str(p["scene_scale_path"]),
                "render_t0_coverage": str(p["render_t0_coverage"]),
                "render_t0_confidence": str(p["render_t0_confidence"]),
                "render_t0_corroboration": str(p["render_t0_corroboration"]),
            }
            for p in all_paths.values()
        ]
        manifest_path = scene_dir / "detect_batch_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        detect_cmd = [
            "conda", "run", "-n", pair.DETECT_ENV, "python", str(pair.REPO / "scripts/detect_batch.py"),
            "--manifest", str(manifest_path), "--config", str(config_path),
        ]
        if detect_lock_file:
            # detect_batch.py alone uses ~7.3GB (SAM3+SAM2+DINOv2 all
            # loaded); two of those don't fit alongside anything else on a
            # 16GB GPU (confirmed by direct OOM testing under
            # --parallel-instances). Reconstruction (this loop, above) is
            # cheap enough (~4.6GB) to run genuinely concurrently across
            # instances; only the detect stage needs to be serialized
            # system-wide, via a plain flock on a shared file, while other
            # instances' reconstruction keeps running unblocked.
            import fcntl
            import time

            detect_lock_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{dataset}/{instance}] waiting for detect lock ({detect_lock_file})", flush=True)
            with detect_lock_file.open("w") as lock_handle:
                fcntl.flock(lock_handle, fcntl.LOCK_EX)
                try:
                    print(f"[{dataset}/{instance}] running detect_batch.py on {len(manifest)} queries", flush=True)
                    subprocess.run(detect_cmd, check=True)
                    # Combined peak usage of two detect_batch processes
                    # (~7.3-7.8GB each) leaves under 500MB of headroom on a
                    # 16GB GPU -- observed OOMing another waiting instance's
                    # detect_batch even with the flock correctly held,
                    # because the just-exited process's CUDA memory is not
                    # always reclaimed by the driver in the same instant the
                    # process exits. This margin gives that reclaim time to
                    # actually complete before releasing the lock.
                    time.sleep(10)
                finally:
                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
        else:
            print(f"[{dataset}/{instance}] running detect_batch.py on {len(manifest)} queries", flush=True)
            subprocess.run(detect_cmd, check=True)

    for test_image in test_images:
        row = pair.evaluate_one_pair(dataset, instance, test_image, scene_dir, reference_images, all_paths[test_image.stem])
        print(f"[{dataset}/{instance}] {test_image.stem} iou={row['iou']:.4f} f1={row['f1']:.4f} precision={row['precision']:.4f} recall={row['recall']:.4f}", flush=True)

    summary = pair.write_scene_summary(scene_dir, pair.scene_name(dataset, instance), dataset, instance)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "PASLCD")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--max-reference-images", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N query images (for quick iteration)")
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "paslcd")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--reference-scene-cache-dir", type=Path, default=None, help="reuse a saved ReferenceScene .npz (see build_paslcd_reference_scenes.py) instead of rebuilding")
    parser.add_argument("--resume", action="store_true", help="skip query images already present in this scene's metrics.csv")
    parser.add_argument("--detect-lock-file", type=Path, default=None, help="serialize the detect_batch.py subprocess against this flock file -- use when running multiple instances concurrently (see run_paslcd_benchmark.py --parallel-instances), since detect_batch.py alone uses ~7.3GB and two do not fit on a 16GB GPU")
    args = parser.parse_args()

    summary = run_scene(
        args.data_root, args.dataset, args.instance, args.output_root, args.config,
        args.max_reference_images, args.skip_refine, args.limit,
        args.reference_scene_cache_dir, args.resume, args.detect_lock_file,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
