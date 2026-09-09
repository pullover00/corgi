#!/usr/bin/env python3
"""Run the pipeline on one PASLCD before/after pair and score it against
PASLCD's own ground-truth binary change mask.

Dataset layout (data/PASLCD/<Scene>/<Instance>/):
  images/      a posed "reference" (before) photo set -- every filename
               *without* a "_test_" infix -- plus a handful of unposed
               "query" (after) photos, one per evaluation instance, named
               ``<prefix>_test_<id>.jpg``.
  sparse/0/    COLMAP reconstruction of the reference set (unused here --
               VGGT-Omega estimates its own poses jointly, same as every
               other dataset this pipeline runs on).
  gt_mask/     one binary PNG per query photo, ``<query_stem>.png``,
               downsampled from the query photo's own resolution (~1600px
               longest side). This is PASLCD's official evaluation
               resolution -- see ocmask_pipeline/metrics.py.

Pairing protocol used here: for one query photo, t0 = the full (or
subsampled, see --max-reference-images) reference image set, t1 = that
single query photo. This matches PASLCD's own "pose-agnostic" setup: each
query is an independent one-shot observation of the after-scene, not part
of a video/walkthrough, evaluated against the SAME reference reconstruction
used for every other query in that instance.

Unlike run_scenediff_pair.py (which reconstructs T0 and T1 jointly in one
VGGT-Omega call), the reference scene here is reconstructed once, in
isolation from any query photo (reconstruction.reconstruct_reference_scene),
then localized against per query (reconstruction.localize_and_render_query).
This was changed after diagnosing a real accuracy failure: with one joint
call, small added objects on an otherwise-static surface (e.g. items placed
on a table) were being spuriously matched as "unchanged" against the
reference scene, and per-image debugging traced this to the reference
reconstruction itself being subtly influenced by the odd-one-out query frame
sharing the same batch (via the model's cross-attention across frames), not
to any proposal-generation or dedup bug. Isolating the reference
reconstruction also means it is computed once per scene instead of once per
query -- see run_paslcd_scene.py.

We call this a "scene" (results/paslcd/<scene>/) at <Dataset>_<Instance>
granularity, matching the granularity PASLCD's own run.sh evaluates at (one
evaluate.py call per Instance, over that instance's ~25 query photos).

Run in the vggt-omega conda env (same as run_scenediff_pair.py); shells out
to the other two envs for the refine/detect stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DIFIX_ENV = os.environ.get("DIFIX3D_CONDA_ENV", "difix3d")
DETECT_ENV = os.environ.get("DETECTION_CONDA_ENV", "goldilocs")

METRICS_FIELDS = [
    "scene", "instance", "test_image", "num_reference_images",
    "iou", "f1", "precision", "recall", "tp", "fp", "fn", "tn",
    "changed_pixel_fraction_native", "output_dir",
]


def scene_name(dataset: str, instance: str) -> str:
    return f"{dataset}_{instance}"


def discover_instance_images(instance_dir: Path) -> tuple[list[Path], list[Path]]:
    """Split PASLCD's flat images/ directory into (reference, test) images."""
    images_dir = instance_dir / "images"
    all_images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    reference = [p for p in all_images if "_test_" not in p.stem]
    test = [p for p in all_images if "_test_" in p.stem]
    if not reference:
        raise FileNotFoundError(f"no reference images found under {images_dir}")
    if not test:
        raise FileNotFoundError(f"no query/test images found under {images_dir}")
    return reference, test


def even_subsample(items: list[Path], n: int) -> list[Path]:
    """Uniform-stride subsample preserving the first/last item, mirroring
    run_scenediff_pair.py's sample_frame_indices (there applied to video
    frame indices, here to an already-extracted image list)."""
    if n >= len(items):
        return items
    if n <= 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    return [items[i] for i in indices]


def gt_mask_path(instance_dir: Path, test_image: Path) -> Path:
    return instance_dir / "gt_mask" / f"{test_image.stem}.png"


def append_metrics_row(scene_dir: Path, row: dict[str, Any]) -> None:
    """Upsert by test_image, not a blind append -- a --resume run that
    redoes a query (e.g. to add refine, see _already_evaluated's
    require_refine) must replace that query's stale row rather than
    duplicate it, or write_scene_summary's mean would double-count it."""
    metrics_path = scene_dir / "metrics.csv"
    existing_rows = []
    if metrics_path.exists():
        with metrics_path.open() as handle:
            existing_rows = [r for r in csv.DictReader(handle) if r["test_image"] != row["test_image"]]
    existing_rows.append({key: row[key] for key in METRICS_FIELDS})
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def write_scene_summary(scene_dir: Path, scene: str, dataset: str, instance: str) -> dict[str, Any]:
    """Recompute summary.json from metrics.csv (the source of truth), so it
    stays correct however many test images have been evaluated so far."""
    metrics_path = scene_dir / "metrics.csv"
    with metrics_path.open() as handle:
        rows = list(csv.DictReader(handle))

    def mean(field: str) -> float:
        return sum(float(r[field]) for r in rows) / len(rows) if rows else 0.0

    summary = {
        "scene": scene,
        "dataset": dataset,
        "instance": instance,
        "num_test_images_evaluated": len(rows),
        "mean_iou": mean("iou"),
        "mean_f1": mean("f1"),
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
        "per_image": [r["test_image"] for r in rows],
    }
    (scene_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_reference_scene(reference_images: list[Path], config_path: Path):
    """Reconstruct the reference ("before") scene once, in isolation from any
    query photo -- see reconstruction.reconstruct_reference_scene's
    docstring for why this replaced a single joint T0+T1 VGGT-Omega call.
    Reused across every query photo in a scene by run_paslcd_scene.py;
    run_one_pair also accepts a pre-built one via ``reference_scene=``."""
    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import reconstruct_reference_scene

    config = load_config(config_path)
    return reconstruct_reference_scene(reference_images, config)


def reconstruct_and_refine_one_pair(
    dataset: str,
    instance: str,
    reference_images: list[Path],
    test_image: Path,
    instance_dir: Path,
    scene_dir: Path,
    config_path: Path,
    skip_refine: bool = False,
    reference_scene=None,
) -> dict[str, Path]:
    """Stages 1-2 only: localize + render, then optional DI2FIX refinement.
    Returns the paths detect needs (render_t0/clean_render/image_t1) plus
    gt_path/images_dir/etc, without running stage 3 -- so a caller can batch
    many queries' detect calls into one warm process (see
    scripts/detect_batch.py) instead of one subprocess per query.
    """
    from PIL import Image

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import localize_and_render_query, reconstruct_reference_scene

    stem = test_image.stem
    images_dir = scene_dir / "images" / stem
    intermediate_dir = scene_dir / "intermediate" / stem
    predictions_dir = scene_dir / "predictions" / stem
    visualizations_dir = scene_dir / "visualizations" / stem
    for d in (images_dir, intermediate_dir, predictions_dir, visualizations_dir):
        d.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    gt_path = gt_mask_path(instance_dir, test_image)
    if not gt_path.exists():
        raise FileNotFoundError(f"no GT mask for {test_image.name}: expected {gt_path}")

    recon_dir = intermediate_dir / "reconstruction"
    recon_marker_files = [
        recon_dir / "render_t0.png", recon_dir / "clean_render.png", recon_dir / "image_t1.png",
        recon_dir / "render_t0_positions.npy", recon_dir / "clean_render_positions.npy",
        recon_dir / "image_t1_positions.npy", recon_dir / "scene_scale.json",
        recon_dir / "render_t0_coverage.npy", recon_dir / "render_t0_confidence.npy",
        recon_dir / "render_t0_corroboration.npy",
    ]
    reconstruction_cached = all(p.exists() for p in recon_marker_files)

    if reconstruction_cached:
        # Stage 1 (VGGT-Omega reconstruction+localization) is the expensive
        # part of this function -- reused as-is from a prior run when its
        # full output set is already on disk, so a rerun that only needs to
        # add refine/detect (e.g. the 2026-09-08 refine-audit fix) does not
        # have to pay for it again. Safe because DI2FIX (stage below) only
        # ever touches render_t0/clean_render's RGB appearance, never the
        # geometry these files encode.
        print(f"  [{stem}] reusing cached reconstruction from {recon_dir}", flush=True)
        alignment_residual = float("nan")
        prior_result_path = intermediate_dir / "paslcd_result.json"
        if prior_result_path.exists():
            alignment_residual = json.loads(prior_result_path.read_text()).get("alignment_residual", alignment_residual)
    else:
        if reference_scene is None:
            reference_scene = reconstruct_reference_scene(reference_images, config)
        result = localize_and_render_query(
            t0_image_paths=reference_images,
            query_image_path=test_image,
            reference_scene=reference_scene,
            t0_reference_index=0,
            config=config,
        )
        print(f"  reference/query alignment residual: {result.alignment_residual}", flush=True)
        alignment_residual = result.alignment_residual

        recon_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.render_t0).save(recon_dir / "render_t0.png")
        Image.fromarray(result.clean_render).save(recon_dir / "clean_render.png")
        Image.fromarray(result.image_t1).save(recon_dir / "image_t1.png")
        # 3D world-position buffers for the geometric-identity test (unaffected
        # by DI2FIX refine below, which only touches RGB appearance, not pixel
        # alignment -- these stay valid even when render_t0/clean_render below
        # get replaced by their refined versions).
        import numpy as np

        np.save(recon_dir / "render_t0_positions.npy", result.render_t0_positions)
        np.save(recon_dir / "clean_render_positions.npy", result.clean_render_positions)
        np.save(recon_dir / "image_t1_positions.npy", result.image_t1_positions)
        (recon_dir / "scene_scale.json").write_text(json.dumps({"scene_scale": result.scene_scale}))
        np.save(recon_dir / "render_t0_coverage.npy", result.render_t0_coverage)
        np.save(recon_dir / "render_t0_confidence.npy", result.render_t0_confidence)
        np.save(recon_dir / "render_t0_corroboration.npy", result.render_t0_corroboration)

    Image.open(test_image).convert("RGB").save(images_dir / "query_after.png")
    Image.open(reference_images[0]).convert("RGB").save(images_dir / "reference_before_sample.png")

    render_t0_path, clean_render_path, image_t1_path = (
        recon_dir / "render_t0.png", recon_dir / "clean_render.png", recon_dir / "image_t1.png",
    )

    if not skip_refine and config.get("refine", {}).get("enabled", False):
        refined_dir = intermediate_dir / "refined"
        subprocess.run(
            [
                "conda", "run", "-n", DIFIX_ENV, "python", str(REPO / "scripts/refine.py"),
                "--render-t0", str(render_t0_path), "--clean-render", str(clean_render_path),
                "--image-t1", str(image_t1_path), "--config", str(config_path),
                "--output-dir", str(refined_dir),
            ],
            check=True,
        )
        render_t0_path, clean_render_path = refined_dir / "render_t0.png", refined_dir / "clean_render.png"

    return {
        "render_t0": render_t0_path,
        "clean_render": clean_render_path,
        "image_t1": image_t1_path,
        "render_t0_positions": recon_dir / "render_t0_positions.npy",
        "clean_render_positions": recon_dir / "clean_render_positions.npy",
        "image_t1_positions": recon_dir / "image_t1_positions.npy",
        "scene_scale_path": recon_dir / "scene_scale.json",
        "render_t0_coverage": recon_dir / "render_t0_coverage.npy",
        "render_t0_confidence": recon_dir / "render_t0_confidence.npy",
        "render_t0_corroboration": recon_dir / "render_t0_corroboration.npy",
        "detect_dir": intermediate_dir / "detect",
        "gt_path": gt_path,
        "images_dir": images_dir,
        "predictions_dir": predictions_dir,
        "visualizations_dir": visualizations_dir,
        "intermediate_dir": intermediate_dir,
        "alignment_residual": alignment_residual,
    }


def evaluate_one_pair(
    dataset: str,
    instance: str,
    test_image: Path,
    scene_dir: Path,
    reference_images: list[Path],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Stage 3's output (detect_dir/labels.png + inference.json) must already
    exist -- via scripts/detect.py or scripts/detect_batch.py. Computes and
    records PASLCD metrics against GT, exactly as run_one_pair used to
    inline after its own detect.py subprocess call."""
    import numpy as np
    from PIL import Image

    from ocmask_pipeline.metrics import (
        binarize_prediction,
        compute_binary_metrics,
        load_paslcd_gt,
        tp_fp_fn_visualization,
    )

    stem = test_image.stem
    detect_dir = paths["detect_dir"]
    labels = np.asarray(Image.open(detect_dir / "labels.png"))
    inference = json.loads((detect_dir / "inference.json").read_text())

    binary_native = (labels != 0).astype(np.uint8) * 255
    Image.fromarray(binary_native).save(paths["predictions_dir"] / "prediction_native.png")

    gt = load_paslcd_gt(paths["gt_path"])
    pred = binarize_prediction(binary_native, gt.shape)
    Image.fromarray((pred * 255).astype(np.uint8)).save(paths["predictions_dir"] / "prediction_gt_resolution.png")
    Image.fromarray((gt * 255).astype(np.uint8)).save(paths["images_dir"] / "gt_mask.png")

    metrics = compute_binary_metrics(gt, pred)
    vis = tp_fp_fn_visualization(gt, pred)
    Image.fromarray(vis).save(paths["visualizations_dir"] / "tp_fp_fn.png")

    row = {
        "scene": scene_name(dataset, instance),
        "instance": instance,
        "test_image": stem,
        "num_reference_images": len(reference_images),
        "iou": metrics.iou,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "tn": metrics.tn,
        "changed_pixel_fraction_native": float((labels != 0).mean()),
        "alignment_residual": paths["alignment_residual"],
        "output_dir": str(scene_dir),
    }
    (paths["intermediate_dir"] / "paslcd_result.json").write_text(
        json.dumps({**row, "decision_counts": inference.get("decision_counts"), "timings_seconds": inference.get("timings")}, indent=2)
    )
    append_metrics_row(scene_dir, row)
    write_scene_summary(scene_dir, scene_name(dataset, instance), dataset, instance)
    return row


def run_one_pair(
    dataset: str,
    instance: str,
    reference_images: list[Path],
    test_image: Path,
    instance_dir: Path,
    scene_dir: Path,
    config_path: Path,
    skip_refine: bool = False,
    reference_scene=None,
) -> dict[str, Any]:
    """Single-query convenience wrapper: reconstruct+refine, then run
    detect.py (one subprocess, one model load) rather than batching --
    for standalone/CLI use where there's only one query. See
    run_paslcd_scene.py's batched flow (reconstruct_and_refine_one_pair +
    detect_batch.py + evaluate_one_pair) for the multi-query path."""
    paths = reconstruct_and_refine_one_pair(
        dataset, instance, reference_images, test_image, instance_dir, scene_dir, config_path, skip_refine, reference_scene,
    )
    subprocess.run(
        [
            "conda", "run", "-n", DETECT_ENV, "python", str(REPO / "scripts/detect.py"),
            "--render-t0", str(paths["render_t0"]), "--clean-render", str(paths["clean_render"]),
            "--image-t1", str(paths["image_t1"]), "--config", str(config_path),
            "--output-dir", str(paths["detect_dir"]),
        ],
        check=True,
    )
    return evaluate_one_pair(dataset, instance, test_image, scene_dir, reference_images, paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "PASLCD")
    parser.add_argument("--dataset", required=True, help="e.g. Cantina")
    parser.add_argument("--instance", required=True, help="Instance_1 or Instance_2")
    parser.add_argument("--test-image", help="query image stem, e.g. Inst_1_test_IMG_2863 (default: first alphabetically)")
    parser.add_argument("--max-reference-images", type=int, default=None, help="uniformly subsample the reference set to this many images (default: use all)")
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "paslcd")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--skip-refine", action="store_true")
    args = parser.parse_args()

    instance_dir = args.data_root / args.dataset / args.instance
    reference_images, test_images = discover_instance_images(instance_dir)
    if args.max_reference_images:
        reference_images = even_subsample(reference_images, args.max_reference_images)

    if args.test_image:
        matches = [p for p in test_images if p.stem == args.test_image]
        if not matches:
            raise SystemExit(f"no query image with stem {args.test_image!r} found; available: {[p.stem for p in test_images]}")
        test_image = matches[0]
    else:
        test_image = test_images[0]

    scene_dir = args.output_root / scene_name(args.dataset, args.instance)
    row = run_one_pair(
        args.dataset, args.instance, reference_images, test_image, instance_dir, scene_dir, args.config, args.skip_refine,
    )
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
