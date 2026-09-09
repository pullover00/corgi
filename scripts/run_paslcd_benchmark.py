#!/usr/bin/env python3
"""Evaluate every PASLCD scene instance and write a final table --
results/paslcd/benchmark_summary.csv/.json -- with one row per scene plus
an overall mean.

Thin loop over run_paslcd_scene.run_scene -- no orchestration logic is
duplicated. This is the expensive command: each query photo triggers its
own full VGGT-Omega joint reconstruction over the reference set (there is
no cross-query reference-reconstruction cache -- see reconstruction.py's
docstring: t0/t1 are always reconstructed jointly for one call). Use
--max-reference-images and/or --limit-per-instance to bound runtime while
iterating; drop them for the final reported numbers.

Run in the vggt-omega conda env, same as run_paslcd_pair.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_paslcd_scene as scene_runner  # noqa: E402
import run_paslcd_pair as pair  # noqa: E402

ALL_DATASETS = [
    "Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
    "Playground", "Porch", "Pots", "Printing_area", "Zen",
]
ALL_INSTANCES = ["Instance_1", "Instance_2"]

SUMMARY_FIELDS = ["scene", "dataset", "instance", "num_test_images_evaluated", "mean_iou", "mean_f1", "mean_precision", "mean_recall"]


def _run_instance_subprocess(dataset: str, instance: str, args, log_dir: Path) -> tuple[str, str, int, Path]:
    """One instance's full run_paslcd_scene.py, as its own OS process (its
    own Python interpreter and CUDA context) so N of these can run
    concurrently without sharing GPU state unsafely across threads."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset}_{instance}.log"
    cmd = [
        sys.executable, str(REPO / "scripts/run_paslcd_scene.py"),
        "--dataset", dataset, "--instance", instance,
        "--data-root", str(args.data_root), "--output-root", str(args.output_root),
        "--config", str(args.config),
    ]
    if args.max_reference_images:
        cmd += ["--max-reference-images", str(args.max_reference_images)]
    if args.limit_per_instance:
        cmd += ["--limit", str(args.limit_per_instance)]
    if args.skip_refine:
        cmd.append("--skip-refine")
    if args.reference_scene_cache_dir:
        cmd += ["--reference-scene-cache-dir", str(args.reference_scene_cache_dir)]
    if args.resume:
        cmd.append("--resume")
    if args.dump_inventory:
        cmd.append("--dump-inventory")
    if args.parallel_instances > 1:
        # detect_batch.py alone uses ~7.3GB; two don't fit on a 16GB GPU
        # (confirmed by direct OOM testing) even though reconstruction
        # alone parallelizes fine. Serialize just the detect stage across
        # all concurrent instance subprocesses via a shared flock file, so
        # reconstruction keeps overlapping while detect_batch runs one at a time.
        cmd += ["--detect-lock-file", str(args.output_root / "detect_batch.lock")]
    with log_path.open("w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return dataset, instance, result.returncode, log_path


def run_parallel(jobs: list[tuple[str, str]], args) -> list[dict]:
    log_dir = args.output_root / "parallel_logs"
    print(f"running {len(jobs)} instances with {args.parallel_instances} concurrent workers -- per-instance logs in {log_dir}", flush=True)
    scene_summaries = []
    with ThreadPoolExecutor(max_workers=args.parallel_instances) as pool:
        futures = {pool.submit(_run_instance_subprocess, dataset, instance, args, log_dir): (dataset, instance) for dataset, instance in jobs}
        for future in as_completed(futures):
            dataset, instance = futures[future]
            _, _, returncode, log_path = future.result()
            status = "OK" if returncode == 0 else f"FAILED (exit {returncode}, see {log_path})"
            print(f"=== {dataset}/{instance}: {status} ===", flush=True)
            scene_dir = args.output_root / pair.scene_name(dataset, instance)
            if (scene_dir / "metrics.csv").exists():
                scene_summaries.append(pair.write_scene_summary(scene_dir, pair.scene_name(dataset, instance), dataset, instance))
    return scene_summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "PASLCD")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--instances", nargs="+", default=ALL_INSTANCES)
    parser.add_argument("--max-reference-images", type=int, default=None)
    parser.add_argument("--limit-per-instance", type=int, default=None, help="only evaluate the first N query images per instance")
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "paslcd")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--reference-scene-cache-dir", type=Path, default=None, help="reuse saved ReferenceScene .npz files (see build_paslcd_reference_scenes.py) instead of rebuilding each instance")
    parser.add_argument("--resume", action="store_true", help="skip query images already present in each scene's metrics.csv -- safe to re-run this command to continue an interrupted benchmark")
    parser.add_argument("--parallel-instances", type=int, default=1, help="run this many instances concurrently, each as its own subprocess/CUDA context (see run_parallel's docstring for the GPU-memory tradeoff). 1 = sequential, in-process (default).")
    parser.add_argument("--dump-inventory", action="store_true", help="save each query's stage-1-3 detection bundle for later detect-only replays (see run_paslcd_scene.py --dump-inventory)")
    args = parser.parse_args()

    jobs = [
        (dataset, instance)
        for dataset in args.datasets
        for instance in args.instances
        if (args.data_root / dataset / instance).exists()
    ]
    for dataset in args.datasets:
        for instance in args.instances:
            if not (args.data_root / dataset / instance).exists():
                print(f"skipping {dataset}/{instance}: not found", flush=True)

    if args.parallel_instances > 1:
        scene_summaries = run_parallel(jobs, args)
    else:
        scene_summaries = []
        for dataset, instance in jobs:
            print(f"=== {dataset}/{instance} ===", flush=True)
            summary = scene_runner.run_scene(
                args.data_root, dataset, instance, args.output_root, args.config,
                args.max_reference_images, args.skip_refine, args.limit_per_instance,
                args.reference_scene_cache_dir, args.resume, dump_inventory=args.dump_inventory,
            )
            scene_summaries.append(summary)

    def overall_mean(field: str) -> float:
        return sum(s[field] for s in scene_summaries) / len(scene_summaries) if scene_summaries else 0.0

    # Pooled mean = mean over every individual query image across all
    # scenes (each image weighted equally). Mean-of-scene-means above
    # instead weights each scene equally regardless of how many of its
    # images were evaluated. The two coincide when every scene contributes
    # the same number of images (the PASLCD default, 25 each); they can
    # diverge under --limit-per-instance or a partial/resumed run.
    all_rows = []
    for s in scene_summaries:
        scene_dir = args.output_root / s["scene"]
        metrics_path = scene_dir / "metrics.csv"
        if metrics_path.exists():
            with metrics_path.open() as handle:
                all_rows.extend(csv.DictReader(handle))

    def pooled_mean(field: str) -> float:
        return sum(float(r[field]) for r in all_rows) / len(all_rows) if all_rows else 0.0

    overall = {
        "scene": "OVERALL",
        "dataset": "",
        "instance": "",
        "num_test_images_evaluated": sum(s["num_test_images_evaluated"] for s in scene_summaries),
        "mean_iou": overall_mean("mean_iou"),
        "mean_f1": overall_mean("mean_f1"),
        "mean_precision": overall_mean("mean_precision"),
        "mean_recall": overall_mean("mean_recall"),
        "pooled_mean_iou": pooled_mean("iou"),
        "pooled_mean_f1": pooled_mean("f1"),
        "pooled_mean_precision": pooled_mean("precision"),
        "pooled_mean_recall": pooled_mean("recall"),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for s in scene_summaries:
            writer.writerow({key: s[key] for key in SUMMARY_FIELDS})
        writer.writerow({key: overall[key] for key in SUMMARY_FIELDS})
    (args.output_root / "benchmark_summary.json").write_text(
        json.dumps({"scenes": scene_summaries, "overall": overall}, indent=2)
    )

    print(json.dumps(overall, indent=2))
    print(f"wrote {args.output_root / 'benchmark_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
