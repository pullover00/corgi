#!/usr/bin/env python3
"""Evaluate a diverse, time-budgeted sample of PASLCD query images: one
continuous process reconstructs (VGGT-Omega, reusing cached reference
scenes) queries drawn from as many different scene instances as possible,
then a single detect_batch.py call scores all of them together -- so
diversity (spreading across many instances) does not cost a repeated
SAM3/SAM2/DINOv2 cold-start per instance, only once total for the whole
sample.

Instance selection order interleaves round-robin across all 20 instances
(1st query of every instance, then 2nd query of every instance, ...) so a
prefix of any length is already maximally diverse -- picking N images
always covers min(N, 20) distinct instances before repeating any.

Writes into the normal per-scene results/paslcd/<Scene>/{metrics.csv,
summary.json} layout, so it composes with --resume on any later full run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_paslcd_pair as pair  # noqa: E402

ALL_DATASETS = [
    "Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
    "Playground", "Porch", "Pots", "Printing_area", "Zen",
]
ALL_INSTANCES = ["Instance_1", "Instance_2"]


def _already_evaluated(scene_dir: Path, stem: str) -> bool:
    metrics_path = scene_dir / "metrics.csv"
    if not metrics_path.exists():
        return False
    import csv

    with metrics_path.open() as handle:
        return any(row["test_image"] == stem for row in csv.DictReader(handle))


def build_round_robin_plan(
    data_root: Path, output_root: Path, max_reference_images: int | None, num_images: int, resume: bool,
) -> list[dict]:
    """One entry per (dataset, instance, query), ordered round-robin across
    instances so any prefix is maximally diverse."""
    per_instance = {}
    for dataset in ALL_DATASETS:
        for instance in ALL_INSTANCES:
            instance_dir = data_root / dataset / instance
            if not instance_dir.exists():
                continue
            reference_images, test_images = pair.discover_instance_images(instance_dir)
            if max_reference_images:
                reference_images = pair.even_subsample(reference_images, max_reference_images)
            scene_dir = output_root / pair.scene_name(dataset, instance)
            if resume:
                test_images = [t for t in test_images if not _already_evaluated(scene_dir, t.stem)]
            per_instance[(dataset, instance)] = {
                "instance_dir": instance_dir, "reference_images": reference_images,
                "scene_dir": scene_dir, "remaining": test_images,
            }

    plan = []
    round_index = 0
    while len(plan) < num_images:
        added_this_round = False
        for key, info in per_instance.items():
            if len(plan) >= num_images:
                break
            if round_index < len(info["remaining"]):
                dataset, instance = key
                plan.append({
                    "dataset": dataset, "instance": instance,
                    "instance_dir": info["instance_dir"], "reference_images": info["reference_images"],
                    "scene_dir": info["scene_dir"], "test_image": info["remaining"][round_index],
                })
                added_this_round = True
        if not added_this_round:
            break
        round_index += 1
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "PASLCD")
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "paslcd")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--max-reference-images", type=int, default=24)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--reference-scene-cache-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ocmask_pipeline.reconstruction import ReferenceScene

    plan = build_round_robin_plan(args.data_root, args.output_root, args.max_reference_images, args.num_images, args.resume)
    print(f"planned {len(plan)} images across {len({(p['dataset'], p['instance']) for p in plan})} distinct instances", flush=True)
    if not plan:
        print("nothing to do", flush=True)
        return 0

    started_overall = time.perf_counter()
    reference_scene_cache = {}
    all_paths = []
    for item in plan:
        key = (item["dataset"], item["instance"])
        if key not in reference_scene_cache:
            scene = None
            if args.reference_scene_cache_dir:
                cache_path = args.reference_scene_cache_dir / f"{pair.scene_name(*key)}.npz"
                if cache_path.exists():
                    scene = ReferenceScene.load(cache_path)
                    if [p.resolve() for p in scene.image_paths] != [p.resolve() for p in item["reference_images"]]:
                        scene = None
            if scene is None:
                print(f"[{key[0]}/{key[1]}] building reference scene (not cached)", flush=True)
                scene = pair.build_reference_scene(item["reference_images"], args.config)
            reference_scene_cache[key] = scene
        reference_scene = reference_scene_cache[key]

        started = time.perf_counter()
        paths = pair.reconstruct_and_refine_one_pair(
            item["dataset"], item["instance"], item["reference_images"], item["test_image"],
            item["instance_dir"], item["scene_dir"], args.config, args.skip_refine,
            reference_scene=reference_scene,
        )
        print(f"[{key[0]}/{key[1]}] reconstructed {item['test_image'].stem} ({time.perf_counter() - started:.1f}s)", flush=True)
        all_paths.append((item, paths))

    manifest = [
        {
            "render_t0": str(paths["render_t0"]), "clean_render": str(paths["clean_render"]),
            "image_t1": str(paths["image_t1"]), "output_dir": str(paths["detect_dir"]),
        }
        for _, paths in all_paths
    ]
    manifest_path = args.output_root / "diverse_sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"running detect_batch.py on all {len(manifest)} images in one pass (one model load total)", flush=True)
    started = time.perf_counter()
    subprocess.run(
        [
            "conda", "run", "-n", pair.DETECT_ENV, "python", str(pair.REPO / "scripts/detect_batch.py"),
            "--manifest", str(manifest_path), "--config", str(args.config),
        ],
        check=True,
    )
    print(f"detect_batch.py finished in {time.perf_counter() - started:.1f}s", flush=True)

    rows = []
    for item, paths in all_paths:
        row = pair.evaluate_one_pair(item["dataset"], item["instance"], item["test_image"], item["scene_dir"], item["reference_images"], paths)
        rows.append(row)
        print(f"[{item['dataset']}/{item['instance']}] {item['test_image'].stem} iou={row['iou']:.4f} f1={row['f1']:.4f}", flush=True)

    mean_iou = sum(r["iou"] for r in rows) / len(rows)
    mean_f1 = sum(r["f1"] for r in rows) / len(rows)
    print(json.dumps({
        "num_images": len(rows), "num_instances": len({(p["dataset"], p["instance"]) for p in plan}),
        "total_wall_seconds": time.perf_counter() - started_overall,
        "mean_iou": mean_iou, "mean_f1": mean_f1,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
