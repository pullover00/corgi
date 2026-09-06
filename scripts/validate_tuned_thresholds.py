#!/usr/bin/env python3
"""Validate tuned identity thresholds on a real end-to-end run, reusing
already-reconstructed render_t0/clean_render/target.png from completed
baseline queries -- only stage 3 (detect) reruns, with the tuned config,
writing to a separate output tree so the baseline results are untouched.

This is the real-run confirmation the threshold sweep's own docstring asks
for: that sweep only replays the direct_identity path offline; this script
runs the actual pipeline (direct_identity + clean_bridge_identity +
tracking_recovery all included) so we know the tuned thresholds still help
once every decision path that uses them is exercised for real.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_paslcd_pair as pair  # noqa: E402


def find_baseline_queries(results_root: Path, data_root: Path) -> list[dict]:
    """Every completed query with a baseline metrics.csv row -- reuses its
    already-rendered render_t0/clean_render/target.png and already-recorded
    baseline metrics for direct comparison."""
    found = []
    for scene_dir in sorted(results_root.iterdir()):
        if not scene_dir.is_dir() or "_Instance_" not in scene_dir.name:
            continue
        metrics_path = scene_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        parts = scene_dir.name.split("_")
        instance = "_".join(parts[-2:])
        dataset = "_".join(parts[:-2])
        with metrics_path.open() as handle:
            baseline_rows = {row["test_image"]: row for row in csv.DictReader(handle)}
        for stem, row in baseline_rows.items():
            recon_dir = scene_dir / "intermediate" / stem / "reconstruction"
            if not (recon_dir / "render_t0.png").exists():
                continue
            gt_path = data_root / dataset / instance / "gt_mask" / f"{stem}.png"
            if not gt_path.exists():
                continue
            found.append({
                "dataset": dataset, "instance": instance, "stem": stem,
                "render_t0": recon_dir / "render_t0.png", "clean_render": recon_dir / "clean_render.png",
                "image_t1": recon_dir / "image_t1.png", "gt_path": gt_path,
                "baseline": row,
            })
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=REPO / "results" / "paslcd")
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "PASLCD")
    parser.add_argument("--tuned-config", type=Path, default=REPO / "configs/pipeline_tuned.yaml")
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "paslcd_tuned_validation")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from ocmask_pipeline.metrics import binarize_prediction, compute_binary_metrics, load_paslcd_gt

    queries = find_baseline_queries(args.results_root, args.data_root)
    if args.limit:
        queries = queries[: args.limit]
    print(f"validating tuned thresholds on {len(queries)} already-reconstructed queries", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "render_t0": str(q["render_t0"]), "clean_render": str(q["clean_render"]), "image_t1": str(q["image_t1"]),
            "output_dir": str(args.output_root / q["dataset"] / q["instance"] / q["stem"] / "detect"),
        }
        for q in queries
    ]
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"running detect_batch.py once on all {len(manifest)} queries with the tuned config", flush=True)
    subprocess.run(
        [
            "conda", "run", "-n", pair.DETECT_ENV, "python", str(REPO / "scripts/detect_batch.py"),
            "--manifest", str(manifest_path), "--config", str(args.tuned_config),
        ],
        check=True,
    )

    comparisons = []
    for q, item in zip(queries, manifest):
        detect_dir = Path(item["output_dir"])
        labels = np.asarray(Image.open(detect_dir / "labels.png"))
        binary_native = (labels != 0).astype(np.uint8) * 255
        gt = load_paslcd_gt(q["gt_path"])
        pred = binarize_prediction(binary_native, gt.shape)
        metrics = compute_binary_metrics(gt, pred)
        baseline = q["baseline"]
        row = {
            "scene": f"{q['dataset']}_{q['instance']}", "test_image": q["stem"],
            "baseline_iou": float(baseline["iou"]), "tuned_iou": metrics.iou,
            "baseline_f1": float(baseline["f1"]), "tuned_f1": metrics.f1,
            "baseline_precision": float(baseline["precision"]), "tuned_precision": metrics.precision,
            "baseline_recall": float(baseline["recall"]), "tuned_recall": metrics.recall,
        }
        comparisons.append(row)
        print(f"[{row['scene']}] {row['test_image']}: iou {row['baseline_iou']:.4f}->{row['tuned_iou']:.4f}  "
              f"f1 {row['baseline_f1']:.4f}->{row['tuned_f1']:.4f}", flush=True)

    def mean(key: str) -> float:
        return sum(r[key] for r in comparisons) / len(comparisons) if comparisons else 0.0

    summary = {
        "n_queries": len(comparisons),
        "mean_baseline_iou": mean("baseline_iou"), "mean_tuned_iou": mean("tuned_iou"),
        "mean_baseline_f1": mean("baseline_f1"), "mean_tuned_f1": mean("tuned_f1"),
        "mean_baseline_precision": mean("baseline_precision"), "mean_tuned_precision": mean("tuned_precision"),
        "mean_baseline_recall": mean("baseline_recall"), "mean_tuned_recall": mean("tuned_recall"),
        "num_improved_f1": sum(1 for r in comparisons if r["tuned_f1"] > r["baseline_f1"]),
        "num_regressed_f1": sum(1 for r in comparisons if r["tuned_f1"] < r["baseline_f1"]),
        "num_unchanged_f1": sum(1 for r in comparisons if r["tuned_f1"] == r["baseline_f1"]),
    }
    (args.output_root / "comparison.json").write_text(json.dumps({"summary": summary, "per_query": comparisons}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
