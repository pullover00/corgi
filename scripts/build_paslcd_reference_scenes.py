#!/usr/bin/env python3
"""Overnight batch: reconstruct every PASLCD scene instance's reference
("before") scene with either VGGT-Omega or MASt3R and save it to disk, so
later work (the per-query evaluation, and a MASt3R-vs-VGGT-Omega reference-
reconstruction ablation) can load it back instead of recomputing.

Each instance is reconstructed in its own subprocess with a hard timeout, so
one pathological instance (a hang, or runaway shared-GPU contention) cannot
stall the whole batch. Already-saved instances are skipped, so this is safe
to kill and resume. results/paslcd/reference_scenes/{vggt_omega,mast3r}/
end up holding one .npz per <Dataset>_<Instance> plus a shared batch_log.txt.

Run --method vggt_omega in the vggt-omega conda env; --method mast3r in the
goldilocs conda env (both have mast3r importable, but goldilocs is where the
checkpoint/model_paths convention already lives -- see reconstruction_mast3r.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_paslcd_pair as pair  # noqa: E402

ALL_DATASETS = [
    "Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
    "Playground", "Porch", "Pots", "Printing_area", "Zen",
]
ALL_INSTANCES = ["Instance_1", "Instance_2"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["vggt_omega", "mast3r"], required=True)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "PASLCD")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--instances", nargs="+", default=ALL_INSTANCES)
    parser.add_argument("--max-reference-images", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--python", default=sys.executable, help="interpreter to run the per-instance worker with")
    parser.add_argument("--timeout", type=float, default=2700.0, help="seconds before an instance is killed and skipped")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=120.0)
    args = parser.parse_args()

    output_root = args.output_root or (REPO / "results" / "paslcd" / "reference_scenes" / args.method)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "batch_log.txt"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    jobs = [(d, i) for d in args.datasets for i in args.instances]
    log(f"starting {args.method} reference-scene batch: {len(jobs)} instances, "
        f"max_reference_images={args.max_reference_images}, timeout={args.timeout:.0f}s")

    for dataset, instance in jobs:
        scene = pair.scene_name(dataset, instance)
        out_path = output_root / f"{scene}.npz"
        if out_path.exists():
            log(f"{scene}: already exists, skipping")
            continue

        instance_dir = args.data_root / dataset / instance
        if not instance_dir.exists():
            log(f"{scene}: instance dir not found, skipping")
            continue

        reference_images, _ = pair.discover_instance_images(instance_dir)
        if args.max_reference_images:
            reference_images = pair.even_subsample(reference_images, args.max_reference_images)

        for attempt in range(1, args.retries + 2):
            started = time.time()
            try:
                result = subprocess.run(
                    [
                        args.python, str(REPO / "scripts/_reconstruct_one_reference_scene.py"),
                        "--method", args.method,
                        "--images", *[str(p) for p in reference_images],
                        "--config", str(args.config),
                        "--output", str(out_path),
                    ],
                    timeout=args.timeout,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    log(f"{scene}: OK, {len(reference_images)} images, {time.time() - started:.1f}s -> {out_path}")
                    break
                raise RuntimeError(f"worker exited {result.returncode}: {result.stderr[-2000:]}")
            except subprocess.TimeoutExpired:
                log(f"{scene}: attempt {attempt} TIMED OUT after {args.timeout:.0f}s")
                error = "timeout"
            except Exception as exc:
                log(f"{scene}: attempt {attempt} failed: {exc}")
                error = str(exc)

            if attempt <= args.retries:
                log(f"{scene}: retrying in {args.retry_delay:.0f}s")
                time.sleep(args.retry_delay)
            else:
                log(f"{scene}: giving up after {attempt} attempts")
                (output_root / f"{scene}.error.txt").write_text(error)

    log("batch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
