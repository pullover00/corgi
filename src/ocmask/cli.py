from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .adapters import Mast3rAdapter, Sam2Adapter
from .ablation import No3DPipeline
from .changesim import MetricAccumulator, deterministic_subset, load_manifest, normalize_target
from .config import load_config
from .io import load_rgb, save_image, save_json
from .model_paths import configure_mast3r_paths
from .pipeline import PairwisePipeline
from .provenance import measure_evaluation_provenance
from .report import build_evaluation_report
from .visualization import colorize, overlay


def build_parser() -> argparse.ArgumentParser:
    """Define stable command-line interfaces for inference and evaluation."""
    parser = argparse.ArgumentParser(prog="ocmask", description="Object-Consistent Mask scene-change pipeline")
    parser.add_argument("--config", default="configs/stage01_reconstruction.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="infer changes for one image pair")
    infer.add_argument("--image0", required=True)
    infer.add_argument("--image1", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--force-reconstruction", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a benchmark")
    evaluation_sub = evaluate.add_subparsers(dest="dataset", required=True)
    changesim = evaluation_sub.add_parser("changesim")
    changesim.add_argument("--manifest", required=True)
    changesim.add_argument("--output", required=True)
    changesim.add_argument("--fraction", type=float, default=1.0)
    changesim.add_argument("--continue-on-error", action="store_true")
    changesim.add_argument("--ablation", choices=["none", "no-3d"], default="none")
    changesim.add_argument(
        "--artifact-level",
        choices=["metrics", "minimal", "cache", "full"],
        default="metrics",
        help=(
            "metrics saves no pair files; minimal saves labels; cache additionally "
            "saves lean reusable geometry; full saves all diagnostics"
        ),
    )
    changesim.add_argument(
        "--full-pipeline",
        action="store_true",
        help=(
            "run the complete object-consistent-masks method (all 11 stages, "
            "see ocmask.inference.run_pair) instead of the stage-1-only "
            "reconstruction baseline --ablation selects between; requires "
            "the SAM3/SAM3.1 environment described in README.md"
        ),
    )
    changesim.add_argument(
        "--pipeline-config",
        default="configs/pipeline.yaml",
        help="merged pipeline config for --full-pipeline (default: configs/pipeline.yaml)",
    )

    visualize = subparsers.add_parser("visualize", help="recreate visual outputs for an artifact directory")
    visualize.add_argument("--artifacts", required=True)

    report = subparsers.add_parser("report", help="build an HTML pipeline and error walkthrough")
    report.add_argument("--evaluation", required=True)
    report.add_argument("--manifest", required=True)
    report.add_argument("--output")

    provenance = subparsers.add_parser(
        "measure-provenance",
        help="measure how much of the canonical render's covered pixels originate from T1 rather than T0",
    )
    provenance.add_argument("--evaluation", required=True)
    provenance.add_argument("--manifest", required=True)
    provenance.add_argument("--output")

    subparsers.add_parser("doctor", help="report model/runtime readiness")
    return parser


def make_pipeline(config: dict, ablation: str = "none"):
    """Construct real adapters only when a model-backed command needs them."""
    segmentation = Sam2Adapter(config)
    if ablation == "no-3d":
        return No3DPipeline(config, segmentation)
    return PairwisePipeline(config, Mast3rAdapter(config), segmentation)


def infer_command(args, config: dict) -> int:
    """Execute and report one pairwise inference."""
    result = make_pipeline(config).run(
        args.image0, args.image1, args.output, force_reconstruction=args.force_reconstruction
    )
    print(json.dumps({"artifacts": str(result.artifacts_dir), "timings": result.timings}, indent=2))
    return 0


def evaluate_command(args, config: dict) -> int:
    """Evaluate a deterministic ChangeSim selection with resumable pair caches."""
    if getattr(args, "full_pipeline", False):
        from .config import load_config as _load_config

        return evaluate_changesim_full_pipeline(args, _load_config(args.pipeline_config))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pairs = deterministic_subset(load_manifest(args.manifest), args.fraction, config["seed"])
    selection_path = output / "selection.json"
    selected_ids = [pair.pair_id for pair in pairs]
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        if previous.get("ids") != selected_ids:
            raise ValueError(f"{output} contains a different evaluation selection")
    else:
        save_json(selection_path, {"seed": config["seed"], "fraction": args.fraction, "ids": selected_ids})
    progress_path = output / "progress.jsonl"
    completed = {}
    if progress_path.exists():
        for line_number, line in enumerate(
            progress_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip():
                try:
                    record = json.loads(line)
                    completed[record["id"]] = record
                except (json.JSONDecodeError, KeyError):
                    # A hard shutdown can truncate only the last append. The
                    # corresponding pair has no trustworthy completion marker
                    # and is therefore recomputed on resume.
                    print(
                        f"Ignoring incomplete progress record on line {line_number}",
                        flush=True,
                    )
    pipeline = make_pipeline(config, args.ablation)
    accumulator = MetricAccumulator()
    failures = []
    per_pair = []
    started = time.perf_counter()
    # Persist failures immediately so a long benchmark interrupted by a model or
    # data error still leaves an actionable record.
    for index, pair in enumerate(pairs, 1):
        try:
            target = normalize_target(pair.target)
            previous = completed.get(pair.pair_id)
            if previous and previous.get("status") == "success":
                if "confusion" in previous:
                    accumulator.add_confusion(previous["confusion"])
                    per_pair.append(previous)
                    print(f"[{index}/{len(pairs)}] {pair.pair_id} (cached)", flush=True)
                    continue
            result = pipeline.run(
                pair.image0,
                pair.image1,
                output / "pairs",
                artifact_level=args.artifact_level,
            )
            pair_accumulator = MetricAccumulator()
            pair_accumulator.add(result.labels, target)
            accumulator.add_confusion(pair_accumulator.confusion)
            record = {
                "id": pair.pair_id,
                "status": "success",
                "confusion": pair_accumulator.confusion.tolist(),
                "timings": result.timings,
            }
            if args.artifact_level != "metrics":
                record["artifacts"] = str(result.artifacts_dir.resolve())
            per_pair.append(record)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(f"[{index}/{len(pairs)}] {pair.pair_id}", flush=True)
        except Exception as exc:
            failure = {"id": pair.pair_id, "status": "failure", "type": type(exc).__name__, "message": str(exc)}
            failures.append(failure)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, separators=(",", ":")) + "\n")
            save_json(output / "failures.json", failures)
            if not args.continue_on_error:
                raise
    metrics = accumulator.compute()
    # Mirror the exact IoU columns of paper Table 3 in a compact machine-readable
    # block in addition to the detailed per-class metrics above.
    table3_iou_percent = {
        "binary": {
            "changed": metrics["binary"]["changed"]["iou"] * 100,
            "unchanged": metrics["binary"]["unchanged"]["iou"] * 100,
            "miou": metrics["binary_miou"] * 100,
        },
        "multiclass": {
            **{
                name: metrics["multiclass"][name]["iou"] * 100
                for name in ("added", "removed", "moved", "replaced", "unchanged")
            },
            "miou": metrics["multiclass_miou"] * 100,
        },
    }
    # Metrics are stored as fractions internally. Paper references and gaps are
    # explicitly percentages to prevent unit ambiguity in downstream analysis.
    report = {
        "protocol": {"dataset": "ChangeSim", "ablation": args.ablation, "fraction": args.fraction, "seed": config["seed"], "pairs_selected": len(pairs), "pairs_succeeded": len(per_pair)},
        "metrics": metrics,
        "table3_iou_percent": table3_iou_percent,
        "paper_reference_percent": {"binary_miou": 64.9, "multiclass_miou": 33.6},
        "gap_percentage_points": {
            "binary_miou": metrics["binary_miou"] * 100 - 64.9,
            "multiclass_miou": metrics["multiclass_miou"] * 100 - 33.6,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "failures": failures,
        "pairs": per_pair,
    }
    save_json(output / "report.json", report)
    print(json.dumps(report["metrics"], indent=2))
    return 0


def evaluate_changesim_full_pipeline(args, pipeline_config: dict) -> int:
    """Run the complete method (:func:`ocmask.inference.run_pair`) over a
    ChangeSim manifest and report the paper's Table-3-style metrics.

    Deliberately thin: no per-variant bookkeeping, no prediction-freeze
    ledger, no cross-stage selection.json validation -- those existed to
    keep a multi-script ablation study honest and have no purpose once
    there is only one composition to run. Progress is still checkpointed
    to a flat ``progress.jsonl`` (one line per completed pair) purely so a
    long GPU run can resume after an interruption; delete it to start over.
    """

    from .inference import run_pair
    from .model_paths import configure_mast3r_paths

    configure_mast3r_paths()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pairs = deterministic_subset(load_manifest(args.manifest), args.fraction, pipeline_config["reconstruction"]["seed"])
    progress_path = output / "progress.jsonl"
    completed: dict[str, dict] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["id"]] = record

    accumulator = MetricAccumulator()
    failures: list[dict] = []
    per_pair: list[dict] = []
    started = time.perf_counter()
    for index, pair in enumerate(pairs, 1):
        previous = completed.get(pair.pair_id)
        if previous and previous.get("status") == "success":
            accumulator.add_confusion(previous["confusion"])
            per_pair.append(previous)
            print(f"[{index}/{len(pairs)}] {pair.pair_id} (cached)", flush=True)
            continue
        try:
            target = normalize_target(pair.target)
            result = run_pair(pair.image0, pair.image1, output / "pairs" / pair.pair_id, pipeline_config)
            pair_accumulator = MetricAccumulator()
            pair_accumulator.add(result.labels, target)
            accumulator.add_confusion(pair_accumulator.confusion)
            record = {
                "id": pair.pair_id,
                "status": "success",
                "confusion": pair_accumulator.confusion.tolist(),
                "timings": result.timings,
            }
            per_pair.append(record)
        except Exception as exc:
            record = {"id": pair.pair_id, "status": "failure", "type": type(exc).__name__, "message": str(exc)}
            failures.append(record)
            print(f"[{index}/{len(pairs)}] {pair.pair_id} FAILED: {exc}", flush=True)
            if not args.continue_on_error:
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                raise
        else:
            print(f"[{index}/{len(pairs)}] {pair.pair_id}", flush=True)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    metrics = accumulator.compute()
    table3_iou_percent = {
        "binary": {
            "changed": metrics["binary"]["changed"]["iou"] * 100,
            "unchanged": metrics["binary"]["unchanged"]["iou"] * 100,
            "miou": metrics["binary_miou"] * 100,
        },
        "multiclass": {
            **{
                name: metrics["multiclass"][name]["iou"] * 100
                for name in ("added", "removed", "moved", "replaced", "unchanged")
            },
            "miou": metrics["multiclass_miou"] * 100,
        },
    }
    report = {
        "protocol": {
            "dataset": "ChangeSim",
            "method": "object_consistent_masks_full_pipeline",
            "manifest": str(Path(args.manifest).resolve()),
            "fraction": args.fraction,
            "seed": pipeline_config["reconstruction"]["seed"],
            "pairs_selected": len(pairs),
            "pairs_succeeded": len(per_pair),
        },
        "metrics": metrics,
        "table3_iou_percent": table3_iou_percent,
        "elapsed_seconds": time.perf_counter() - started,
        "failures": failures,
        "pairs": per_pair,
    }
    save_json(output / "report.json", report)
    print(json.dumps({"table3_iou_percent": table3_iou_percent, "pairs_succeeded": len(per_pair), "failures": len(failures)}, indent=2))
    return 0 if not failures else 1


def visualize_command(args) -> int:
    """Regenerate inexpensive visualization files from saved labels."""
    artifacts = Path(args.artifacts)
    labels = np.asarray(Image.open(artifacts / "labels.png"))
    inputs = json.loads((artifacts / "inputs.json").read_text())
    image = load_rgb(inputs["image1"], labels.shape[::-1])
    save_image(artifacts / "labels_color.png", colorize(labels))
    save_image(artifacts / "overlay.png", overlay(image, labels))
    print(str(artifacts / "overlay.png"))
    return 0


def report_command(args) -> int:
    """Generate an inspectable static report from completed evaluation artifacts."""
    result = build_evaluation_report(args.evaluation, args.manifest, args.output)
    print(str(result))
    return 0


def measure_provenance_command(args) -> int:
    """Measure T0-vs-T1 provenance of the canonical render across an evaluation."""
    result = measure_evaluation_provenance(args.evaluation, args.manifest)
    output = Path(args.output) if args.output else Path(args.evaluation) / "provenance.json"
    save_json(output, result)
    print(json.dumps(result["summary"], indent=2))
    return 0


def doctor_command(config: dict) -> int:
    """Check CUDA, model packages, and checkpoint availability without inference."""
    report = {"python": sys.version.split()[0], "device_requested": config["device"]}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except ImportError:
        report["torch"] = None
        report["cuda_available"] = False
    try:
        configure_mast3r_paths()
    except RuntimeError as exc:
        report["mast3r_source_error"] = str(exc)
    for name, module in (("mast3r", "mast3r"), ("sam2", "sam2")):
        try:
            __import__(module)
            report[name] = "installed"
        except ImportError:
            report[name] = "missing"
    checkpoints = {}
    for section in ("mast3r", "sam2"):
        checkpoint = Path(config[section]["checkpoint"])
        checkpoints[section] = {"path": str(checkpoint), "exists": checkpoint.exists()}
    report["checkpoints"] = checkpoints
    print(json.dumps(report, indent=2))
    ready = report["cuda_available"] and all(report[x] == "installed" for x in ("mast3r", "sam2")) and all(x["exists"] for x in checkpoints.values())
    return 0 if ready else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "infer":
        return infer_command(args, config)
    if args.command == "evaluate":
        return evaluate_command(args, config)
    if args.command == "visualize":
        return visualize_command(args)
    if args.command == "report":
        return report_command(args)
    if args.command == "measure-provenance":
        return measure_provenance_command(args)
    if args.command == "doctor":
        return doctor_command(config)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
