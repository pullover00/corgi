from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO

import numpy as np
from PIL import Image
from tqdm import tqdm

from .adapters import Mast3rAdapter, Sam2Adapter
from .ablation import No3DPipeline
from .changesim import MetricAccumulator, deterministic_subset, load_manifest, normalize_target
from .config import load_config
from .io import load_rgb, save_image, save_json
from .model_paths import configure_mast3r_paths
from .pipeline import PairwisePipeline
from .provenance import measure_evaluation_provenance
from .report import build_evaluation_report
from .reproducibility import (
    measure_checkpoint_assets,
    measure_runtime,
    measure_source_assets,
    validate_runtime,
)
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
            "the external SAM3 source/checkpoint environment described in README.md"
        ),
    )
    changesim.add_argument(
        "--pipeline-config",
        default="configs/pipeline.yaml",
        help="merged pipeline config for --full-pipeline (default: configs/pipeline.yaml)",
    )
    changesim.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "optional: root of a pre-computed a3_overnight.py-style cache "
            "(see ocmask.weekend_cache, docs/cache_audit.md) that --full-pipeline "
            "will validate and reuse stages 1-4 from where possible, falling back "
            "to full computation per pair/stage otherwise. Machine-specific; "
            "unset by default, matching plain full-computation behavior."
        ),
    )
    changesim.add_argument(
        "--prediction-variant",
        choices=["guarded", "full"],
        default="guarded",
        help=(
            "full-pipeline raster to score: guarded is the conservative, "
            "heldout-preferred default; full reproduces the historical "
            "headline footprint"
        ),
    )
    changesim.add_argument(
        "--pair-retries",
        type=int,
        default=0,
        help="number of clean-process retries after a pair worker fails",
    )
    changesim.add_argument(
        "--pair-timeout-seconds",
        type=float,
        default=1800.0,
        help=(
            "maximum wall-clock seconds for each isolated pair attempt "
            "(default: 1800)"
        ),
    )
    changesim.add_argument(
        "--save-stage-artifacts",
        action="store_true",
        help=(
            "additionally persist every stage's intermediate evidence (proposals, "
            "descriptors, similarity matrices, accept/reject reasons, before/after "
            "label maps -- see ocmask.artifact_capture) under each pair's output "
            "directory, for later ablation studies and failure analysis. Off by "
            "default: does not change any prediction, only adds file writes."
        ),
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

    doctor = subparsers.add_parser(
        "doctor",
        help="verify full-production runtime, checkpoints, and source trees",
    )
    doctor.add_argument(
        "--pipeline-config",
        default="configs/pipeline.yaml",
        help="production pipeline config to validate (default: configs/pipeline.yaml)",
    )
    doctor.add_argument(
        "--stage1-only",
        action="store_true",
        help="run the legacy stage-1-only readiness check using the global --config",
    )
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


@contextmanager
def _exclusive_evaluation_output_lock(output: Path) -> Iterator[None]:
    """Hold a nonblocking Linux advisory lock for one full evaluation lifetime."""

    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".evaluation.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o644)
    lock_file: TextIO = os.fdopen(descriptor, "r+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            lock_file.seek(0)
            try:
                owner = json.loads(lock_file.read())
            except (json.JSONDecodeError, OSError):
                owner = {}
            owner_pid = owner.get("pid", "unknown")
            owner_host = owner.get("host", "unknown")
            raise RuntimeError(
                "Another full-pipeline evaluation is already running in "
                f"{output} (owner pid={owner_pid}, host={owner_host}; "
                f"lock={lock_path}). Wait for it to finish or choose a different "
                "--output directory."
            ) from None

        metadata = {
            "schema_version": 1,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at_unix": time.time(),
        }
        lock_file.seek(0)
        lock_file.truncate()
        json.dump(metadata, lock_file, sort_keys=True)
        lock_file.write("\n")
        lock_file.flush()
        os.fsync(lock_file.fileno())
        yield
    finally:
        if acquired:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _pair_attempt_log(
    pair_output: Path, pair_id: str, attempt: int
) -> tuple[Path, BinaryIO]:
    """Open a unique combined worker log beneath a validated pair directory."""

    pair_output = pair_output.resolve()
    log_directory = (pair_output / "worker_logs").resolve()
    if pair_output not in log_directory.parents:
        raise ValueError(f"Worker log directory escapes pair output for {pair_id!r}")
    log_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"worker-attempt-{attempt:03d}-pid-{os.getpid()}-{time.time_ns()}.log"
    )
    log_path = (log_directory / filename).resolve()
    if log_directory not in log_path.parents:
        raise ValueError(f"Worker log path escapes pair output for {pair_id!r}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(log_path, flags, 0o644)
    return log_path, os.fdopen(descriptor, "wb")


def evaluate_changesim_full_pipeline(args, pipeline_config: dict) -> int:
    """Freeze isolated predictions first, then score them in a second phase.

    Each image pair runs in a new Python process. This bounds SAM3/SAM2 GPU
    lifetime and prevents SAM3's global TF32/autocast changes from affecting
    the next pair's MASt3R reconstruction. Resume is allowed only when the
    manifest, config, code, model assets, runtime, numerical policy, cache,
    selection, and output variant have the same execution fingerprint.
    """

    if args.pair_retries < 0:
        raise ValueError("--pair-retries must be non-negative")
    pair_timeout_seconds = float(args.pair_timeout_seconds)
    if not math.isfinite(pair_timeout_seconds) or pair_timeout_seconds <= 0:
        raise ValueError("--pair-timeout-seconds must be a positive finite number")
    output = Path(args.output).resolve()
    with _exclusive_evaluation_output_lock(output):
        return _evaluate_changesim_full_pipeline_locked(
            args, pipeline_config, pair_timeout_seconds
        )


def _evaluate_changesim_full_pipeline_locked(
    args, pipeline_config: dict, pair_timeout_seconds: float
) -> int:
    """Run a full evaluation while the caller holds its output-directory lock."""

    from .reproducibility import (
        build_ground_truth_manifest,
        build_run_manifest,
        pair_input_identity,
        pair_output_directory,
        prepare_ground_truth_manifest,
        prepare_run_directory,
        sha256_file,
    )

    repository = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    pipeline_config_path = Path(args.pipeline_config).resolve()
    manifest_pairs = load_manifest(manifest_path, require_declared_classes=True)
    manifest_ids = [pair.pair_id for pair in manifest_pairs]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("ChangeSim manifest contains duplicate pair IDs")
    for pair_id in manifest_ids:
        pair_output_directory(output / "pairs", pair_id)
    pairs = deterministic_subset(
        manifest_pairs,
        args.fraction,
        pipeline_config["reconstruction"]["seed"],
    )
    selected_ids = [pair.pair_id for pair in pairs]
    inference_inputs = {
        pair.pair_id: pair_input_identity((pair.image0, pair.image1)) for pair in pairs
    }

    run_manifest = build_run_manifest(
        config=pipeline_config,
        config_path=pipeline_config_path,
        benchmark_manifest_path=manifest_path,
        repository=repository,
        prediction_variant=args.prediction_variant,
        cache_dir=args.cache_dir,
        fraction=args.fraction,
        selection_ids=selected_ids,
        inference_inputs=inference_inputs,
    )
    prepare_run_directory(output, run_manifest)
    run_fingerprint = run_manifest["run_fingerprint"]
    save_json(
        output / "selection.json",
        {
            "run_fingerprint": run_fingerprint,
            "seed": pipeline_config["reconstruction"]["seed"],
            "fraction": args.fraction,
            "ids": selected_ids,
        },
    )
    progress_path = output / "progress.jsonl"

    def append_progress(record: dict) -> None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    completed: dict[str, dict] = {}
    if progress_path.exists():
        for line_number, line in enumerate(
            progress_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed[str(record["id"])] = record
            except (json.JSONDecodeError, KeyError):
                print(
                    f"Ignoring incomplete progress record on line {line_number}",
                    flush=True,
                )

    scored_variants = ("guarded", "full")

    def validate_pair_freeze(pair) -> tuple[dict, dict[str, Path]] | None:
        pair_output = pair_output_directory(output / "pairs", pair.pair_id)
        marker_path = pair_output / "pair_manifest.json"
        if not marker_path.is_file():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if marker.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported pair marker schema: {marker_path}")
        if marker.get("id") != pair.pair_id:
            raise RuntimeError(f"Pair marker ID mismatch: {marker_path}")
        if marker.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Pair output belongs to a different run: {pair_output}. "
                "Choose a new --output directory."
            )
        expected_inputs = run_manifest["inference_inputs"][pair.pair_id]
        current_inputs = pair_input_identity((pair.image0, pair.image1))
        if current_inputs != expected_inputs:
            raise RuntimeError(
                f"Pair input content changed after run start: {pair.pair_id}"
            )
        if marker.get("inputs") != expected_inputs:
            raise RuntimeError(f"Pair input content changed after inference: {pair.pair_id}")
        prediction_paths: dict[str, Path] = {}
        for variant in scored_variants:
            prediction = marker.get("predictions", {}).get(variant)
            if not prediction:
                raise RuntimeError(
                    f"Pair marker lacks {variant} prediction: {marker_path}"
                )
            prediction_path = (pair_output / prediction["path"]).resolve()
            if pair_output.resolve() not in prediction_path.parents:
                raise RuntimeError(
                    f"{variant} prediction path escapes pair output: {prediction_path}"
                )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            if sha256_file(prediction_path) != prediction.get("sha256"):
                raise RuntimeError(
                    f"Frozen {variant} prediction hash mismatch: {prediction_path}"
                )
            prediction_paths[variant] = prediction_path
        return marker, prediction_paths

    frozen: dict[str, tuple[dict, dict[str, Path]]] = {}
    failures: list[dict] = []
    prediction_started = time.perf_counter()
    progress_bar = tqdm(pairs, total=len(pairs), desc="freezing predictions", unit="pair")
    for pair in progress_bar:
        progress_bar.set_postfix_str(pair.pair_id)
        try:
            validated = validate_pair_freeze(pair)
            if validated is None:
                command = [
                    sys.executable,
                    "-m",
                    "ocmask.full_pair_worker",
                    "--image0",
                    str(pair.image0),
                    "--image1",
                    str(pair.image1),
                    "--output",
                    str(pair_output_directory(output / "pairs", pair.pair_id)),
                    "--pair-id",
                    pair.pair_id,
                    "--pipeline-config",
                    str(pipeline_config_path),
                    "--repository",
                    str(repository),
                    "--run-fingerprint",
                    run_fingerprint,
                    "--run-manifest",
                    str(output / "run_manifest.json"),
                ]
                if args.cache_dir:
                    command.extend(("--cache-dir", str(Path(args.cache_dir).resolve())))
                if args.save_stage_artifacts:
                    command.append("--save-stage-artifacts")
                last_error = None
                worker_environment = os.environ.copy()
                source_path = str(repository / "src")
                worker_environment["PYTHONHASHSEED"] = str(
                    pipeline_config["reconstruction"]["seed"]
                )
                worker_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                worker_environment["PYTHONPATH"] = os.pathsep.join(
                    value
                    for value in (source_path, worker_environment.get("PYTHONPATH"))
                    if value
                )
                pair_output = pair_output_directory(output / "pairs", pair.pair_id)
                for attempt in range(1, args.pair_retries + 2):
                    log_path, log_stream = _pair_attempt_log(
                        pair_output, pair.pair_id, attempt
                    )
                    try:
                        try:
                            result = subprocess.run(
                                command,
                                cwd=repository,
                                env=worker_environment,
                                check=False,
                                stdout=log_stream,
                                stderr=subprocess.STDOUT,
                                timeout=pair_timeout_seconds,
                            )
                        except subprocess.TimeoutExpired:
                            last_error = RuntimeError(
                                "isolated worker timed out after "
                                f"{pair_timeout_seconds:g} seconds "
                                f"(attempt {attempt}/{args.pair_retries + 1}; "
                                f"log={log_path})"
                            )
                            continue
                    finally:
                        log_stream.flush()
                        os.fsync(log_stream.fileno())
                        log_stream.close()
                    if result.returncode == 0:
                        last_error = None
                        break
                    last_error = RuntimeError(
                        f"isolated worker exited with code {result.returncode} "
                        f"(attempt {attempt}/{args.pair_retries + 1}; "
                        f"log={log_path})"
                    )
                if last_error is not None:
                    raise last_error
                validated = validate_pair_freeze(pair)
                if validated is None:
                    raise RuntimeError("worker exited successfully without a pair freeze marker")
            marker, prediction_paths = validated
            frozen[pair.pair_id] = validated
            prediction_path = prediction_paths[args.prediction_variant]
            previous = completed.get(pair.pair_id, {})
            if (
                previous.get("status") != "prediction_frozen"
                or previous.get("run_fingerprint") != run_fingerprint
                or previous.get("prediction_variant") != args.prediction_variant
                or previous.get("prediction_sha256")
                != marker["predictions"][args.prediction_variant]["sha256"]
            ):
                record = {
                    "id": pair.pair_id,
                    "status": "prediction_frozen",
                    "run_fingerprint": run_fingerprint,
                    "prediction_variant": args.prediction_variant,
                    "prediction": str(prediction_path.relative_to(output)),
                    "prediction_sha256": marker["predictions"][args.prediction_variant]["sha256"],
                    "pair_manifest_sha256": sha256_file(
                        pair_output_directory(output / "pairs", pair.pair_id)
                        / "pair_manifest.json"
                    ),
                    "timings": marker.get("timings", {}),
                    "ground_truth_opened": False,
                }
                append_progress(record)
                completed[pair.pair_id] = record
        except Exception as exc:
            failure = {
                "id": pair.pair_id,
                "status": "failure",
                "run_fingerprint": run_fingerprint,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            append_progress(failure)
            tqdm.write(f"{pair.pair_id} FAILED: {exc}")
            if not args.continue_on_error:
                save_json(output / "failures.json", failures)
                raise
    prediction_elapsed = time.perf_counter() - prediction_started
    # Always replace this summary, including with [], so a recovered resume
    # cannot leave a stale failures file from an earlier attempt.
    save_json(output / "failures.json", failures)
    missing_prediction_ids = [
        pair.pair_id for pair in pairs if pair.pair_id not in frozen
    ]
    if missing_prediction_ids:
        preview = ", ".join(missing_prediction_ids[:10])
        if len(missing_prediction_ids) > 10:
            preview += f", ... (+{len(missing_prediction_ids) - 10} more)"
        raise RuntimeError(
            "Cannot score an incomplete prediction freeze: "
            f"{len(missing_prediction_ids)} of {len(pairs)} selected pairs lack "
            f"validated predictions ({preview}). Ground truth was not opened; "
            "rerun the same command to resume the missing workers."
        )

    # Ground truth is first opened only here, after every selected inference
    # worker has exited and every selected prediction has a SHA-256 marker.
    # Its content identity is immutable on resume, while deliberately absent
    # from the pre-inference run manifest.
    ground_truth_manifest = build_ground_truth_manifest(
        run_fingerprint=run_fingerprint,
        targets={
            pair.pair_id: pair_input_identity((pair.target,)) for pair in pairs
        },
    )
    prepare_ground_truth_manifest(output, ground_truth_manifest)
    scoring_started = time.perf_counter()
    accumulators = {variant: MetricAccumulator() for variant in scored_variants}
    sequence_accumulators: dict[str, dict[str, MetricAccumulator]] = {
        variant: {} for variant in scored_variants
    }
    warehouse_accumulators: dict[str, dict[str, MetricAccumulator]] = {
        variant: {} for variant in scored_variants
    }
    per_pair: list[dict] = []
    for pair in tqdm(pairs, desc="scoring frozen predictions", unit="pair"):
        if pair.pair_id not in frozen:
            continue
        # A complete ChangeSim run can spend days in phase 1. Revalidate the
        # live marker, input identities, prediction paths, and prediction
        # hashes immediately before consuming any pixels; the phase-1 result
        # retained in ``frozen`` is only proof that the global GT gate opened.
        rescored_freeze = validate_pair_freeze(pair)
        if rescored_freeze is None:
            raise RuntimeError(
                f"Frozen prediction marker disappeared before scoring: {pair.pair_id}"
            )
        marker, prediction_paths = rescored_freeze
        target = normalize_target(pair.target)
        expected_target_identity = ground_truth_manifest["targets"][pair.pair_id]
        if pair_input_identity((pair.target,)) != expected_target_identity:
            raise RuntimeError(
                f"Ground truth changed while scoring pair {pair.pair_id}"
            )
        warehouse = pair.image1.parents[2].name
        sequence = f"{warehouse}/{pair.image1.parents[1].name}"
        pair_variants: dict[str, dict] = {}
        for variant in scored_variants:
            with Image.open(prediction_paths[variant]) as prediction_image:
                prediction = np.asarray(prediction_image, dtype=np.uint8).copy()
            pair_accumulator = MetricAccumulator()
            pair_accumulator.add(prediction, target)
            accumulators[variant].add_confusion(pair_accumulator.confusion)
            sequence_accumulators[variant].setdefault(
                sequence, MetricAccumulator()
            ).add_confusion(pair_accumulator.confusion)
            warehouse_accumulators[variant].setdefault(
                warehouse, MetricAccumulator()
            ).add_confusion(pair_accumulator.confusion)
            pair_variants[variant] = {
                "prediction_sha256": marker["predictions"][variant]["sha256"],
                "confusion": pair_accumulator.confusion.tolist(),
            }
        selected_pair = pair_variants[args.prediction_variant]
        per_pair.append(
            {
                "id": pair.pair_id,
                "status": "success",
                "warehouse": warehouse,
                "sequence": sequence,
                "run_fingerprint": run_fingerprint,
                "prediction_variant": args.prediction_variant,
                "prediction_sha256": selected_pair["prediction_sha256"],
                "target_sha256": next(iter(expected_target_identity.values())),
                "confusion": selected_pair["confusion"],
                "prediction_variants": pair_variants,
                "timings": marker.get("timings", {}),
                "ground_truth_used_in_inference": False,
            }
        )

    def summarize_variant(
        accumulator: MetricAccumulator,
        per_sequence_accumulators: dict[str, MetricAccumulator],
        per_warehouse_accumulators: dict[str, MetricAccumulator],
    ) -> dict:
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
                    for name in (
                        "added",
                        "removed",
                        "moved",
                        "replaced",
                        "unchanged",
                    )
                },
                "miou": metrics["multiclass_miou"] * 100,
            },
        }
        per_sequence = {
            name: {"pairs": value.count, "metrics": value.compute()}
            for name, value in sorted(per_sequence_accumulators.items())
        }
        per_warehouse = {
            name: {"pairs": value.count, "metrics": value.compute()}
            for name, value in sorted(per_warehouse_accumulators.items())
        }
        binary_by_sequence = np.asarray(
            [value["metrics"]["binary_miou"] for value in per_sequence.values()],
            dtype=np.float64,
        )
        multiclass_by_sequence = np.asarray(
            [
                value["metrics"]["multiclass_miou"]
                for value in per_sequence.values()
            ],
            dtype=np.float64,
        )
        return {
            "pairs_scored": accumulator.count,
            "metrics": metrics,
            "confusion_matrix": accumulator.confusion.tolist(),
            "table3_iou_percent": table3_iou_percent,
            "per_sequence": per_sequence,
            "per_warehouse": per_warehouse,
            "sequence_macro": {
                "binary_miou_mean": float(binary_by_sequence.mean()),
                "binary_miou_stddev": float(binary_by_sequence.std()),
                "binary_miou_worst": float(binary_by_sequence.min()),
                "multiclass_miou_mean": float(multiclass_by_sequence.mean()),
                "multiclass_miou_stddev": float(multiclass_by_sequence.std()),
                "multiclass_miou_worst": float(multiclass_by_sequence.min()),
            },
        }

    prediction_variants = {
        variant: summarize_variant(
            accumulators[variant], sequence_accumulators[variant], warehouse_accumulators[variant]
        )
        for variant in scored_variants
    }
    selected_summary = prediction_variants[args.prediction_variant]
    metrics = selected_summary["metrics"]
    table3_iou_percent = selected_summary["table3_iou_percent"]
    per_sequence = selected_summary["per_sequence"]
    per_warehouse = selected_summary["per_warehouse"]
    sequence_macro = selected_summary["sequence_macro"]
    report = {
        "protocol": {
            "dataset": "ChangeSim",
            "method": "object_consistent_masks_full_pipeline",
            "manifest": str(manifest_path),
            "fraction": args.fraction,
            "seed": pipeline_config["reconstruction"]["seed"],
            "cache_dir": str(Path(args.cache_dir).resolve()) if args.cache_dir else None,
            "prediction_variant": args.prediction_variant,
            "prediction_variants_scored": list(scored_variants),
            "pair_process_isolation": True,
            "pair_retries": args.pair_retries,
            "pair_timeout_seconds": pair_timeout_seconds,
            "save_stage_artifacts": args.save_stage_artifacts,
            "run_fingerprint": run_fingerprint,
            "ground_truth_fingerprint": ground_truth_manifest[
                "ground_truth_fingerprint"
            ],
            "predictions_frozen_before_ground_truth_scoring": True,
            "ground_truth_used_in_inference": False,
            "pairs_selected": len(pairs),
            "pairs_succeeded": len(per_pair),
        },
        "metrics": metrics,
        "table3_iou_percent": table3_iou_percent,
        "per_sequence": per_sequence,
        "per_warehouse": per_warehouse,
        "sequence_macro": sequence_macro,
        "prediction_variants": prediction_variants,
        "timing": {
            "prediction_seconds": prediction_elapsed,
            "scoring_seconds": time.perf_counter() - scoring_started,
        },
        "failures": failures,
        "pairs": per_pair,
    }
    save_json(output / "report.json", report)
    # metrics (report["metrics"], saved above in full) already has
    # precision/recall/f1/iou for every class plus multiclass_miou/
    # multiclass_macro_f1/binary_miou/binary_macro_f1 -- this stdout summary
    # is a compact excerpt, not a second computation of any of it.
    print(
        json.dumps(
            {
                "binary_miou_percent": metrics["binary_miou"] * 100,
                "binary_macro_f1_percent": metrics["binary_macro_f1"] * 100,
                "multiclass_miou_percent": metrics["multiclass_miou"] * 100,
                "multiclass_macro_f1_percent": metrics["multiclass_macro_f1"] * 100,
                "table3_iou_percent": table3_iou_percent,
                "prediction_variants": {
                    variant: {
                        "binary_miou_percent": summary["metrics"]["binary_miou"]
                        * 100,
                        "multiclass_miou_percent": summary["metrics"][
                            "multiclass_miou"
                        ]
                        * 100,
                    }
                    for variant, summary in prediction_variants.items()
                },
                "pairs_succeeded": len(per_pair),
                "failures": len(failures),
                "run_fingerprint": run_fingerprint,
                "report": str((output / "report.json").resolve()),
            },
            indent=2,
        )
    )
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


def _doctor_error(exc: Exception) -> dict[str, str]:
    """Return a stable, machine-readable failure with an actionable message."""

    return {
        "type": type(exc).__name__,
        "message": str(exc) or repr(exc),
    }


def _doctor_runtime_summary(runtime: dict) -> dict:
    """Keep doctor output useful without dumping the full distribution inventory."""

    keys = (
        "python",
        "implementation",
        "executable",
        "platform",
        "packages",
        "torch_cuda",
        "cudnn",
        "cuda_available",
        "gpu",
        "gpu_compute_capability",
        "gpu_vram_bytes",
        "installed_distributions_sha256",
    )
    summary = {key: runtime[key] for key in keys if key in runtime}
    distributions = runtime.get("installed_distributions")
    if isinstance(distributions, list):
        summary["installed_distributions_count"] = len(distributions)
    return summary


def production_doctor_command(
    pipeline_config_path: str | Path,
    *,
    repository: str | Path | None = None,
) -> int:
    """Validate every production execution dependency without model inference."""

    config_path = Path(pipeline_config_path).resolve()
    repository_path = (
        Path(repository).resolve()
        if repository is not None
        else Path(__file__).resolve().parents[2]
    )
    report = {
        "schema_version": 1,
        "mode": "full_pipeline_production",
        "ready": False,
        "model_inference_performed": False,
        "pipeline_config": str(config_path),
        "repository": str(repository_path),
        "checks": {},
    }

    try:
        config = load_config(config_path)
    except Exception as exc:
        report["checks"]["configuration"] = {
            "ok": False,
            "error": _doctor_error(exc),
        }
        report["failed_checks"] = ["configuration"]
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    report["checks"]["configuration"] = {"ok": True}

    runtime_check = {"ok": False}
    try:
        runtime = measure_runtime()
        runtime_check["actual"] = _doctor_runtime_summary(runtime)
        expected_runtime = config["reproducibility"]["expected_runtime"]
        runtime_check["expected"] = expected_runtime
        validate_runtime(runtime, expected_runtime)
    except Exception as exc:
        runtime_check["error"] = _doctor_error(exc)
    else:
        runtime_check["ok"] = True
    report["checks"]["runtime"] = runtime_check

    for check_name, measure in (
        ("checkpoints", measure_checkpoint_assets),
        ("sources", measure_source_assets),
    ):
        try:
            assets = measure(config, repository_path)
        except Exception as exc:
            report["checks"][check_name] = {
                "ok": False,
                "error": _doctor_error(exc),
            }
        else:
            report["checks"][check_name] = {
                "ok": True,
                "assets": assets,
            }

    failed_checks = [
        name for name, check in report["checks"].items() if not check["ok"]
    ]
    report["failed_checks"] = failed_checks
    report["ready"] = not failed_checks
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


def doctor_command(config: dict) -> int:
    """Run the retained stage-1-only readiness check without inference."""
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
    if args.command == "doctor":
        if args.stage1_only:
            return doctor_command(load_config(args.config))
        return production_doctor_command(args.pipeline_config)
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
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
