"""One-process-per-pair worker for the complete GPU pipeline.

This module intentionally receives no ground-truth path.  It freezes model
predictions and their hashes; the parent evaluator opens ground truth only
after this process exits successfully.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from .config import load_config
from .io import save_json
from .numerics import finalize_pair_process, initialize_pair_process
from .reproducibility import (
    pair_input_identity,
    pair_output_directory,
    sha256_file,
    validate_worker_execution_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="isolated object-consistent-mask pair worker")
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-fingerprint", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--save-stage-artifacts",
        action="store_true",
        help="additionally persist every stage's intermediate evidence for ablation studies (see ocmask.artifact_capture)",
    )
    return parser


@contextmanager
def _exclusive_pair_output_lock(
    output: Path, pair_id: str, run_fingerprint: str
) -> Iterator[None]:
    """Serialize workers that can write one pair's output directory.

    The evaluator's run-level lock is owned by its parent process.  If that
    parent is killed while this worker survives, the run-level lock is
    released before the worker stops writing.  A resumed evaluator can then
    launch another worker for the same pair.  Keeping a second lock in the
    worker itself closes that hard-crash race without conflicting with the
    parent's distinct evaluation lock.
    """

    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".pair-worker.lock"
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
            print(
                "Waiting for an existing worker that still owns "
                f"{pair_id} (pid={owner.get('pid', 'unknown')}, "
                f"host={owner.get('host', 'unknown')})",
                flush=True,
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            acquired = True

        lock_file.seek(0)
        lock_file.truncate()
        json.dump(
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "pair_id": pair_id,
                "run_fingerprint": run_fingerprint,
                "started_at_unix": time.time(),
            },
            lock_file,
            sort_keys=True,
        )
        lock_file.write("\n")
        lock_file.flush()
        os.fsync(lock_file.fileno())
        yield
    finally:
        if acquired:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _completed_pair_freeze_is_valid(
    output: Path,
    *,
    pair_id: str,
    run_fingerprint: str,
    inputs: dict[str, str],
) -> bool:
    """Recognize a freeze completed by an orphan while a resume waited.

    ``False`` means no complete atomic marker exists and inference should run.
    A present marker whose identity or frozen files disagree is an integrity
    error, not permission to overwrite a purportedly completed pair.
    """

    marker_path = output / "pair_manifest.json"
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if marker.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported pair marker schema: {marker_path}")
    if marker.get("id") != pair_id:
        raise RuntimeError(f"Pair marker ID mismatch: {marker_path}")
    if marker.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(f"Pair marker run fingerprint mismatch: {marker_path}")
    if marker.get("inputs") != inputs:
        raise RuntimeError(f"Pair marker input identity mismatch: {marker_path}")

    output = output.resolve()
    for variant in ("full", "guarded", "base"):
        prediction = marker.get("predictions", {}).get(variant)
        if not isinstance(prediction, dict):
            raise RuntimeError(
                f"Pair marker lacks {variant} prediction: {marker_path}"
            )
        prediction_path = (output / str(prediction.get("path", ""))).resolve()
        if output not in prediction_path.parents:
            raise RuntimeError(
                f"{variant} prediction path escapes pair output: {prediction_path}"
            )
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        if sha256_file(prediction_path) != prediction.get("sha256"):
            raise RuntimeError(
                f"Frozen {variant} prediction hash mismatch: {prediction_path}"
            )
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(args.repository).resolve()
    os.chdir(repository)
    pipeline_config_path = Path(args.pipeline_config).resolve()
    config = load_config(pipeline_config_path)
    seed = int(config["reconstruction"]["seed"])
    # These must be fixed before importing torch/CUDA in this fresh process.
    # PYTHONHASHSEED is also set by the parent before interpreter startup;
    # assigning it here documents and verifies the worker contract for direct
    # invocations.
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(
            "Pair worker must be launched with PYTHONHASHSEED set before "
            f"interpreter startup (expected {seed})"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    output = Path(args.output).resolve()
    expected_output = pair_output_directory(output.parent, args.pair_id)
    if output != expected_output:
        raise ValueError(
            f"Worker output must be the pair-ID directory {expected_output}, got {output}"
        )

    with _exclusive_pair_output_lock(output, args.pair_id, args.run_fingerprint):
        # Acquire the write lock before importing torch or model code. A resume
        # waiting on an orphan therefore consumes neither GPU memory nor a
        # second model stack.
        input_identity = pair_input_identity((args.image0, args.image1))
        validate_worker_execution_identity(
            config=config,
            config_path=pipeline_config_path,
            repository=repository,
            run_manifest_path=args.run_manifest,
            run_fingerprint=args.run_fingerprint,
            pair_id=args.pair_id,
            inputs=input_identity,
            cache_dir=args.cache_dir,
        )
        if _completed_pair_freeze_is_valid(
            output,
            pair_id=args.pair_id,
            run_fingerprint=args.run_fingerprint,
            inputs=input_identity,
        ):
            return 0

        initial_state = initialize_pair_process(seed)

        # Import model orchestration only after the clean numerical policy is
        # in place. In particular, this prevents a future eager SAM3 import
        # from changing MASt3R before stage 1 starts.
        from .inference import run_pair
        from .model_paths import configure_mast3r_paths
        from .weekend_cache import ChangesimWeekendCache

        configure_mast3r_paths()
        cache = (
            ChangesimWeekendCache(args.cache_dir, config)
            if args.cache_dir
            else None
        )
        result = run_pair(
            Path(args.image0).resolve(),
            Path(args.image1).resolve(),
            output,
            config,
            pair_id=args.pair_id,
            cache=cache,
            save_stage_artifacts=args.save_stage_artifacts,
        )
        post_inference_state, restored_state = finalize_pair_process(initial_state)

        final_input_identity = pair_input_identity((args.image0, args.image1))
        if final_input_identity != input_identity:
            raise RuntimeError(
                "An inference input changed while the pair worker was running: "
                f"{args.pair_id}"
            )
        validate_worker_execution_identity(
            config=config,
            config_path=pipeline_config_path,
            repository=repository,
            run_manifest_path=args.run_manifest,
            run_fingerprint=args.run_fingerprint,
            pair_id=args.pair_id,
            inputs=final_input_identity,
            cache_dir=args.cache_dir,
        )

        predictions = {}
        for variant, filename in (
            ("full", "labels.png"),
            ("guarded", "labels_guarded.png"),
            ("base", "labels_base.png"),
        ):
            path = output / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Pair worker did not create {variant} prediction: {path}"
                )
            predictions[variant] = {
                "path": filename,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        save_json(
            output / "pair_manifest.json",
            {
                "schema_version": 1,
                "id": args.pair_id,
                "run_fingerprint": args.run_fingerprint,
                "seed": seed,
                "inputs": input_identity,
                "predictions": predictions,
                "timings": result.timings,
                "cache": result.diagnostics.get("cache", {}),
                "numerical_state_before": initial_state.to_dict(),
                "numerical_state_after_inference": post_inference_state.to_dict(),
                "numerical_state_after_cleanup": restored_state.to_dict(),
                "ground_truth_received_by_worker": False,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
