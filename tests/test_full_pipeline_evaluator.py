from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import ocmask.reproducibility as reproducibility
from ocmask.changesim import ChangeSimPair
from ocmask.cli import evaluate_changesim_full_pipeline
from ocmask.io import save_json


def _arguments(
    tmp_path: Path,
    *,
    continue_on_error: bool = False,
    prediction_variant: str = "guarded",
    pair_retries: int = 0,
    pair_timeout_seconds: float = 1800.0,
    save_stage_artifacts: bool = False,
):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        output=str(tmp_path / "evaluation"),
        manifest=str(manifest_path),
        pipeline_config=str(config_path),
        cache_dir=None,
        fraction=1.0,
        prediction_variant=prediction_variant,
        pair_retries=pair_retries,
        pair_timeout_seconds=pair_timeout_seconds,
        continue_on_error=continue_on_error,
        save_stage_artifacts=save_stage_artifacts,
    )


def _pair(
    tmp_path: Path, *, sequence: str = "Seq_0", frame: str = "1"
) -> ChangeSimPair:
    image0 = tmp_path / "Warehouse" / sequence / "t0" / "rgb" / f"{frame}.png"
    image1 = tmp_path / "Warehouse" / sequence / "rgb" / f"{frame}.png"
    target = (
        tmp_path
        / "Warehouse"
        / sequence
        / "change_segmentation"
        / f"{frame}.png"
    )
    for path, value in ((image0, 10), (image1, 20), (target, 0)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((3, 4), value, dtype=np.uint8)).save(path)
    return ChangeSimPair(f"Warehouse_{sequence}_{frame}", image0, image1, target, ())


def _patch_lightweight_run_identity(monkeypatch) -> None:
    def build_run_manifest(**kwargs):
        return {
            "run_fingerprint": "test-run-fingerprint",
            "inference_inputs": dict(kwargs["inference_inputs"]),
        }

    def prepare_run_directory(output, manifest):
        path = Path(output) / "run_manifest.json"
        if not path.exists():
            save_json(path, manifest)

    monkeypatch.setattr(reproducibility, "build_run_manifest", build_run_manifest)
    monkeypatch.setattr(reproducibility, "prepare_run_directory", prepare_run_directory)


@pytest.mark.parametrize("selected_variant", ["guarded", "full"])
def test_full_evaluator_freezes_and_scores_both_variants_without_extra_worker(
    tmp_path: Path, monkeypatch, selected_variant: str
) -> None:
    args = _arguments(tmp_path, prediction_variant=selected_variant)
    pair = _pair(tmp_path)
    config = {"reconstruction": {"seed": 2026}}
    import ocmask.cli as cli

    monkeypatch.setattr(cli, "load_manifest", lambda *_args, **_kwargs: [pair])
    _patch_lightweight_run_identity(monkeypatch)
    original_pair_identity = reproducibility.pair_input_identity
    ground_truth_touched = False

    def observed_pair_identity(paths):
        nonlocal ground_truth_touched
        materialized = tuple(Path(path).resolve() for path in paths)
        if pair.target.resolve() in materialized:
            ground_truth_touched = True
        return original_pair_identity(materialized)

    monkeypatch.setattr(reproducibility, "pair_input_identity", observed_pair_identity)
    worker_calls = 0

    def fake_worker(command, *, cwd, env, check, stdout, stderr, timeout):
        nonlocal worker_calls
        worker_calls += 1
        assert not ground_truth_touched
        assert check is False
        assert stderr is subprocess.STDOUT
        assert timeout == 1800.0
        stdout.write(b"combined worker output\n")
        assert Path(cwd).name == "change_detect"
        assert env["PYTHONHASHSEED"] == "2026"
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert "--run-manifest" in command
        output = Path(command[command.index("--output") + 1])
        pair_id = command[command.index("--pair-id") + 1]
        run_fingerprint = command[command.index("--run-fingerprint") + 1]
        output.mkdir(parents=True, exist_ok=True)
        predictions = {}
        for variant, filename, value in (
            ("full", "labels.png", 1),
            ("guarded", "labels_guarded.png", 0),
            ("base", "labels_base.png", 0),
        ):
            path = output / filename
            Image.fromarray(np.full((3, 4), value, dtype=np.uint8)).save(path)
            predictions[variant] = {
                "path": filename,
                "sha256": reproducibility.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        save_json(
            output / "pair_manifest.json",
            {
                "schema_version": 1,
                "id": pair_id,
                "run_fingerprint": run_fingerprint,
                "inputs": original_pair_identity((pair.image0, pair.image1)),
                "predictions": predictions,
                "timings": {},
                "ground_truth_received_by_worker": False,
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_worker)

    assert evaluate_changesim_full_pipeline(args, config) == 0
    assert worker_calls == 1
    assert ground_truth_touched
    report = json.loads(
        (Path(args.output) / "report.json").read_text(encoding="utf-8")
    )
    assert report["protocol"]["predictions_frozen_before_ground_truth_scoring"]
    assert report["protocol"]["ground_truth_used_in_inference"] is False
    assert report["protocol"]["ground_truth_fingerprint"]
    assert report["protocol"]["prediction_variants_scored"] == ["guarded", "full"]
    assert report["protocol"]["pair_retries"] == 0
    assert report["protocol"]["pair_timeout_seconds"] == 1800.0
    assert set(report["prediction_variants"]) == {"guarded", "full"}
    selected = report["prediction_variants"][selected_variant]
    assert report["metrics"] == selected["metrics"]
    assert report["table3_iou_percent"] == selected["table3_iou_percent"]
    assert report["per_sequence"] == selected["per_sequence"]
    assert report["sequence_macro"] == selected["sequence_macro"]
    assert report["prediction_variants"]["guarded"]["metrics"][
        "multiclass_miou"
    ] == pytest.approx(1.0)
    assert report["prediction_variants"]["full"]["metrics"][
        "multiclass_miou"
    ] == pytest.approx(0.0)
    for variant in ("guarded", "full"):
        variant_summary = report["prediction_variants"][variant]
        assert variant_summary["pairs_scored"] == 1
        assert variant_summary["per_sequence"]["Warehouse/Seq_0"]["pairs"] == 1
        assert report["pairs"][0]["prediction_variants"][variant][
            "prediction_sha256"
        ]
    assert report["pairs"][0]["confusion"] == report["pairs"][0][
        "prediction_variants"
    ][selected_variant]["confusion"]
    assert json.loads(
        (Path(args.output) / "failures.json").read_text(encoding="utf-8")
    ) == []

    ground_truth_touched = False

    def worker_must_not_run(*_args, **_kwargs):
        raise AssertionError("valid frozen prediction should resume without a worker")

    monkeypatch.setattr(cli.subprocess, "run", worker_must_not_run)
    assert evaluate_changesim_full_pipeline(args, config) == 0
    assert worker_calls == 1
    assert ground_truth_touched


def test_full_evaluator_persists_worker_failures_without_opening_gt(
    tmp_path: Path, monkeypatch
) -> None:
    args = _arguments(tmp_path, continue_on_error=True)
    pair = _pair(tmp_path)
    config = {"reconstruction": {"seed": 2026}}
    import ocmask.cli as cli

    monkeypatch.setattr(cli, "load_manifest", lambda *_args, **_kwargs: [pair])
    _patch_lightweight_run_identity(monkeypatch)
    original_pair_identity = reproducibility.pair_input_identity
    ground_truth_touched = False

    def observed_pair_identity(paths):
        nonlocal ground_truth_touched
        materialized = tuple(Path(path).resolve() for path in paths)
        if pair.target.resolve() in materialized:
            ground_truth_touched = True
        return original_pair_identity(materialized)

    monkeypatch.setattr(reproducibility, "pair_input_identity", observed_pair_identity)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
    )

    with pytest.raises(RuntimeError, match="incomplete prediction freeze"):
        evaluate_changesim_full_pipeline(args, config)

    assert not ground_truth_touched
    failures = json.loads(
        (Path(args.output) / "failures.json").read_text(encoding="utf-8")
    )
    assert failures[0]["id"] == pair.pair_id
    assert failures[0]["status"] == "failure"
    assert "code 7" in failures[0]["message"]


def test_full_evaluator_refuses_partial_freeze_before_opening_any_gt(
    tmp_path: Path, monkeypatch
) -> None:
    args = _arguments(tmp_path, continue_on_error=True)
    successful_pair = _pair(tmp_path, sequence="Seq_0", frame="1")
    failed_pair = _pair(tmp_path, sequence="Seq_1", frame="2")
    pairs = [successful_pair, failed_pair]
    config = {"reconstruction": {"seed": 2026}}
    import ocmask.cli as cli

    monkeypatch.setattr(cli, "load_manifest", lambda *_args, **_kwargs: pairs)
    _patch_lightweight_run_identity(monkeypatch)
    original_pair_identity = reproducibility.pair_input_identity
    target_paths = {pair.target.resolve() for pair in pairs}
    ground_truth_touched = False

    def observed_pair_identity(paths):
        nonlocal ground_truth_touched
        materialized = tuple(Path(path).resolve() for path in paths)
        if target_paths.intersection(materialized):
            ground_truth_touched = True
        return original_pair_identity(materialized)

    monkeypatch.setattr(reproducibility, "pair_input_identity", observed_pair_identity)
    worker_calls: list[str] = []

    def one_success_one_failure(
        command, *, cwd, env, check, stdout, stderr, timeout
    ):
        assert not ground_truth_touched
        assert stderr is subprocess.STDOUT
        assert timeout == 1800.0
        pair_id = command[command.index("--pair-id") + 1]
        stdout.write(f"worker {pair_id}\n".encode())
        worker_calls.append(pair_id)
        if pair_id == failed_pair.pair_id:
            return SimpleNamespace(returncode=7)

        output = Path(command[command.index("--output") + 1])
        run_fingerprint = command[command.index("--run-fingerprint") + 1]
        output.mkdir(parents=True, exist_ok=True)
        predictions = {}
        for variant, filename in (
            ("full", "labels.png"),
            ("guarded", "labels_guarded.png"),
            ("base", "labels_base.png"),
        ):
            path = output / filename
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(path)
            predictions[variant] = {
                "path": filename,
                "sha256": reproducibility.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        save_json(
            output / "pair_manifest.json",
            {
                "schema_version": 1,
                "id": pair_id,
                "run_fingerprint": run_fingerprint,
                "inputs": original_pair_identity(
                    (successful_pair.image0, successful_pair.image1)
                ),
                "predictions": predictions,
                "timings": {},
                "ground_truth_received_by_worker": False,
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", one_success_one_failure)

    with pytest.raises(
        RuntimeError,
        match=r"1 of 2 selected pairs.*Ground truth was not opened",
    ):
        evaluate_changesim_full_pipeline(args, config)

    assert set(worker_calls) == {successful_pair.pair_id, failed_pair.pair_id}
    assert not ground_truth_touched
    output = Path(args.output)
    assert not (output / "ground_truth_manifest.json").exists()
    assert not (output / "report.json").exists()
    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
    assert [failure["id"] for failure in failures] == [failed_pair.pair_id]


def test_full_evaluator_revalidates_frozen_bytes_immediately_before_scoring(
    tmp_path: Path, monkeypatch
) -> None:
    args = _arguments(tmp_path)
    pair = _pair(tmp_path)
    config = {"reconstruction": {"seed": 2026}}
    import ocmask.cli as cli

    monkeypatch.setattr(cli, "load_manifest", lambda *_args, **_kwargs: [pair])
    _patch_lightweight_run_identity(monkeypatch)
    original_pair_identity = reproducibility.pair_input_identity
    guarded_path: Path | None = None

    def fake_worker(command, *, cwd, env, check, stdout, stderr, timeout):
        nonlocal guarded_path
        assert stderr is subprocess.STDOUT
        assert timeout == 1800.0
        stdout.write(b"worker output\n")
        output = Path(command[command.index("--output") + 1])
        pair_id = command[command.index("--pair-id") + 1]
        run_fingerprint = command[command.index("--run-fingerprint") + 1]
        output.mkdir(parents=True, exist_ok=True)
        predictions = {}
        for variant, filename in (
            ("full", "labels.png"),
            ("guarded", "labels_guarded.png"),
            ("base", "labels_base.png"),
        ):
            path = output / filename
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(path)
            predictions[variant] = {
                "path": filename,
                "sha256": reproducibility.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if variant == "guarded":
                guarded_path = path
        save_json(
            output / "pair_manifest.json",
            {
                "schema_version": 1,
                "id": pair_id,
                "run_fingerprint": run_fingerprint,
                "inputs": original_pair_identity((pair.image0, pair.image1)),
                "predictions": predictions,
                "timings": {},
                "ground_truth_received_by_worker": False,
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_worker)
    original_build_ground_truth_manifest = (
        reproducibility.build_ground_truth_manifest
    )

    def mutate_after_prediction_phase(**kwargs):
        assert guarded_path is not None
        # Keep a valid PNG and the same shape; only the frozen bytes/content
        # change after phase-1 validation and before the scoring loop.
        Image.fromarray(np.ones((3, 4), dtype=np.uint8)).save(guarded_path)
        return original_build_ground_truth_manifest(**kwargs)

    monkeypatch.setattr(
        reproducibility,
        "build_ground_truth_manifest",
        mutate_after_prediction_phase,
    )

    def target_must_not_be_scored(*_args, **_kwargs):
        raise AssertionError("altered prediction reached target scoring")

    monkeypatch.setattr(cli, "normalize_target", target_must_not_be_scored)

    with pytest.raises(RuntimeError, match="Frozen guarded prediction hash mismatch"):
        evaluate_changesim_full_pipeline(args, config)

    assert guarded_path is not None
    assert not (Path(args.output) / "report.json").exists()


def test_full_evaluator_output_lock_is_nonblocking_persistent_and_not_legacy(
    tmp_path: Path,
) -> None:
    import ocmask.cli as cli

    output = tmp_path / "evaluation"
    identity = {"schema_version": 1, "method": "test"}
    manifest = {
        **identity,
        "run_fingerprint": reproducibility.sha256_json(identity),
    }

    with cli._exclusive_evaluation_output_lock(output):
        metadata = json.loads(
            (output / ".evaluation.lock").read_text(encoding="utf-8")
        )
        assert metadata["pid"] > 0
        assert metadata["host"]
        with pytest.raises(
            RuntimeError,
            match=rf"already running.*pid={metadata['pid']}.*host={metadata['host']}",
        ):
            with cli._exclusive_evaluation_output_lock(output):
                raise AssertionError("a concurrent holder acquired the output lock")

        # Creating the persistent lock before the immutable run manifest must
        # not make a brand-new output directory look like a legacy evaluation.
        reproducibility.prepare_run_directory(output, manifest)

    assert (output / ".evaluation.lock").is_file()
    assert (output / "run_manifest.json").is_file()
    # The lock is released on context exit even though its metadata file stays.
    with cli._exclusive_evaluation_output_lock(output):
        pass


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_full_evaluator_rejects_nonpositive_or_nonfinite_pair_timeout(
    tmp_path: Path, timeout: float
) -> None:
    args = _arguments(tmp_path, pair_timeout_seconds=timeout)
    with pytest.raises(
        ValueError, match="--pair-timeout-seconds must be a positive finite number"
    ):
        evaluate_changesim_full_pipeline(args, {"reconstruction": {"seed": 2026}})


def test_full_evaluator_times_out_retries_and_fsyncs_combined_attempt_logs(
    tmp_path: Path, monkeypatch
) -> None:
    args = _arguments(
        tmp_path,
        continue_on_error=True,
        pair_retries=1,
        pair_timeout_seconds=0.25,
    )
    pair = _pair(tmp_path)
    config = {"reconstruction": {"seed": 2026}}
    import ocmask.cli as cli

    monkeypatch.setattr(cli, "load_manifest", lambda *_args, **_kwargs: [pair])
    _patch_lightweight_run_identity(monkeypatch)
    original_pair_identity = reproducibility.pair_input_identity
    ground_truth_touched = False

    def observed_pair_identity(paths):
        nonlocal ground_truth_touched
        materialized = tuple(Path(path).resolve() for path in paths)
        if pair.target.resolve() in materialized:
            ground_truth_touched = True
        return original_pair_identity(materialized)

    monkeypatch.setattr(reproducibility, "pair_input_identity", observed_pair_identity)
    attempts = 0
    real_fsync = os.fsync
    fsynced_attempt_logs: set[Path] = set()

    def observed_fsync(descriptor: int) -> None:
        try:
            descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            descriptor_path = None
        if descriptor_path is not None and descriptor_path.parent.name == "worker_logs":
            fsynced_attempt_logs.add(descriptor_path)
        real_fsync(descriptor)

    monkeypatch.setattr(cli.os, "fsync", observed_fsync)

    def timeout_worker(
        command, *, cwd, env, check, stdout, stderr, timeout
    ):
        nonlocal attempts
        attempts += 1
        assert not ground_truth_touched
        assert check is False
        assert stderr is subprocess.STDOUT
        assert timeout == 0.25
        stdout.write(f"attempt {attempts} stdout and stderr\n".encode())
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(cli.subprocess, "run", timeout_worker)

    with pytest.raises(
        RuntimeError,
        match=r"incomplete prediction freeze.*Ground truth was not opened",
    ):
        evaluate_changesim_full_pipeline(args, config)

    assert attempts == 2
    assert not ground_truth_touched
    output = Path(args.output)
    logs = sorted(
        (output / "pairs" / pair.pair_id / "worker_logs").glob(
            "worker-attempt-*.log"
        )
    )
    assert len(logs) == 2
    assert {path.read_text(encoding="utf-8") for path in logs} == {
        "attempt 1 stdout and stderr\n",
        "attempt 2 stdout and stderr\n",
    }
    assert fsynced_attempt_logs == set(logs)
    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
    assert len(failures) == 1
    assert "timed out after 0.25 seconds" in failures[0]["message"]
    assert "attempt 2/2" in failures[0]["message"]
    assert str(logs[1]) in failures[0]["message"]
    assert not (output / "ground_truth_manifest.json").exists()
    assert not (output / "report.json").exists()

    # An exception path must release the full-evaluation lock for a clean resume.
    lock_file = (output / ".evaluation.lock").open("r+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()
