from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import ocmask.full_pair_worker as worker
from ocmask.cli import _exclusive_evaluation_output_lock
from ocmask.io import save_json
from ocmask.reproducibility import pair_input_identity, sha256_file


def _write_freeze(
    output: Path,
    *,
    pair_id: str,
    run_fingerprint: str,
    inputs: dict[str, str],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for variant, filename in (
        ("full", "labels.png"),
        ("guarded", "labels_guarded.png"),
        ("base", "labels_base.png"),
    ):
        path = output / filename
        path.write_bytes(f"{variant}-prediction".encode("ascii"))
        predictions[variant] = {
            "path": filename,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    save_json(
        output / "pair_manifest.json",
        {
            "schema_version": 1,
            "id": pair_id,
            "run_fingerprint": run_fingerprint,
            "inputs": inputs,
            "predictions": predictions,
        },
    )


def test_pair_output_lock_blocks_a_competing_worker_process(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    output = evaluation / "pairs" / "pair-1"
    attempted = tmp_path / "competitor-started"
    acquired = tmp_path / "competitor-acquired"
    repository = Path(__file__).resolve().parents[1]
    source = repository / "src"
    child_code = f"""
from pathlib import Path
from ocmask.full_pair_worker import _exclusive_pair_output_lock
Path({str(attempted)!r}).write_text('started', encoding='utf-8')
with _exclusive_pair_output_lock(Path({str(output)!r}), 'pair-1', 'run-1'):
    Path({str(acquired)!r}).write_text('acquired', encoding='utf-8')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(source), environment.get("PYTHONPATH"))
        if value
    )

    process = None
    # The parent keeps the distinct run-level lock while the child eventually
    # acquires the pair lock. This is the evaluator/worker ownership pattern.
    with _exclusive_evaluation_output_lock(evaluation):
        with worker._exclusive_pair_output_lock(output, "pair-1", "run-1"):
            metadata = json.loads(
                (output / ".pair-worker.lock").read_text(encoding="utf-8")
            )
            assert metadata["pid"] == os.getpid()
            assert metadata["pair_id"] == "pair-1"
            assert metadata["run_fingerprint"] == "run-1"

            process = subprocess.Popen(
                [sys.executable, "-c", child_code],
                cwd=repository,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not attempted.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert attempted.is_file()
            assert not acquired.exists()
            assert process.poll() is None

        assert process is not None
        stdout, _ = process.communicate(timeout=5)
        assert process.returncode == 0, stdout
        assert "Waiting for an existing worker" in stdout
        assert acquired.read_text(encoding="utf-8") == "acquired"


def test_completed_pair_freeze_requires_all_prediction_hashes(tmp_path: Path) -> None:
    output = tmp_path / "pairs" / "pair-1"
    image0 = tmp_path / "image0.png"
    image1 = tmp_path / "image1.png"
    image0.write_bytes(b"image zero")
    image1.write_bytes(b"image one")
    inputs = pair_input_identity((image0, image1))
    _write_freeze(
        output,
        pair_id="pair-1",
        run_fingerprint="run-1",
        inputs=inputs,
    )

    assert worker._completed_pair_freeze_is_valid(
        output,
        pair_id="pair-1",
        run_fingerprint="run-1",
        inputs=inputs,
    )

    (output / "labels_base.png").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen base prediction hash mismatch"):
        worker._completed_pair_freeze_is_valid(
            output,
            pair_id="pair-1",
            run_fingerprint="run-1",
            inputs=inputs,
        )


def test_worker_skips_inference_when_orphan_freezes_while_resume_waits(
    tmp_path: Path, monkeypatch
) -> None:
    pair_id = "pair-1"
    run_fingerprint = "run-1"
    output = tmp_path / "pairs" / pair_id
    image0 = tmp_path / "image0.png"
    image1 = tmp_path / "image1.png"
    image0.write_bytes(b"image zero")
    image1.write_bytes(b"image one")
    inputs = pair_input_identity((image0, image1))
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text("{}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "2026")
    monkeypatch.setattr(
        worker, "load_config", lambda _path: {"reconstruction": {"seed": 2026}}
    )
    monkeypatch.setattr(
        worker, "validate_worker_execution_identity", lambda **_kwargs: None
    )

    @contextmanager
    def orphan_finishes_before_resumed_worker_enters(
        pair_output: Path, resumed_pair_id: str, resumed_fingerprint: str
    ):
        _write_freeze(
            pair_output,
            pair_id=resumed_pair_id,
            run_fingerprint=resumed_fingerprint,
            inputs=inputs,
        )
        yield

    monkeypatch.setattr(
        worker,
        "_exclusive_pair_output_lock",
        orphan_finishes_before_resumed_worker_enters,
    )

    def inference_must_not_initialize(*_args, **_kwargs):
        raise AssertionError("a freeze completed under the pair lock was recomputed")

    monkeypatch.setattr(worker, "initialize_pair_process", inference_must_not_initialize)

    assert (
        worker.main(
            [
                "--image0",
                str(image0),
                "--image1",
                str(image1),
                "--output",
                str(output),
                "--pair-id",
                pair_id,
                "--pipeline-config",
                str(config_path),
                "--repository",
                str(tmp_path),
                "--run-fingerprint",
                run_fingerprint,
                "--run-manifest",
                str(run_manifest),
            ]
        )
        == 0
    )
