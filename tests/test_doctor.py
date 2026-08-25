"""CPU-only coverage for the production and legacy doctor entry points."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocmask.cli as cli


def _production_config() -> dict:
    return {
        "reproducibility": {"expected_runtime": {"runtime": "expected"}},
        # This historical field is deliberately unused by run_pair. Production
        # doctor must neither require it nor expose it in its report.
        "sam3_proposals": {"sam31_checkpoint": "${SAM31_CHECKPOINT}"},
    }


def _runtime() -> dict:
    return {
        "python": "3.11.15",
        "implementation": "CPython",
        "executable": "/env/bin/python",
        "platform": "test-platform",
        "packages": {"torch": "2.5.1"},
        "torch_cuda": "12.4",
        "cudnn": 90100,
        "cuda_available": True,
        "gpu": "test-gpu",
        "gpu_compute_capability": [8, 6],
        "gpu_vram_bytes": 16 * 1024**3,
        "installed_distributions": [{"name": "one"}, {"name": "two"}],
        "installed_distributions_sha256": "distribution-fingerprint",
    }


def test_production_doctor_runs_all_identity_checks_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _production_config()
    runtime = _runtime()
    calls: list[tuple] = []
    checkpoints = {
        name: {"resolved_path": f"/{name}.pt", "sha256": f"{name}-sha"}
        for name in (
            "mast3r",
            "sam2",
            "sam3_proposals",
            "sam3_features",
            "dinov2",
            "sam3_sentinel",
        )
    }
    sources = {
        name: {
            "path": f"/{name}",
            "actual_git_commit": f"{name}-commit",
            "source_tree_sha256": f"{name}-tree",
        }
        for name in (
            "mast3r",
            "sam2",
            "sam3_proposals",
            "sam3_features",
            "dinov2",
            "sam3_sentinel",
        )
    }

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "measure_runtime", lambda: calls.append(("runtime",)) or runtime)

    def validate(actual, expected) -> None:
        calls.append(("validate_runtime", actual, expected))

    def measure_checkpoints(actual_config, repository):
        calls.append(("checkpoints", actual_config, Path(repository)))
        return checkpoints

    def measure_sources(actual_config, repository):
        calls.append(("sources", actual_config, Path(repository)))
        return sources

    monkeypatch.setattr(cli, "validate_runtime", validate)
    monkeypatch.setattr(cli, "measure_checkpoint_assets", measure_checkpoints)
    monkeypatch.setattr(cli, "measure_source_assets", measure_sources)
    monkeypatch.setattr(
        cli,
        "make_pipeline",
        lambda *_args, **_kwargs: pytest.fail("doctor must not construct models"),
    )

    status = cli.production_doctor_command(
        tmp_path / "pipeline.yaml", repository=tmp_path
    )
    output = capsys.readouterr().out
    report = json.loads(output)

    assert status == 0
    assert report["ready"] is True
    assert report["model_inference_performed"] is False
    assert report["failed_checks"] == []
    assert all(check["ok"] for check in report["checks"].values())
    assert set(report["checks"]["checkpoints"]["assets"]) == set(checkpoints)
    assert set(report["checks"]["sources"]["assets"]) == set(sources)
    assert report["checks"]["runtime"]["actual"][
        "installed_distributions_count"
    ] == 2
    assert "installed_distributions" not in report["checks"]["runtime"]["actual"]
    assert "sam31" not in output.casefold()
    assert [call[0] for call in calls] == [
        "runtime",
        "validate_runtime",
        "checkpoints",
        "sources",
    ]
    assert calls[1][1:] == (runtime, config["reproducibility"]["expected_runtime"])
    assert calls[2][1:] == (config, tmp_path.resolve())
    assert calls[3][1:] == (config, tmp_path.resolve())


def test_production_doctor_reports_every_failed_check_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: _production_config())
    monkeypatch.setattr(cli, "measure_runtime", _runtime)
    monkeypatch.setattr(
        cli,
        "validate_runtime",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("CUDA expected 12.4, got 12.1")),
    )
    monkeypatch.setattr(
        cli,
        "measure_checkpoint_assets",
        lambda *_args: (_ for _ in ()).throw(
            FileNotFoundError("Missing sam3_features checkpoint: /models/sam3.pt")
        ),
    )
    monkeypatch.setattr(
        cli,
        "measure_source_assets",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("sam3 source commit mismatch: expected abc, got def")
        ),
    )

    status = cli.production_doctor_command(
        tmp_path / "pipeline.yaml", repository=tmp_path
    )
    report = json.loads(capsys.readouterr().out)

    assert status == 1
    assert report["ready"] is False
    assert set(report["failed_checks"]) == {"runtime", "checkpoints", "sources"}
    assert "CUDA expected" in report["checks"]["runtime"]["error"]["message"]
    assert "Missing sam3_features checkpoint" in report["checks"]["checkpoints"]["error"]["message"]
    assert "source commit mismatch" in report["checks"]["sources"]["error"]["message"]
    assert report["checks"]["runtime"]["actual"]["cuda_available"] is True


def test_production_doctor_reports_configuration_failure_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
    )
    monkeypatch.setattr(
        cli,
        "measure_runtime",
        lambda: pytest.fail("invalid config must stop dependent checks"),
    )

    status = cli.production_doctor_command(missing, repository=tmp_path)
    report = json.loads(capsys.readouterr().out)

    assert status == 1
    assert report["ready"] is False
    assert report["failed_checks"] == ["configuration"]
    assert report["checks"]["configuration"]["error"]["type"] == "FileNotFoundError"
    assert str(missing) in report["checks"]["configuration"]["error"]["message"]


def test_plain_doctor_uses_the_production_pipeline_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        cli,
        "production_doctor_command",
        lambda path: seen.append(path) or 7,
    )
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path: pytest.fail("plain doctor must not load the legacy config"),
    )

    assert cli.main(["doctor"]) == 7
    assert seen == ["configs/pipeline.yaml"]


def test_stage1_only_doctor_retains_the_legacy_global_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"legacy": True}
    loaded: list[str] = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: loaded.append(path) or config,
    )
    monkeypatch.setattr(cli, "doctor_command", lambda value: 3 if value is config else 4)
    monkeypatch.setattr(
        cli,
        "production_doctor_command",
        lambda _path: pytest.fail("legacy doctor must not run production checks"),
    )

    assert cli.main(["--config", "legacy-stage1.yaml", "doctor", "--stage1-only"]) == 3
    assert loaded == ["legacy-stage1.yaml"]


def test_full_pipeline_help_requires_sam3_but_not_unused_sam31(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["evaluate", "changesim", "--help"])

    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "external SAM3 source/checkpoint" in help_text
    assert "SAM3.1" not in help_text
