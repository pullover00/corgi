from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ocmask.reproducibility import (
    build_ground_truth_manifest,
    build_run_manifest,
    measure_checkpoint_assets,
    measure_installed_distributions,
    pair_input_identity,
    pair_output_directory,
    prepare_ground_truth_manifest,
    prepare_run_directory,
    sha256_file,
    sha256_json,
    sha256_tree,
    validate_runtime,
    validate_worker_execution_identity,
)


def _write_run_manifest(
    root: Path,
    config_path: Path,
    config: dict,
    image: Path,
    *,
    source: Path,
    checkpoint: Path,
) -> tuple[Path, dict]:
    checkpoint_stat = checkpoint.stat()
    installed_distributions = measure_installed_distributions()
    identity = {
        "schema_version": 2,
        "pipeline_config_file_sha256": sha256_file(config_path),
        "pipeline_config_expanded_sha256": sha256_json(config),
        "implementation_tree_sha256": sha256_tree(root / "src" / "ocmask"),
        "inference_inputs": {"pair-1": pair_input_identity((image,))},
        "source_assets": {
            "model": {
                "path": str(source.resolve()),
                "source_tree_sha256": sha256_tree(
                    source, suffixes=(".py", ".yaml", ".yml", ".json")
                ),
            }
        },
        "checkpoint_assets": {
            "model": {
                "resolved_path": str(checkpoint.resolve()),
                "size_bytes": checkpoint_stat.st_size,
                "mtime_ns": checkpoint_stat.st_mtime_ns,
            }
        },
        "runtime": {
            "installed_distributions": installed_distributions,
            "installed_distributions_sha256": sha256_json(
                installed_distributions
            ),
        },
    }
    manifest = {
        **identity,
        "run_fingerprint": sha256_json(identity),
        "created_at_utc": "ignored-by-identity",
    }
    path = root / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_prepare_run_directory_rejects_tampered_manifest_body(tmp_path: Path) -> None:
    identity = {"schema_version": 2, "value": "original"}
    manifest = {**identity, "run_fingerprint": sha256_json(identity)}
    output = tmp_path / "run"
    prepare_run_directory(output, manifest)

    saved = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    saved["value"] = "tampered"
    (output / "run_manifest.json").write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match its contents"):
        prepare_run_directory(output, manifest)


def test_ground_truth_identity_is_immutable_but_created_time_is_not(
    tmp_path: Path,
) -> None:
    first = build_ground_truth_manifest(
        run_fingerprint="run", targets={"pair": {"target.png": "aaa"}}
    )
    prepare_ground_truth_manifest(tmp_path, first)

    equivalent = build_ground_truth_manifest(
        run_fingerprint="run", targets={"pair": {"target.png": "aaa"}}
    )
    prepare_ground_truth_manifest(tmp_path, equivalent)

    changed = build_ground_truth_manifest(
        run_fingerprint="run", targets={"pair": {"target.png": "bbb"}}
    )
    with pytest.raises(RuntimeError, match="Ground-truth content changed"):
        prepare_ground_truth_manifest(tmp_path, changed)


@pytest.mark.parametrize("pair_id", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b"])
def test_pair_output_directory_rejects_unsafe_ids(
    tmp_path: Path, pair_id: str
) -> None:
    with pytest.raises(ValueError, match="Unsafe pair ID"):
        pair_output_directory(tmp_path, pair_id)


def test_checkpoint_measurement_covers_all_consumed_sam3_paths_once(
    tmp_path: Path, monkeypatch
) -> None:
    mast3r = tmp_path / "mast3r.pt"
    sam2 = tmp_path / "sam2.pt"
    sam3 = tmp_path / "sam3.pt"
    dino = tmp_path / "dino.pt"
    for index, path in enumerate((mast3r, sam2, sam3, dino)):
        path.write_bytes(bytes([index]))
    mast3r_sha = sha256_file(mast3r)
    sam2_sha = sha256_file(sam2)
    sam3_sha = sha256_file(sam3)
    dino_sha = sha256_file(dino)
    config = {
        "reconstruction": {
            "mast3r": {
                "checkpoint": str(mast3r),
                "checkpoint_sha256": mast3r_sha,
            },
            "sam2": {
                "checkpoint": str(sam2),
                "checkpoint_sha256": sam2_sha,
            },
        },
        "sam3_proposals": {
            "sam3_image_checkpoint": str(sam3),
            "sam3_image_checkpoint_sha256": sam3_sha,
        },
        "sam3_features": {
            "sam3": {"checkpoint": str(sam3), "checkpoint_sha256": sam3_sha}
        },
        "dinov2_features": {
            "dinov2": {
                "checkpoint": str(dino),
                "checkpoint_sha256": dino_sha,
            }
        },
        "obvious_object_sentinel": {
            "sam3": {"checkpoint": str(sam3), "checkpoint_sha256": sam3_sha}
        },
    }

    import ocmask.reproducibility as reproducibility

    original = reproducibility.sha256_file
    calls: list[Path] = []

    def counted(path):
        calls.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(reproducibility, "sha256_file", counted)
    measured = measure_checkpoint_assets(config, tmp_path)

    assert set(measured) == {
        "mast3r",
        "sam2",
        "sam3_proposals",
        "sam3_features",
        "dinov2",
        "sam3_sentinel",
    }
    assert calls.count(sam3.resolve()) == 1


def test_checkpoint_measurement_rejects_a_missing_expected_hash(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    digest = sha256_file(checkpoint)
    config = {
        "reconstruction": {
            "mast3r": {"checkpoint": str(checkpoint)},
            "sam2": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            },
        },
        "sam3_proposals": {
            "sam3_image_checkpoint": str(checkpoint),
            "sam3_image_checkpoint_sha256": digest,
        },
        "sam3_features": {
            "sam3": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            }
        },
        "dinov2_features": {
            "dinov2": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            }
        },
        "obvious_object_sentinel": {
            "sam3": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            }
        },
    }

    with pytest.raises(
        RuntimeError, match="mast3r checkpoint has no configured expected SHA-256"
    ):
        measure_checkpoint_assets(config, tmp_path)


class _FakeDistribution:
    def __init__(
        self, name: str, version: str, files: dict[str, str | None]
    ) -> None:
        self.metadata = {"Name": name, "Version": version}
        self._files = files

    def read_text(self, filename: str) -> str | None:
        return self._files.get(filename)


def test_installed_distribution_inventory_is_sorted_and_hashes_metadata(
    monkeypatch,
) -> None:
    distributions = [
        _FakeDistribution(
            "z-package",
            "2.0",
            {
                "METADATA": "z metadata",
                "RECORD": None,
                "direct_url.json": '{"url":"/src/z"}',
            },
        ),
        _FakeDistribution(
            "A_Package",
            "1.0",
            {
                "METADATA": "a metadata",
                "RECORD": "a.py,sha256=abc,1\n",
                "direct_url.json": None,
            },
        ),
    ]
    monkeypatch.setattr(
        "ocmask.reproducibility.importlib.metadata.distributions",
        lambda **_kwargs: iter(distributions),
    )

    inventory = measure_installed_distributions()

    assert [record["name"] for record in inventory] == ["A_Package", "z-package"]
    assert inventory[0] == {
        "name": "A_Package",
        "version": "1.0",
        "metadata_sha256": {
            "METADATA": hashlib.sha256(b"a metadata").hexdigest(),
            "RECORD": hashlib.sha256(b"a.py,sha256=abc,1\n").hexdigest(),
        },
    }
    assert inventory[1]["metadata_sha256"] == {
        "METADATA": hashlib.sha256(b"z metadata").hexdigest(),
        "direct_url.json": hashlib.sha256(b'{"url":"/src/z"}').hexdigest(),
    }


def _expected_runtime() -> dict:
    return {
        "python_major_minor": "3.11",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "numpy": "1.26.4",
        "Pillow": "10.4.0",
        "scipy": "1.14.1",
        "scikit-image": "0.24.0",
        "cuda": "12.4",
        "cudnn": 90100,
        "cuda_available": True,
    }


def _measured_runtime() -> dict:
    expected = _expected_runtime()
    return {
        "python": "3.11.9",
        "packages": {
            name: expected[name]
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "Pillow",
                "scipy",
                "scikit-image",
            )
        },
        "torch_cuda": "12.4",
        "cudnn": 90100,
        "cuda_available": True,
    }


def test_runtime_validation_requires_complete_pinned_gpu_runtime() -> None:
    validate_runtime(_measured_runtime(), _expected_runtime())

    missing = _expected_runtime()
    del missing["Pillow"]
    with pytest.raises(RuntimeError, match="missing: Pillow"):
        validate_runtime(_measured_runtime(), missing)

    cpu_runtime = _measured_runtime()
    cpu_runtime["cuda_available"] = False
    with pytest.raises(RuntimeError, match="CUDA availability expected true"):
        validate_runtime(cpu_runtime, _expected_runtime())

    wrong_cudnn = _measured_runtime()
    wrong_cudnn["cudnn"] = 90000
    with pytest.raises(RuntimeError, match="cuDNN expected 90100"):
        validate_runtime(wrong_cudnn, _expected_runtime())

    wrong_scipy = _measured_runtime()
    wrong_scipy["packages"]["scipy"] = "1.13.0"
    with pytest.raises(RuntimeError, match="scipy expected 1.14.1"):
        validate_runtime(wrong_scipy, _expected_runtime())


def test_build_run_manifest_rejects_incomplete_input_identity_before_models(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"missing=\['pair-2'\]"):
        build_run_manifest(
            config={"reconstruction": {"seed": 2026}},
            config_path=tmp_path / "config.yaml",
            benchmark_manifest_path=tmp_path / "manifest.jsonl",
            repository=tmp_path,
            prediction_variant="guarded",
            cache_dir=None,
            fraction=1.0,
            selection_ids=["pair-1", "pair-2"],
            inference_inputs={"pair-1": {"a.png": "hash"}},
        )


def test_worker_identity_detects_live_source_and_input_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    implementation = repository / "src" / "ocmask"
    implementation.mkdir(parents=True)
    (implementation / "pipeline.py").write_text("REVISION = 1\n", encoding="utf-8")
    source = tmp_path / "model-source"
    source.mkdir()
    source_file = source / "model.py"
    source_file.write_text("REVISION = 1\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels-v1")
    config = {"schema_version": 1, "setting": 1}
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("schema_version: 1\nsetting: 1\n", encoding="utf-8")
    manifest_path, manifest = _write_run_manifest(
        repository,
        config_path,
        config,
        image,
        source=source,
        checkpoint=checkpoint,
    )

    validate_worker_execution_identity(
        config=config,
        config_path=config_path,
        repository=repository,
        run_manifest_path=manifest_path,
        run_fingerprint=manifest["run_fingerprint"],
        pair_id="pair-1",
        inputs=pair_input_identity((image,)),
    )

    source_file.write_text("REVISION = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="model source changed"):
        validate_worker_execution_identity(
            config=config,
            config_path=config_path,
            repository=repository,
            run_manifest_path=manifest_path,
            run_fingerprint=manifest["run_fingerprint"],
            pair_id="pair-1",
            inputs=pair_input_identity((image,)),
        )

    source_file.write_text("REVISION = 1\n", encoding="utf-8")
    image.write_bytes(b"pixels-v2")
    with pytest.raises(RuntimeError, match="Inference inputs"):
        validate_worker_execution_identity(
            config=config,
            config_path=config_path,
            repository=repository,
            run_manifest_path=manifest_path,
            run_fingerprint=manifest["run_fingerprint"],
            pair_id="pair-1",
            inputs=pair_input_identity((image,)),
        )


def test_worker_identity_detects_installed_distribution_change(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    implementation = repository / "src" / "ocmask"
    implementation.mkdir(parents=True)
    (implementation / "pipeline.py").write_text("REVISION = 1\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("REVISION = 1\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    config = {"schema_version": 1}
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    manifest_path, manifest = _write_run_manifest(
        repository,
        config_path,
        config,
        image,
        source=source,
        checkpoint=checkpoint,
    )
    changed_inventory = [
        *manifest["runtime"]["installed_distributions"],
        {
            "name": "new-package",
            "version": "1.0",
            "metadata_sha256": {"METADATA": "changed"},
        },
    ]
    monkeypatch.setattr(
        "ocmask.reproducibility.measure_installed_distributions",
        lambda: changed_inventory,
    )

    with pytest.raises(RuntimeError, match="Installed distribution inventory changed"):
        validate_worker_execution_identity(
            config=config,
            config_path=config_path,
            repository=repository,
            run_manifest_path=manifest_path,
            run_fingerprint=manifest["run_fingerprint"],
            pair_id="pair-1",
            inputs=pair_input_identity((image,)),
        )


def test_worker_identity_detects_checkpoint_replacement_without_rehashing(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    implementation = repository / "src" / "ocmask"
    implementation.mkdir(parents=True)
    (implementation / "pipeline.py").write_text("REVISION = 1\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("REVISION = 1\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights-a")
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    config = {"schema_version": 1}
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    manifest_path, manifest = _write_run_manifest(
        repository,
        config_path,
        config,
        image,
        source=source,
        checkpoint=checkpoint,
    )

    checkpoint.write_bytes(b"weights-b")
    # Guarantee an mtime change even on a coarse/virtual test filesystem.
    stat = checkpoint.stat()
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    with pytest.raises(RuntimeError, match="checkpoint changed"):
        validate_worker_execution_identity(
            config=config,
            config_path=config_path,
            repository=repository,
            run_manifest_path=manifest_path,
            run_fingerprint=manifest["run_fingerprint"],
            pair_id="pair-1",
            inputs=pair_input_identity((image,)),
        )
