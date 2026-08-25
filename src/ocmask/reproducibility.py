"""Execution identities, asset validation, and immutable resume markers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import site
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import save_json
from .numerics import numerical_policy


RUN_MANIFEST_SCHEMA_VERSION = 2
GROUND_TRUTH_MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with stable separators/key ordering."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_tree(
    root: str | Path,
    *,
    suffixes: tuple[str, ...] = (".py",),
    names: tuple[str, ...] = (),
) -> str:
    """Hash selected files in a source tree, including relative filenames."""

    root = Path(root).resolve()
    digest = hashlib.sha256()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and (path.suffix in suffixes or path.name in names)
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _resolve_local_path(value: str | Path, repository: Path) -> Path:
    text = str(value)
    if "$" in text:
        raise ValueError(f"Unexpanded environment variable in path: {text}")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _checkpoint_specs(config: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Return only checkpoints consumed by ``ocmask.inference.run_pair``."""

    return [
        (
            "mast3r",
            config["reconstruction"]["mast3r"]["checkpoint"],
            config["reconstruction"]["mast3r"].get("checkpoint_sha256"),
        ),
        (
            "sam2",
            config["reconstruction"]["sam2"]["checkpoint"],
            config["reconstruction"]["sam2"].get("checkpoint_sha256"),
        ),
        (
            "sam3_proposals",
            config["sam3_proposals"]["sam3_image_checkpoint"],
            config["sam3_proposals"].get("sam3_image_checkpoint_sha256"),
        ),
        (
            "sam3_features",
            config["sam3_features"]["sam3"]["checkpoint"],
            config["sam3_features"]["sam3"].get("checkpoint_sha256"),
        ),
        (
            "dinov2",
            config["dinov2_features"]["dinov2"]["checkpoint"],
            config["dinov2_features"]["dinov2"].get("checkpoint_sha256"),
        ),
        (
            "sam3_sentinel",
            config["obvious_object_sentinel"]["sam3"]["checkpoint"],
            config["obvious_object_sentinel"]["sam3"].get("checkpoint_sha256"),
        ),
    ]


def measure_checkpoint_assets(
    config: dict[str, Any], repository: str | Path
) -> dict[str, dict[str, Any]]:
    """Hash every used checkpoint and reject configured hash mismatches."""

    repository = Path(repository).resolve()
    measured: dict[str, dict[str, Any]] = {}
    resolved_hashes: dict[Path, str] = {}
    resolved_stats: dict[Path, tuple[int, int]] = {}
    for name, configured_path, expected in _checkpoint_specs(config):
        if not expected:
            raise RuntimeError(
                f"{name} checkpoint has no configured expected SHA-256"
            )
        path = _resolve_local_path(configured_path, repository)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
        resolved = path.resolve()
        # ``dict.setdefault`` would eagerly evaluate ``sha256_file`` and hash
        # the same multi-gigabyte checkpoint repeatedly when several stages
        # intentionally share it.
        if resolved not in resolved_hashes:
            before = resolved.stat()
            resolved_hashes[resolved] = sha256_file(resolved)
            after = resolved.stat()
            before_identity = (before.st_size, before.st_mtime_ns)
            after_identity = (after.st_size, after.st_mtime_ns)
            if before_identity != after_identity:
                raise RuntimeError(
                    f"{name} checkpoint changed while it was being fingerprinted: "
                    f"{resolved}"
                )
            resolved_stats[resolved] = after_identity
        actual = resolved_hashes[resolved]
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"{name} checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
            )
        size_bytes, mtime_ns = resolved_stats[resolved]
        measured[name] = {
            "configured_path": str(configured_path),
            "resolved_path": str(resolved),
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "sha256": actual,
            "expected_sha256": expected,
        }
    return measured


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _module_root(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def measure_source_assets(
    config: dict[str, Any], repository: str | Path
) -> dict[str, dict[str, Any]]:
    """Record the exact Python sources used by local/external model wrappers."""

    repository = Path(repository).resolve()
    specifications = [
        (
            "sam3_proposals",
            _resolve_local_path(config["sam3_proposals"]["sam3_source"], repository),
            config["sam3_proposals"].get("sam3_source_commit"),
        ),
        (
            "sam3_features",
            _resolve_local_path(
                config["sam3_features"]["sam3"]["source"], repository
            ),
            config["sam3_features"]["sam3"].get("source_commit"),
        ),
        (
            "sam3_sentinel",
            _resolve_local_path(
                config["obvious_object_sentinel"]["sam3"]["source"], repository
            ),
            config["obvious_object_sentinel"]["sam3"].get("source_commit"),
        ),
        (
            "dinov2",
            _resolve_local_path(
                config["dinov2_features"]["dinov2"]["source"], repository
            ),
            config["dinov2_features"]["dinov2"].get("source_commit"),
        ),
        (
            "mast3r",
            _resolve_local_path(
                config["reconstruction"]["mast3r"].get("source", "src/mast3r"),
                repository,
            ),
            config["reconstruction"]["mast3r"].get("source_commit"),
        ),
    ]
    sam2_root = _module_root("sam2")
    measured: dict[str, dict[str, Any]] = {}
    tree_hashes: dict[Path, str] = {}
    commits: dict[Path, str | None] = {}
    for name, path, expected_commit in specifications:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {name} source tree: {path}")
        if path not in commits:
            commits[path] = _git_commit(path)
        actual_commit = commits[path]
        if expected_commit is not None and actual_commit != expected_commit:
            raise RuntimeError(
                f"{name} source commit mismatch: expected {expected_commit}, "
                f"got {actual_commit} at {path}"
            )
        if path not in tree_hashes:
            tree_hashes[path] = sha256_tree(
                path, suffixes=(".py", ".yaml", ".yml", ".json")
            )
        measured[name] = {
            "path": str(path),
            "source_tree_sha256": tree_hashes[path],
            "actual_git_commit": actual_commit,
            "expected_git_commit": expected_commit,
        }
    if sam2_root is None:
        raise RuntimeError("SAM2 is not importable in the active environment")
    expected_sam2_commit = config["reconstruction"]["sam2"].get("source_commit")
    try:
        sam2_distribution = importlib.metadata.distribution("sam-2")
        sam2_direct_url_text = sam2_distribution.read_text("direct_url.json")
        sam2_direct_url = (
            json.loads(sam2_direct_url_text) if sam2_direct_url_text else {}
        )
        actual_sam2_commit = sam2_direct_url.get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        actual_sam2_commit = None
    if expected_sam2_commit is not None and actual_sam2_commit != expected_sam2_commit:
        raise RuntimeError(
            "sam2 source commit mismatch: expected "
            f"{expected_sam2_commit}, got {actual_sam2_commit} at {sam2_root}"
        )
    measured["sam2"] = {
        "path": str(sam2_root),
        # Hydra resolves ``reconstruction.sam2.model_cfg`` from YAML files in
        # this package, so hashing Python alone does not identify the model.
        "source_tree_sha256": sha256_tree(
            sam2_root, suffixes=(".py", ".yaml", ".yml", ".json")
        ),
        "actual_git_commit": actual_sam2_commit,
        "expected_git_commit": expected_sam2_commit,
    }
    return measured


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


_DISTRIBUTION_METADATA_FILES = ("METADATA", "RECORD", "direct_url.json")


def measure_installed_distributions() -> list[dict[str, Any]]:
    """Fingerprint every installed distribution without importing its package.

    Version strings alone do not identify editable installs, locally rebuilt
    wheels, or a package whose installation metadata changed during a long
    evaluation.  ``importlib.metadata`` reads the installer-owned
    ``*.dist-info``/``*.egg-info`` records directly, so this inventory does
    not execute any model package's import-time code.

    A list is used instead of a name-keyed mapping because Python environments
    can contain more than one distribution with the same normalized name.
    Sorting the complete records makes the result independent of the order in
    which import finders enumerate site-packages.
    """

    # Restrict discovery to installer-owned site-package roots. Model adapters
    # intentionally add MASt3R/DINO/SAM3 source directories to ``sys.path`` at
    # runtime; an egg-info directory in one of those trees must not make the
    # supposedly installed-environment inventory change between the worker's
    # pre- and post-inference checks.
    search_paths = {
        str(Path(path).resolve())
        for path in (*site.getsitepackages(), site.getusersitepackages())
        if path and Path(path).is_dir()
    }
    inventory: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions(path=sorted(search_paths)):
        metadata = distribution.metadata
        record: dict[str, Any] = {
            "name": str(metadata.get("Name") or ""),
            "version": str(metadata.get("Version") or ""),
            "metadata_sha256": {},
        }
        for filename in _DISTRIBUTION_METADATA_FILES:
            content = distribution.read_text(filename)
            if content is not None:
                record["metadata_sha256"][filename] = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()
        inventory.append(record)

    inventory.sort(
        key=lambda record: (
            record["name"].casefold(),
            record["name"],
            record["version"],
            json.dumps(
                record["metadata_sha256"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    return inventory


def measure_runtime() -> dict[str, Any]:
    """Record library/CUDA versions that can change model numerics."""

    import torch

    installed_distributions = measure_installed_distributions()
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "Pillow",
                "scipy",
                "scikit-image",
            )
        },
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "installed_distributions": installed_distributions,
        "installed_distributions_sha256": sha256_json(installed_distributions),
    }
    if torch.cuda.is_available():
        runtime["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        runtime["gpu_compute_capability"] = [props.major, props.minor]
        runtime["gpu_vram_bytes"] = props.total_memory
    return runtime


def validate_runtime(runtime: dict[str, Any], expected: dict[str, Any]) -> None:
    """Fail before inference when the active runtime is not the frozen one."""

    mismatches: list[str] = []
    required = {
        "python_major_minor",
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "scipy",
        "scikit-image",
        "cuda",
        "cudnn",
        "cuda_available",
    }
    missing = sorted(required - set(expected))
    if missing:
        raise RuntimeError(
            "Validated runtime configuration is incomplete; missing: "
            + ", ".join(missing)
        )
    if runtime["python"].split(".")[:2] != str(expected["python_major_minor"]).split("."):
        mismatches.append(
            f"python expected {expected['python_major_minor']}.x, got {runtime['python']}"
        )
    for name in ("torch", "torchvision", "numpy", "Pillow", "scipy", "scikit-image"):
        actual = runtime["packages"].get(name)
        if actual is None or actual.split("+")[0] != str(expected[name]):
            mismatches.append(f"{name} expected {expected[name]}, got {actual}")
    actual_cuda = runtime.get("torch_cuda")
    if str(actual_cuda) != str(expected["cuda"]):
        mismatches.append(f"CUDA expected {expected['cuda']}, got {actual_cuda}")
    actual_cudnn = runtime.get("cudnn")
    if str(actual_cudnn) != str(expected["cudnn"]):
        mismatches.append(f"cuDNN expected {expected['cudnn']}, got {actual_cudnn}")
    if expected["cuda_available"] is not True:
        mismatches.append("expected_runtime.cuda_available must be true")
    elif runtime.get("cuda_available") is not True:
        mismatches.append(
            f"CUDA availability expected true, got {runtime.get('cuda_available')}"
        )
    if mismatches:
        raise RuntimeError(
            "Active model runtime differs from the validated headline runtime: "
            + "; ".join(mismatches)
        )


def _metadata_tree_sha256(root: Path) -> str:
    """Fingerprint a cache's manifests/progress without hashing large tensors."""

    return sha256_tree(
        root,
        suffixes=(".json", ".jsonl", ".yaml", ".yml"),
        names=("job.json",),
    )


_CACHE_STAGE_FILES = {
    "stage1": (
        "reconstruction.npz",
        "geometry.npz",
        "render_0_to_1.png",
        "render_clean_to_1.png",
        "config.json",
        "inputs.json",
    ),
    "stage2": ("proposal_cache/source.npz", "proposal_cache/target.npz"),
    "stage3": (
        "labels.png",
        "diagnostics.json",
        "tracking_attempts.json",
        "target.png",
    ),
    "stage4": ("sam3_features.npz", "decisions.json"),
}


def _ordered_input_paths(identity: Mapping[str, str]) -> tuple[str, str]:
    """Recover image0/image1 from a freshly measured pair identity.

    ``pair_input_identity`` deliberately preserves the caller's input order.
    This helper is used only on that in-memory mapping (never on the
    sort-key-serialized run manifest), so the two cache-bound paths retain
    their semantic image0/image1 roles.
    """

    paths = tuple(identity)
    if len(paths) != 2:
        raise ValueError(
            "A full-pipeline pair must identify exactly image0 and image1"
        )
    return paths[0], paths[1]


def measure_cache_assets(
    cache_root: str | Path,
    config: Mapping[str, Any],
    selection_ids: Iterable[str],
    inference_inputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Fingerprint exactly the input-bound cache files the selection can consume."""

    from .weekend_cache import ChangesimWeekendCache

    root = Path(cache_root).resolve()
    cache = ChangesimWeekendCache(root, config)
    hashes: dict[Path, str] = {}
    pairs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for pair_id in selection_ids:
        if pair_id not in inference_inputs:
            raise ValueError(f"Missing inference inputs for cache lookup: {pair_id}")
        image0_path, image1_path = _ordered_input_paths(inference_inputs[pair_id])
        lookup = cache.lookup(pair_id, image0_path, image1_path)
        stages: dict[str, list[dict[str, Any]]] = {}
        for stage, relative_paths in _CACHE_STAGE_FILES.items():
            stage_dir = getattr(lookup, f"{stage}_dir")
            if stage_dir is None:
                continue
            records = []
            for relative_path in relative_paths:
                path = (stage_dir / relative_path).resolve()
                if not path.is_file():
                    raise RuntimeError(
                        f"Validated cache lookup lost required file: {path}"
                    )
                if path not in hashes:
                    hashes[path] = sha256_file(path)
                stat = path.stat()
                records.append(
                    {
                        "path": str(path),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": hashes[path],
                    }
                )
            stages[stage] = records
        if stages:
            pairs[pair_id] = stages
    return {
        "path": str(root),
        "metadata_tree_sha256": _metadata_tree_sha256(root),
        "consumed_pair_assets": pairs,
    }


def build_run_manifest(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    benchmark_manifest_path: str | Path,
    repository: str | Path,
    prediction_variant: str,
    cache_dir: str | Path | None,
    fraction: float,
    selection_ids: Iterable[str],
    inference_inputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Measure every execution input and return a self-identifying manifest."""

    repository = Path(repository).resolve()
    config_path = Path(config_path).resolve()
    benchmark_manifest_path = Path(benchmark_manifest_path).resolve()
    seed = int(config["reconstruction"]["seed"])
    selection_ids = list(selection_ids)
    if len(selection_ids) != len(set(selection_ids)):
        raise ValueError("Run manifest selection contains duplicate pair IDs")
    if set(inference_inputs) != set(selection_ids):
        missing = sorted(set(selection_ids) - set(inference_inputs))
        extra = sorted(set(inference_inputs) - set(selection_ids))
        raise ValueError(
            "Inference input identities do not match the evaluation selection: "
            f"missing={missing}, extra={extra}"
        )
    cache_identity = None
    if cache_dir is not None:
        cache_path = Path(cache_dir).resolve()
        if not cache_path.is_dir():
            raise FileNotFoundError(f"Cache directory does not exist: {cache_path}")
        cache_identity = measure_cache_assets(
            cache_path, config, selection_ids, inference_inputs
        )
    runtime = measure_runtime()
    validate_runtime(runtime, config["reproducibility"]["expected_runtime"])
    identity = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "method": "object_consistent_masks_full_pipeline",
        "prediction_variant": prediction_variant,
        "fraction": float(fraction),
        "selection_ids": selection_ids,
        # Manifest text alone does not identify a dataset: image bytes can be
        # replaced in place without changing a JSONL path. These hashes cover
        # only image0/image1, never ground truth, and are therefore safe to
        # measure before the prediction-freeze phase.
        "inference_inputs": dict(inference_inputs),
        "pipeline_config_path": str(config_path),
        "pipeline_config_file_sha256": sha256_file(config_path),
        "pipeline_config_expanded_sha256": sha256_json(config),
        "benchmark_manifest_path": str(benchmark_manifest_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "implementation_tree_sha256": sha256_tree(repository / "src" / "ocmask"),
        "checkpoint_assets": measure_checkpoint_assets(config, repository),
        "source_assets": measure_source_assets(config, repository),
        "runtime": runtime,
        "numerical_policy": numerical_policy(seed),
        "cache": cache_identity,
        "ground_truth_used_in_inference": False,
    }
    return {
        **identity,
        "run_fingerprint": sha256_json(identity),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare_run_directory(output: str | Path, manifest: dict[str, Any]) -> None:
    """Create or validate an immutable run identity without deleting data."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "run_manifest.json"
    _validate_self_identifying_manifest(manifest, "run_fingerprint")
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        _validate_self_identifying_manifest(previous, "run_fingerprint")
        if previous.get("run_fingerprint") != manifest.get("run_fingerprint"):
            raise RuntimeError(
                f"{output} belongs to a different execution fingerprint. "
                "Choose a new --output directory; existing predictions were preserved."
            )
        return
    conflicting = [
        path
        for path in (output / "progress.jsonl", output / "report.json", output / "pairs")
        if path.exists()
    ]
    if conflicting:
        raise RuntimeError(
            f"{output} contains legacy/unfingerprinted evaluation artifacts. "
            "Choose a new --output directory; existing predictions were preserved."
        )
    save_json(marker, manifest)


def _validate_self_identifying_manifest(
    manifest: Mapping[str, Any], fingerprint_key: str
) -> None:
    """Reject a marker whose recorded fingerprint does not match its body."""

    recorded = manifest.get(fingerprint_key)
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in (fingerprint_key, "created_at_utc")
    }
    actual = sha256_json(identity)
    if recorded != actual:
        raise RuntimeError(
            f"Corrupt immutable manifest: {fingerprint_key} does not match its contents"
        )


def validate_worker_execution_identity(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    repository: str | Path,
    run_manifest_path: str | Path,
    run_fingerprint: str,
    pair_id: str,
    inputs: Mapping[str, str],
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify live worker inputs against the parent run's frozen identity.

    A ChangeSim evaluation can run for days. Checking this at both ends of
    every pair prevents an edit to the checkout/config or a replaced input
    image from silently mixing revisions under one run fingerprint.
    Checkpoint content was fully hashed by the parent; workers use the frozen
    size/mtime as a cheap per-pair mutation guard rather than re-hashing
    multi-gigabyte weights for every image.
    """

    repository = Path(repository).resolve()
    config_path = Path(config_path).resolve()
    manifest_path = Path(run_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_self_identifying_manifest(manifest, "run_fingerprint")
    if manifest.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError("Worker run fingerprint does not match run_manifest.json")
    recorded_runtime = manifest.get("runtime")
    if not isinstance(recorded_runtime, Mapping):
        raise RuntimeError(
            "Run manifest has no installed-distribution runtime identity"
        )
    recorded_distributions = recorded_runtime.get("installed_distributions")
    recorded_inventory_hash = recorded_runtime.get(
        "installed_distributions_sha256"
    )
    if not isinstance(recorded_distributions, list) or not isinstance(
        recorded_inventory_hash, str
    ):
        raise RuntimeError(
            "Run manifest has no installed-distribution runtime identity"
        )
    if sha256_json(recorded_distributions) != recorded_inventory_hash:
        raise RuntimeError(
            "Run manifest installed-distribution inventory hash is invalid"
        )
    current_distributions = measure_installed_distributions()
    if (
        sha256_json(current_distributions) != recorded_inventory_hash
        or current_distributions != recorded_distributions
    ):
        raise RuntimeError(
            "Installed distribution inventory changed after the run was "
            "fingerprinted"
        )
    if sha256_file(config_path) != manifest.get("pipeline_config_file_sha256"):
        raise RuntimeError("Pipeline config file changed after the run was fingerprinted")
    if sha256_json(config) != manifest.get("pipeline_config_expanded_sha256"):
        raise RuntimeError(
            "Expanded pipeline config changed after the run was fingerprinted"
        )
    if sha256_tree(repository / "src" / "ocmask") != manifest.get(
        "implementation_tree_sha256"
    ):
        raise RuntimeError("ocmask source changed after the run was fingerprinted")
    expected_inputs = manifest.get("inference_inputs", {}).get(pair_id)
    if expected_inputs != dict(inputs):
        raise RuntimeError(
            f"Inference inputs for {pair_id} differ from the run fingerprint"
        )

    cache_identity = manifest.get("cache")
    expected_cache_path = (
        Path(cache_identity["path"]).resolve() if cache_identity is not None else None
    )
    actual_cache_path = Path(cache_dir).resolve() if cache_dir is not None else None
    if actual_cache_path != expected_cache_path:
        raise RuntimeError(
            "Worker cache directory does not match the run fingerprint: "
            f"expected={expected_cache_path}, actual={actual_cache_path}"
        )
    if cache_identity is not None:
        from .weekend_cache import ChangesimWeekendCache

        pair_assets = cache_identity.get("consumed_pair_assets", {}).get(pair_id, {})
        image0_path, image1_path = _ordered_input_paths(inputs)
        current_lookup = ChangesimWeekendCache(expected_cache_path, config).lookup(
            pair_id, image0_path, image1_path
        )
        for stage, relative_paths in _CACHE_STAGE_FILES.items():
            assets = pair_assets.get(stage, [])
            current_stage_dir = getattr(current_lookup, f"{stage}_dir")
            if bool(assets) != (current_stage_dir is not None):
                raise RuntimeError(
                    f"Cached {stage} availability changed after fingerprinting for "
                    f"{pair_id}"
                )
            if current_stage_dir is not None:
                current_paths = {
                    str((current_stage_dir / relative).resolve())
                    for relative in relative_paths
                }
                recorded_paths = {str(Path(asset["path"]).resolve()) for asset in assets}
                if current_paths != recorded_paths:
                    raise RuntimeError(
                        f"Cached {stage} paths changed after fingerprinting for {pair_id}"
                    )
            for asset in assets:
                asset_path = Path(asset["path"]).resolve()
                try:
                    stat = asset_path.stat()
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"Cached {stage} asset disappeared: {asset_path}"
                    ) from exc
                if (
                    stat.st_size != asset.get("size_bytes")
                    or stat.st_mtime_ns != asset.get("mtime_ns")
                ):
                    raise RuntimeError(
                        f"Cached {stage} asset changed after fingerprinting: "
                        f"{asset_path}"
                    )

    source_hash_cache: dict[Path, str] = {}
    for name, source in manifest.get("source_assets", {}).items():
        source_path = Path(source["path"]).resolve()
        if source_path not in source_hash_cache:
            source_hash_cache[source_path] = sha256_tree(
                source_path, suffixes=(".py", ".yaml", ".yml", ".json")
            )
        if source_hash_cache[source_path] != source.get("source_tree_sha256"):
            raise RuntimeError(
                f"{name} source changed after the run was fingerprinted: {source_path}"
            )

    for name, checkpoint in manifest.get("checkpoint_assets", {}).items():
        checkpoint_path = Path(checkpoint["resolved_path"]).resolve()
        try:
            stat = checkpoint_path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{name} checkpoint disappeared after the run was fingerprinted"
            ) from exc
        if (
            stat.st_size != checkpoint.get("size_bytes")
            or stat.st_mtime_ns != checkpoint.get("mtime_ns")
        ):
            raise RuntimeError(
                f"{name} checkpoint changed after the run was fingerprinted: "
                f"{checkpoint_path}"
            )
    return manifest


def build_ground_truth_manifest(
    *,
    run_fingerprint: str,
    targets: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build the scoring identity after prediction inference has completed."""

    identity = {
        "schema_version": GROUND_TRUTH_MANIFEST_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "targets": dict(targets),
        "measured_after_prediction_freeze": True,
    }
    return {
        **identity,
        "ground_truth_fingerprint": sha256_json(identity),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare_ground_truth_manifest(
    output: str | Path, manifest: Mapping[str, Any]
) -> None:
    """Persist or validate the exact targets used for scoring a frozen run."""

    output = Path(output)
    marker = output / "ground_truth_manifest.json"
    _validate_self_identifying_manifest(manifest, "ground_truth_fingerprint")
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        _validate_self_identifying_manifest(previous, "ground_truth_fingerprint")
        if previous.get("ground_truth_fingerprint") != manifest.get(
            "ground_truth_fingerprint"
        ):
            raise RuntimeError(
                "Ground-truth content changed after predictions were frozen. "
                "Existing predictions and scores were preserved; choose a new "
                "--output directory to score the changed benchmark."
            )
        return
    save_json(marker, manifest)


def pair_input_identity(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash the exact input images consumed by one inference/scoring record."""

    return {str(Path(path).resolve()): sha256_file(path) for path in paths}


def pair_output_directory(root: str | Path, pair_id: str) -> Path:
    """Return a pair directory while rejecting IDs that can escape ``root``."""

    if (
        not pair_id
        or pair_id in {".", ".."}
        or "/" in pair_id
        or "\\" in pair_id
        or any(ord(character) < 32 for character in pair_id)
    ):
        raise ValueError(
            f"Unsafe pair ID {pair_id!r}; IDs must be non-empty single path components"
        )
    root = Path(root).resolve()
    output = (root / pair_id).resolve()
    if root not in output.parents:
        raise ValueError(f"Pair ID escapes output directory: {pair_id!r}")
    return output
