#!/usr/bin/env python3
"""Run the isolated real-image obvious-object sentinel experiment.

The experiment is deliberately additive over a frozen conservative-A3
prediction.  SAM3 proposals and appearance descriptors are computed from the
two *real* input images; synthetic point-cloud renders are used only to locate
objects geometrically.  A sentinel prediction may change only parent pixels
labelled UNCHANGED.

Execution is split into auditable phases:

1. validate every immutable parent and generate/cache real-I0 SAM3 proposals
   plus the shared full-image feature map (``--cache-only`` stops here);
2. build O0--O3 predictions without opening ChangeSim ground truth;
3. hash and verify every prediction;
4. only then open ground truth and calculate Table-3 metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ocmask.cache import load_reconstruction
from ocmask.changesim import MetricAccumulator, load_manifest, normalize_target
from ocmask.config import load_config
from ocmask.stages.obvious_change_sentinel import (
    compose_sentinel,
    evaluate_endpoint_candidates,
    project_masks_with_zbuffer,
    select_large_candidates,
    suppress_added_removed_collisions,
)
from ocmask.stages.sam2_tracking_backend import Sam2MaskTracker
from ocmask.stages.sam3_identity_location import mask_descriptors
from ocmask.stages.sam3_pairwise import (
    load_proposal_cache,
    proposals_to_objects,
    save_proposal_cache,
)
from ocmask.stages.sam3_proposals import Sam3AutomaticMaskGenerator
from ocmask.io import save_image, save_json
from ocmask.types import Label, ObjectMask
from ocmask.visualization import colorize, instance_overlay, overlay


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY
    / "configs/stages/changesim-obvious-object-sentinel-fixed10-densegrid96.yaml"
)
VARIANTS = (
    "o0_conservative_a3",
    "o1_large_mask_only",
    "o2_identity_veto",
    "o3_verified_absence",
)
VARIANT_DESCRIPTIONS = {
    VARIANTS[0]: "Frozen conservative A3 replayed byte-for-byte.",
    VARIANTS[1]: "Large-mask rule only (negative control).",
    VARIANTS[2]: "Global real-image identity absence with moved/replacement vetoes.",
    VARIANTS[3]: "O2 plus geometric observability and explicit SAM2 object absence.",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to recommended_output in the selected config.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Generate/validate real-I0 SAM3 caches, then stop before tracking or GT.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate immutable inputs without loading any GPU model or GT.",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Freeze predictions but do not open ground truth or calculate metrics.",
    )
    parser.add_argument(
        "--force-real-i0",
        action="store_true",
        help="Regenerate only this experiment's real-I0 proposal/feature caches.",
    )
    parser.add_argument(
        "--force-targeted-tracking",
        action="store_true",
        help="Regenerate only this experiment's targeted SAM2 caches.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record pair failures rather than stopping at the first one.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _proposal_objects_hash(objects: Sequence[ObjectMask]) -> str:
    """Reproduce the immutable SAM3 parent's proposal-content digest."""

    digest = hashlib.sha256()
    for obj in objects:
        mask = np.ascontiguousarray(obj.mask, dtype=bool)
        digest.update(str(mask.shape).encode("utf-8"))
        digest.update(np.packbits(mask.reshape(-1)).tobytes())
        digest.update(str(float(obj.score)).encode("utf-8"))
        digest.update(
            json.dumps(obj.metadata, sort_keys=True, default=str).encode("utf-8")
        )
    return digest.hexdigest()


def _parent_pair_fingerprint(
    artifact: Path,
    source_proposals: Sequence[Any],
    target_proposals: Sequence[Any],
) -> str:
    """Recompute the exact cache fingerprint saved by the SAM3 parent."""

    source = proposals_to_objects(list(source_proposals))
    target = proposals_to_objects(list(target_proposals))
    digest = hashlib.sha256()
    for name in (
        "render_0_to_1.png",
        "render_clean_to_1.png",
        "reconstruction.npz",
        "geometry.npz",
        "config.json",
    ):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(artifact / name)))
    digest.update(bytes.fromhex(_proposal_objects_hash(source)))
    digest.update(bytes.fromhex(_proposal_objects_hash(target)))
    return digest.hexdigest()


def _table3(metrics: dict) -> dict[str, dict[str, float]]:
    """Format the exact classes used by ChangeSim Table 3 as percentages."""

    return {
        "binary": {
            "changed": 100.0 * metrics["binary"]["changed"]["iou"],
            "unchanged": 100.0 * metrics["binary"]["unchanged"]["iou"],
            "miou": 100.0 * metrics["binary_miou"],
        },
        "multiclass": {
            **{
                name: 100.0 * metrics["multiclass"][name]["iou"]
                for name in ("added", "removed", "moved", "replaced", "unchanged")
            },
            "miou": 100.0 * metrics["multiclass_miou"],
        },
    }


def _implementation_hash() -> str:
    paths = (
        Path(__file__).resolve(),
        REPOSITORY / "src/ocmask/stages/obvious_change_sentinel.py",
        REPOSITORY / "src/ocmask/stages/sam3_proposals.py",
        REPOSITORY / "src/ocmask/stages/sam3_pairwise.py",
        REPOSITORY / "src/ocmask/stages/sam3_identity_location.py",
        REPOSITORY / "src/ocmask/stages/sam2_tracking_backend.py",
        REPOSITORY / "src/ocmask/adapters/sam2.py",
        REPOSITORY / "src/ocmask/geometry.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing sentinel implementation input: {path}")
        digest.update(str(path.relative_to(REPOSITORY)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _enable_sam3_imports(config: dict) -> Path:
    """Expose the pinned local SAM3 checkout for its intentionally lazy imports."""

    source = _resolve_path(config["sam3"]["source"])
    if not (source / "sam3").is_dir():
        raise FileNotFoundError(f"SAM3 source package is unavailable: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        # Sam3AutomaticMaskGenerator imports `sam3` only when load() runs. The
        # checkout must therefore remain importable for the whole GPU phase.
        sys.path.insert(0, source_text)
    return source


def _load_owned_report(root: Path, name: str) -> tuple[dict, list[str]]:
    report_path = root / "report.json"
    selection_path = root / "selection.json"
    if not report_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"{name} report/selection is incomplete: {root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = list(json.loads(selection_path.read_text(encoding="utf-8"))["ids"])
    if report.get("failures"):
        raise RuntimeError(f"{name} contains failed pairs")
    if not selection or len(selection) != len(set(selection)):
        raise RuntimeError(f"{name} selection must be non-empty and unique")
    records = [str(row["id"]) for row in report.get("pairs", [])]
    if set(records) != set(selection):
        raise RuntimeError(f"{name} report and selection disagree")
    return report, selection


def _validate_roots(config: dict) -> dict[str, Any]:
    roots = {
        "conservative": _resolve_path(config["conservative_parent"]),
        "gate": _resolve_path(config["gate_parent_evaluation"]),
        "geometry": _resolve_path(config["geometry_parent_evaluation"]),
        "proposals": _resolve_path(config["proposal_cache_parent"]),
        "features": _resolve_path(config["identity_feature_parent"]),
    }
    reports: dict[str, dict] = {}
    selection: list[str] | None = None
    for name, root in roots.items():
        report, ids = _load_owned_report(root, name)
        reports[name] = report
        if selection is None:
            selection = ids
        elif ids != selection:
            raise RuntimeError(f"{name} pair order differs from conservative parent")
    assert selection is not None

    configured_selection = json.loads(
        _resolve_path(config["selection"]).read_text(encoding="utf-8")
    )["ids"]
    if list(configured_selection) != selection:
        raise RuntimeError("configured selection differs from immutable parent order")

    conservative_freeze_path = roots["conservative"] / "predictions_frozen.json"
    conservative_freeze = json.loads(
        conservative_freeze_path.read_text(encoding="utf-8")
    )
    parent_variant = str(config["conservative_parent_variant"])
    if parent_variant not in conservative_freeze["variants"]:
        raise RuntimeError(f"parent freeze lacks {parent_variant}")
    frozen_by_id = {str(row["id"]): row for row in conservative_freeze["pairs"]}
    if list(conservative_freeze["selection"]) != selection or set(frozen_by_id) != set(selection):
        raise RuntimeError("conservative prediction freeze differs from selection")

    gate_records = {str(row["id"]): row for row in reports["gate"]["pairs"]}
    geometry_records = {str(row["id"]): row for row in reports["geometry"]["pairs"]}
    # Gate rows contain the exact geometry artifact used by that prediction.
    for pair_id in selection:
        artifact = Path(gate_records[pair_id]["parent_artifacts"]).resolve()
        geometry_artifact = Path(geometry_records[pair_id]["artifacts"]).resolve()
        if artifact != geometry_artifact:
            raise RuntimeError(f"{pair_id}: gate and geometry artifacts disagree")

    return {
        "roots": roots,
        "reports": reports,
        "selection": selection,
        "gate_records": gate_records,
        "geometry_records": geometry_records,
        "conservative_freeze_path": conservative_freeze_path,
        "conservative_frozen": frozen_by_id,
        "parent_variant": parent_variant,
    }


def _selected_ids(args: argparse.Namespace, selection: Sequence[str]) -> list[str]:
    if args.limit is not None and args.pair_id:
        raise ValueError("--limit and --pair-id cannot be combined")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        return list(selection[: args.limit])
    if args.pair_id:
        requested = list(dict.fromkeys(args.pair_id))
        missing = [pair_id for pair_id in requested if pair_id not in selection]
        if missing:
            raise ValueError(f"pair IDs are not in the frozen selection: {missing}")
        return [pair_id for pair_id in selection if pair_id in requested]
    return list(selection)


def _pair_paths(config: dict, context: dict, pair_id: str) -> dict[str, Path]:
    roots = context["roots"]
    geometry = Path(context["gate_records"][pair_id]["parent_artifacts"]).resolve()
    parent_record = context["conservative_frozen"][pair_id]["predictions"][
        context["parent_variant"]
    ]
    parent_labels = roots["conservative"] / parent_record["relative_path"]
    return {
        "geometry_artifact": geometry,
        "reconstruction": geometry / "reconstruction.npz",
        "geometry": geometry / "geometry.npz",
        "baseline_config": geometry / "config.json",
        "parent_labels": parent_labels,
        "i1_proposals": roots["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/target.npz",
        "i1_proposal_metadata": roots["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/metadata.json",
        "identity_features": roots["features"]
        / "pairs"
        / pair_id
        / "sam3_features.npz",
        "identity_feature_metadata": roots["features"]
        / "pairs"
        / pair_id
        / "sam3_features.json",
        "identity_decisions": roots["features"]
        / "pairs"
        / pair_id
        / "decisions.json",
    }


def _validate_pair_inputs(config: dict, context: dict, pair_id: str) -> dict[str, Path]:
    paths = _pair_paths(config, context, pair_id)
    for name, path in paths.items():
        if name == "geometry_artifact":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"{pair_id}: missing {name}: {path}")
    parent_record = context["conservative_frozen"][pair_id]["predictions"][
        context["parent_variant"]
    ]
    labels = np.asarray(Image.open(paths["parent_labels"]), dtype=np.uint8)
    if _sha256_file(paths["parent_labels"]) != parent_record["file_sha256"]:
        raise RuntimeError(f"{pair_id}: conservative parent file differs from freeze")
    if _sha256_array(labels) != parent_record["array_sha256"]:
        raise RuntimeError(f"{pair_id}: conservative parent array differs from freeze")
    feature_metadata = json.loads(
        paths["identity_feature_metadata"].read_text(encoding="utf-8")
    )
    if feature_metadata.get("ground_truth_used"):
        raise RuntimeError(f"{pair_id}: identity feature cache claims GT use")
    if feature_metadata.get("checkpoint_sha256") != config["sam3"]["checkpoint_sha256"]:
        raise RuntimeError(f"{pair_id}: I1 feature checkpoint differs")

    # Old proposal metadata records counts but not content hashes.  Prove that
    # the reused real-I1 cache is nevertheless the exact one consumed by its
    # immutable parent by recomputing that parent's full per-pair fingerprint.
    # This guards against a silently replaced target proposal cache.
    source_parent = load_proposal_cache(
        context["roots"]["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/source.npz"
    )
    target_parent = load_proposal_cache(paths["i1_proposals"])
    parent_row = next(
        row
        for row in context["reports"]["proposals"]["pairs"]
        if str(row["id"]) == pair_id
    )
    # Revision-2 fixed10 predates the persisted fingerprint field. Its cache
    # still has count/checkpoint provenance, but only newer runs permit exact
    # content verification here.
    expected_fingerprint = parent_row.get("pair_fingerprint_sha256")
    if expected_fingerprint is not None:
        actual_fingerprint = _parent_pair_fingerprint(
            paths["geometry_artifact"], source_parent, target_parent
        )
        if actual_fingerprint != expected_fingerprint:
            raise RuntimeError(f"{pair_id}: reused I1 proposal cache differs from parent")
    return paths


def _generation_fingerprint(config: dict, reconstruction_path: Path, image0: np.ndarray) -> str:
    return _sha256_json(
        {
            "reconstruction_sha256": _sha256_file(reconstruction_path),
            "real_i0_array_sha256": _sha256_array(image0),
            "sam3_checkpoint_sha256": config["sam3"]["checkpoint_sha256"],
            "proposal_generation": config["sam3"]["proposal_generation"],
            "feature_protocol": config["sam3"]["features"],
            "input_domain": "real_i0_reconstruction_image0",
        }
    )


def _real_i0_cache_paths(output: Path, pair_id: str) -> dict[str, Path]:
    root = output / "pairs" / pair_id / "real_i0_cache"
    return {
        "root": root,
        "proposals": root / "proposals.npz",
        "features": root / "features.npz",
        "metadata": root / "metadata.json",
    }


def _real_i0_cache_valid(
    paths: Mapping[str, Path], fingerprint: str, checkpoint_hash: str
) -> bool:
    if not all(paths[name].is_file() for name in ("proposals", "features", "metadata")):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("generation_fingerprint_sha256") != fingerprint
        or metadata.get("checkpoint_sha256") != checkpoint_hash
        or metadata.get("ground_truth_used") is not False
        or metadata.get("input_domain") != "real_i0"
    ):
        return False
    try:
        proposals = load_proposal_cache(paths["proposals"])
        with np.load(paths["features"]) as cache:
            feature = np.asarray(cache["feature"])
    except Exception:
        return False
    return (
        len(proposals) == int(metadata.get("proposal_count", -1))
        and list(feature.shape) == list(metadata.get("feature_shape", []))
        and str(feature.dtype) == metadata.get("feature_dtype")
        and _sha256_file(paths["proposals"]) == metadata.get("proposal_file_sha256")
        and _sha256_array(feature) == metadata.get("feature_array_sha256")
    )


def _generator(config: dict) -> Sam3AutomaticMaskGenerator:
    proposal = config["sam3"]["proposal_generation"]
    return Sam3AutomaticMaskGenerator(
        _resolve_path(config["sam3"]["checkpoint"]),
        points_per_side=int(proposal["points_per_side"]),
        points_per_batch=int(proposal["points_per_batch"]),
        pred_iou_threshold=float(proposal["pred_iou_threshold"]),
        stability_threshold=float(proposal["stability_threshold"]),
        stability_offset=float(proposal["stability_offset"]),
        crop_layers=int(proposal["crop_layers"]),
        crop_downscale_factor=int(proposal["crop_downscale_factor"]),
        box_nms_threshold=float(proposal["box_nms_threshold"]),
        crop_nms_threshold=float(proposal["crop_nms_threshold"]),
        minimum_mask_area=int(proposal["minimum_mask_area_pixels"]),
        multimask_output=bool(proposal["multimask_output"]),
    )


def _generate_real_i0_caches(
    config: dict,
    context: dict,
    output: Path,
    pair_ids: Sequence[str],
    *,
    force: bool,
) -> None:
    missing: list[tuple[str, dict[str, Path], str, np.ndarray]] = []
    for pair_id in pair_ids:
        input_paths = _validate_pair_inputs(config, context, pair_id)
        reconstruction = load_reconstruction(input_paths["reconstruction"])
        image0 = np.asarray(reconstruction.images[0], dtype=np.uint8)
        fingerprint = _generation_fingerprint(
            config, input_paths["reconstruction"], image0
        )
        cache_paths = _real_i0_cache_paths(output, pair_id)
        if not force and _real_i0_cache_valid(
            cache_paths, fingerprint, config["sam3"]["checkpoint_sha256"]
        ):
            print(f"[real-I0 cache] {pair_id} (reused)", flush=True)
            continue
        missing.append((pair_id, cache_paths, fingerprint, image0))
    if not missing:
        return

    generator = _generator(config)
    try:
        for number, (pair_id, paths, fingerprint, image0) in enumerate(missing, 1):
            started = time.perf_counter()
            proposals, feature = generator.generate_with_feature_map(image0)
            paths["root"].mkdir(parents=True, exist_ok=True)
            save_proposal_cache(paths["proposals"], proposals, image0.shape[:2])
            np.savez_compressed(paths["features"], feature=np.asarray(feature))
            metadata = {
                "schema_version": 1,
                "pair_id": pair_id,
                "input_domain": "real_i0",
                "input_array_sha256": _sha256_array(image0),
                "generation_fingerprint_sha256": fingerprint,
                "checkpoint_sha256": config["sam3"]["checkpoint_sha256"],
                "proposal_count": len(proposals),
                "proposal_file_sha256": _sha256_file(paths["proposals"]),
                "feature_shape": list(feature.shape),
                "feature_dtype": str(feature.dtype),
                "feature_array_sha256": _sha256_array(feature),
                "elapsed_seconds": time.perf_counter() - started,
                "ground_truth_used": False,
                "shared_backbone_forward": True,
            }
            save_json(paths["metadata"], metadata)
            print(
                f"[real-I0 cache {number}/{len(missing)}] {pair_id}: "
                f"{len(proposals)} proposals, feature {tuple(feature.shape)}",
                flush=True,
            )
    finally:
        generator.release()


def _load_i1_features(paths: Mapping[str, Path]) -> np.ndarray:
    with np.load(paths["identity_features"]) as cache:
        if "target" not in cache:
            raise RuntimeError("identity feature cache lacks target real-I1 map")
        return np.asarray(cache["target"])


def _same_threshold(config: dict, paths: Mapping[str, Path]) -> tuple[float, str]:
    identity = json.loads(paths["identity_decisions"].read_text(encoding="utf-8"))
    calibration = identity["diagnostics"]["calibration"]
    if calibration.get("valid"):
        return float(calibration["threshold"]), "frozen_pair_calibration"
    return (
        float(config["association"]["same_identity_threshold"]["fallback_cosine"]),
        "declared_fallback",
    )


def _attempt_presence(attempt: Any) -> bool | None:
    """Convert one targeted SAM2 attempt into conservative tri-state evidence."""

    reasons = tuple(str(value) for value in attempt.rejection_reasons)
    if bool(attempt.accepted):
        if "object_absent" in reasons:
            raise RuntimeError("targeted track is accepted and object_absent")
        return True
    # Other rejection reasons never prove absence, but SAM2's explicit
    # object-presence logit does even if the resulting mask is also tiny.
    return False if "object_absent" in reasons else None


def _tracking_input_hash(
    config: dict,
    image0: np.ndarray,
    image1: np.ndarray,
    source_ids: Sequence[int],
    source_masks: Sequence[np.ndarray],
    target_ids: Sequence[int],
    target_masks: Sequence[np.ndarray],
    baseline_config: dict,
) -> str:
    return _sha256_json(
        {
            "image0": _sha256_array(image0),
            "image1": _sha256_array(image1),
            "source": [
                [int(proposal_id), _sha256_array(mask)]
                for proposal_id, mask in zip(source_ids, source_masks, strict=True)
            ],
            "target": [
                [int(proposal_id), _sha256_array(mask)]
                for proposal_id, mask in zip(target_ids, target_masks, strict=True)
            ],
            "protocol": config["targeted_absence_verification"],
            "baseline_sam2": baseline_config["sam2"],
            "baseline_tracking": baseline_config["tracking"],
        }
    )


def _save_targeted_cache(
    root: Path,
    source_ids: Sequence[int],
    source_attempts: Sequence[Any],
    target_ids: Sequence[int],
    target_attempts: Sequence[Any],
    *,
    input_hash: str,
    shape: tuple[int, int],
) -> dict:
    if len(source_ids) != len(source_attempts) or len(target_ids) != len(target_attempts):
        raise ValueError("targeted tracking attempt count differs from prompt count")
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("targeted proposal IDs must be unique per endpoint")
    root.mkdir(parents=True, exist_ok=True)
    attempts = list(source_attempts) + list(target_attempts)
    masks = (
        np.stack([np.asarray(attempt.mask, dtype=bool) for attempt in attempts])
        if attempts
        else np.empty((0, *shape), dtype=bool)
    )
    np.savez_compressed(
        root / "attempts.npz",
        masks_packed=np.packbits(masks, axis=2),
        height=np.int32(shape[0]),
        width=np.int32(shape[1]),
    )

    def rows(ids: Sequence[int], values: Sequence[Any], offset: int) -> list[dict]:
        output = []
        for index, (proposal_id, attempt) in enumerate(zip(ids, values, strict=True)):
            output.append(
                {
                    "proposal_id": int(proposal_id),
                    "mask_index": offset + index,
                    "accepted": bool(attempt.accepted),
                    "object_score_logit": (
                        None
                        if attempt.object_score_logit is None
                        else float(attempt.object_score_logit)
                    ),
                    "rejection_reasons": list(attempt.rejection_reasons),
                    "presence": _attempt_presence(attempt),
                    "area": int(np.asarray(attempt.mask, bool).sum()),
                }
            )
        return output

    metadata = {
        "schema_version": 1,
        "input_sha256": input_hash,
        "ground_truth_used": False,
        "qualifying_absence_outcome": "object_absent",
        "source_to_target": rows(source_ids, source_attempts, 0),
        "target_to_source": rows(target_ids, target_attempts, len(source_attempts)),
    }
    metadata["attempt_array_sha256"] = _sha256_file(root / "attempts.npz")
    save_json(root / "metadata.json", metadata)
    return metadata


def _load_targeted_cache(root: Path, *, input_hash: str) -> dict | None:
    metadata_path = root / "metadata.json"
    arrays_path = root / "attempts.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("input_sha256") != input_hash
        or metadata.get("ground_truth_used") is not False
        or metadata.get("qualifying_absence_outcome") != "object_absent"
        or metadata.get("attempt_array_sha256") != _sha256_file(arrays_path)
    ):
        return None
    with np.load(arrays_path) as arrays:
        height, width = int(arrays["height"]), int(arrays["width"])
        packed = np.asarray(arrays["masks_packed"])
        masks = np.unpackbits(packed, axis=2, count=width)[:, :height, :width]
    rows = metadata["source_to_target"] + metadata["target_to_source"]
    if len(masks) != len(rows):
        return None
    for row in rows:
        index = int(row["mask_index"])
        if not 0 <= index < len(masks) or int(masks[index].sum()) != int(row["area"]):
            return None
    return metadata


def _presence_map(rows: Sequence[dict]) -> dict[int, bool | None]:
    return {int(row["proposal_id"]): row.get("presence") for row in rows}


def _add_funnel(total: Counter[str], value: Mapping[str, Any], prefix: str = "") -> None:
    for key, item in value.items():
        if isinstance(item, Mapping):
            _add_funnel(total, item, f"{prefix}{key}.")
        elif isinstance(item, (int, np.integer, bool)):
            total[f"{prefix}{key}"] += int(item)


def _colored_endpoint_overlay(
    image: np.ndarray,
    selection: Any,
    objects: Sequence[ObjectMask],
) -> np.ndarray:
    selected = set(selection.selected_indices)
    subset = [obj for index, obj in enumerate(objects) if index in selected]
    return instance_overlay(image, subset)


def _variant_objects(
    source_result: Any,
    target_result: Any,
    stage: str,
    collision_iou: float,
) -> tuple[list[ObjectMask], list[ObjectMask], list[dict]]:
    removed = list(getattr(source_result, f"{stage}_objects"))
    added = list(getattr(target_result, f"{stage}_objects"))
    added, removed, collisions = suppress_added_removed_collisions(
        added, removed, minimum_iou=collision_iou
    )
    return added, removed, collisions


def _nested_variant_objects(
    source_result: Any,
    target_result: Any,
    collision_iou: float,
) -> dict[str, tuple[list[ObjectMask], list[ObjectMask], list[dict]]]:
    """Apply O1 replacement-collision abstentions to every stricter stage.

    O2/O3 proposal sets are subsets of O1, but independently rerunning the
    collision rule can paradoxically *unmask* one endpoint after its opposite
    endpoint is rejected by a stricter gate.  Carrying the O1 veto forward
    preserves the intended causal nesting and is the conservative choice.
    """

    o1_added, o1_removed, o1_collisions = _variant_objects(
        source_result, target_result, "o1", collision_iou
    )
    blocked_added = {
        int(record["added_proposal_id"]) for record in o1_collisions
    }
    blocked_removed = {
        int(record["removed_proposal_id"]) for record in o1_collisions
    }
    output = {"o1": (o1_added, o1_removed, o1_collisions)}
    for stage in ("o2", "o3"):
        added = [
            obj
            for obj in getattr(target_result, f"{stage}_objects")
            if int(obj.metadata["sentinel_proposal_id"]) not in blocked_added
        ]
        removed = [
            obj
            for obj in getattr(source_result, f"{stage}_objects")
            if int(obj.metadata["sentinel_proposal_id"]) not in blocked_removed
        ]
        added, removed, collisions = suppress_added_removed_collisions(
            added, removed, minimum_iou=collision_iou
        )
        if collisions:
            raise AssertionError(
                f"{stage} introduced a collision absent from its O1 superset"
            )
        output[stage] = (added, removed, [])
    return output


def _run_pair_inference(
    config: dict,
    context: dict,
    output: Path,
    pair_id: str,
    tracker: Sam2MaskTracker | None,
    *,
    force_tracking: bool,
) -> dict[str, Any]:
    paths = _validate_pair_inputs(config, context, pair_id)
    pair_output = output / "pairs" / pair_id
    pair_output.mkdir(parents=True, exist_ok=True)
    real0_cache = _real_i0_cache_paths(output, pair_id)
    reconstruction = load_reconstruction(paths["reconstruction"])
    image0 = np.asarray(reconstruction.images[0], dtype=np.uint8)
    image1 = np.asarray(reconstruction.images[1], dtype=np.uint8)
    shape = tuple(int(value) for value in reconstruction.points[0].shape[:2])
    if image0.shape != (*shape, 3) or image1.shape != (*shape, 3):
        raise RuntimeError(f"{pair_id}: reconstruction images and pointmaps disagree")

    source_objects = proposals_to_objects(load_proposal_cache(real0_cache["proposals"]))
    target_objects = proposals_to_objects(load_proposal_cache(paths["i1_proposals"]))
    with np.load(real0_cache["features"]) as cache:
        source_map = np.asarray(cache["feature"])
    target_map = _load_i1_features(paths)
    minimum_cells = float(config["sam3"]["features"]["minimum_feature_cells"])
    source_features = mask_descriptors(
        source_map, source_objects, minimum_feature_cells=minimum_cells
    )
    target_features = mask_descriptors(
        target_map, target_objects, minimum_feature_cells=minimum_cells
    )

    candidate = config["candidate_selection"]
    duplicate = candidate["duplicate_suppression"]
    selection_kwargs = {
        "minimum_area_fraction": float(candidate["minimum_mask_area_fraction"]),
        "minimum_mask_area": int(
            config["sam3"]["proposal_generation"]["minimum_mask_area_pixels"]
        ),
        "minimum_predicted_iou": float(candidate["minimum_predicted_iou"]),
        "minimum_stability_score": float(candidate["minimum_stability_score"]),
        "duplicate_iou": float(duplicate["mask_iou"]),
        "duplicate_containment": float(duplicate["containment_fraction"]),
        "minimum_bbox_side_fraction": float(candidate["minimum_bbox_side_fraction"]),
        "reject_frame_border": candidate["frame_border_contact_policy"] == "abstain",
    }
    source_selection = select_large_candidates(
        source_objects, source_features, shape, **selection_kwargs
    )
    target_selection = select_large_candidates(
        target_objects, target_features, shape, **selection_kwargs
    )

    # Full pointmap z-buffers establish one deterministic owner per projected
    # destination pixel. No synthetic RGB enters appearance comparison.
    source_to_target = project_masks_with_zbuffer(
        reconstruction.points[0],
        [obj.mask for obj in source_objects],
        reconstruction.intrinsics[1],
        reconstruction.world_to_camera[1],
        shape,
        minimum_depth=float(config["geometry_observability"]["minimum_valid_depth"]),
    )
    target_to_source = project_masks_with_zbuffer(
        reconstruction.points[1],
        [obj.mask for obj in target_objects],
        reconstruction.intrinsics[0],
        reconstruction.world_to_camera[0],
        shape,
        minimum_depth=float(config["geometry_observability"]["minimum_valid_depth"]),
    )
    same_threshold, threshold_origin = _same_threshold(config, paths)
    association = config["association"]
    same_place = association["same_place"]
    geometry = config["geometry_observability"]

    common = {
        "same_threshold": same_threshold,
        "different_margin": float(association["different_identity_margin"]),
        "identity_area_ratio_bounds": tuple(association["area_ratio_bounds"]),
        "replacement_minimum_iou": float(same_place["minimum_projected_mask_iou"]),
        "replacement_maximum_centroid_distance": float(
            same_place["maximum_normalized_centroid_distance"]
        ),
        "visibility_threshold": float(geometry["minimum_observable_fraction"]),
        "depth_epsilon": float(geometry["depth_epsilon"]),
        "output_area_ratio_bounds": tuple(association["area_ratio_bounds"]),
    }
    # First pass determines the exact O2 prompt set. None means targeted
    # absence has not been run and therefore cannot accidentally pass O3.
    source_pre = evaluate_endpoint_candidates(
        "source",
        Label.REMOVED,
        source_objects,
        source_features,
        source_selection,
        target_objects,
        target_features,
        target_to_source.masks,
        source_to_target.masks,
        target_to_source.depth,
        reconstruction.depths[0],
        target_to_source.coverage,
        {},
        opposite_search_indices=target_selection.search_indices,
        **common,
    )
    target_pre = evaluate_endpoint_candidates(
        "target",
        Label.ADDED,
        target_objects,
        target_features,
        target_selection,
        source_objects,
        source_features,
        source_to_target.masks,
        [obj.mask for obj in target_objects],
        source_to_target.depth,
        reconstruction.depths[1],
        source_to_target.coverage,
        {},
        opposite_search_indices=source_selection.search_indices,
        **common,
    )
    source_by_id = {
        int(obj.metadata["automatic_proposal_id"]): obj for obj in source_objects
    }
    target_by_id = {
        int(obj.metadata["automatic_proposal_id"]): obj for obj in target_objects
    }
    source_ids = sorted(
        int(obj.metadata["sentinel_proposal_id"]) for obj in source_pre.o2_objects
    )
    target_ids = sorted(
        int(obj.metadata["sentinel_proposal_id"]) for obj in target_pre.o2_objects
    )
    source_masks = [np.asarray(source_by_id[value].mask, bool) for value in source_ids]
    target_masks = [np.asarray(target_by_id[value].mask, bool) for value in target_ids]
    baseline_config = json.loads(paths["baseline_config"].read_text(encoding="utf-8"))
    track_hash = _tracking_input_hash(
        config,
        image0,
        image1,
        source_ids,
        source_masks,
        target_ids,
        target_masks,
        baseline_config,
    )
    track_root = pair_output / "targeted_tracking_cache"
    track_metadata = None if force_tracking else _load_targeted_cache(
        track_root, input_hash=track_hash
    )
    if track_metadata is None:
        if tracker is None:
            raise RuntimeError("targeted SAM2 tracker is unavailable")
        source_attempts = tracker.track(source_masks, image0, image1)
        target_attempts = tracker.track(target_masks, image1, image0)
        track_metadata = _save_targeted_cache(
            track_root,
            source_ids,
            source_attempts,
            target_ids,
            target_attempts,
            input_hash=track_hash,
            shape=shape,
        )
        tracking_cache_hit = False
    else:
        tracking_cache_hit = True
    source_presence = _presence_map(track_metadata["source_to_target"])
    target_presence = _presence_map(track_metadata["target_to_source"])

    source_result = evaluate_endpoint_candidates(
        "source",
        Label.REMOVED,
        source_objects,
        source_features,
        source_selection,
        target_objects,
        target_features,
        target_to_source.masks,
        source_to_target.masks,
        target_to_source.depth,
        reconstruction.depths[0],
        target_to_source.coverage,
        source_presence,
        opposite_search_indices=target_selection.search_indices,
        **common,
    )
    target_result = evaluate_endpoint_candidates(
        "target",
        Label.ADDED,
        target_objects,
        target_features,
        target_selection,
        source_objects,
        source_features,
        source_to_target.masks,
        [obj.mask for obj in target_objects],
        source_to_target.depth,
        reconstruction.depths[1],
        source_to_target.coverage,
        target_presence,
        opposite_search_indices=source_selection.search_indices,
        **common,
    )

    parent = np.asarray(Image.open(paths["parent_labels"]), dtype=np.uint8)
    predictions = {VARIANTS[0]: parent.copy()}
    composition: dict[str, dict] = {
        VARIANTS[0]: {"parent_replay": True, "added_objects": 0, "removed_objects": 0}
    }
    collisions_by_variant: dict[str, list[dict]] = {}
    collision_iou = float(config["composition"]["replacement_collision_iou"])
    nested_objects = _nested_variant_objects(
        source_result, target_result, collision_iou
    )
    for variant, stage in zip(VARIANTS[1:], ("o1", "o2", "o3"), strict=True):
        added, removed, collisions = nested_objects[stage]
        prediction, diagnostics = compose_sentinel(parent, added, removed)
        predictions[variant] = prediction
        composition[variant] = {
            **diagnostics,
            "added_objects_before_collision_abstention": len(
                getattr(target_result, f"{stage}_objects")
            ),
            "removed_objects_before_collision_abstention": len(
                getattr(source_result, f"{stage}_objects")
            ),
            "collision_abstentions": len(collisions),
        }
        collisions_by_variant[variant] = collisions

    prediction_records: dict[str, dict] = {}
    for variant, labels in predictions.items():
        variant_dir = pair_output / variant
        variant_dir.mkdir(exist_ok=True)
        labels_path = variant_dir / "labels.png"
        save_image(labels_path, labels)
        prediction_records[variant] = {
            "relative_path": str(labels_path.relative_to(output)),
            "file_sha256": _sha256_file(labels_path),
            "array_sha256": _sha256_array(labels),
            "binary_array_sha256": _sha256_array(labels != int(Label.UNCHANGED)),
        }

    # Small qualitative set for the report. The full proposal/mask evidence is
    # retained compactly in NPZ and the candidate ledger for every pair.
    save_image(pair_output / "input0.png", image0)
    save_image(pair_output / "input1.png", image1)
    save_image(
        pair_output / "source_large_candidates.png",
        _colored_endpoint_overlay(image0, source_selection, source_objects),
    )
    save_image(
        pair_output / "target_large_candidates.png",
        _colored_endpoint_overlay(image1, target_selection, target_objects),
    )
    save_image(
        pair_output / "o3_overlay.png",
        overlay(
            np.asarray(
                Image.fromarray(image1).resize(parent.shape[::-1], Image.Resampling.LANCZOS)
            ),
            predictions[VARIANTS[3]],
        ),
    )
    for variant in VARIANTS:
        save_image(pair_output / variant / "labels_color.png", colorize(predictions[variant]))

    targeted_counts = Counter(
        str(row.get("presence"))
        for row in track_metadata["source_to_target"] + track_metadata["target_to_source"]
    )
    funnel = {
        "source": source_result.funnel(),
        "target": target_result.funnel(),
        "source_raw_proposals": len(source_objects),
        "target_raw_proposals": len(target_objects),
        "source_search_pool": len(source_selection.search_indices),
        "target_search_pool": len(target_selection.search_indices),
        "targeted_searches": len(source_ids) + len(target_ids),
        "targeted_object_present": targeted_counts["True"],
        "targeted_object_absent": targeted_counts["False"],
        "targeted_ambiguous": targeted_counts["None"],
        "o3_added_pixels": composition[VARIANTS[3]].get("added_pixels", 0),
        "o3_removed_pixels": composition[VARIANTS[3]].get("removed_pixels", 0),
    }
    ledger = {
        "schema_version": 1,
        "pair_id": pair_id,
        "identity_input_domain": "real_i0_vs_real_i1",
        "synthetic_render_used_for_identity": False,
        "same_identity_threshold": same_threshold,
        "threshold_origin": threshold_origin,
        "source_selection": source_selection.to_dict(),
        "target_selection": target_selection.to_dict(),
        "source_decisions": [value.to_dict() for value in source_result.decisions],
        "target_decisions": [value.to_dict() for value in target_result.decisions],
        "source_projection": {
            "valid_source_points": source_to_target.valid_source_point_count,
            "zbuffer_winners": source_to_target.winner_count,
        },
        "target_projection": {
            "valid_source_points": target_to_source.valid_source_point_count,
            "zbuffer_winners": target_to_source.winner_count,
        },
        "targeted_tracking": {
            "cache_hit": tracking_cache_hit,
            "input_sha256": track_hash,
            "metadata": track_metadata,
        },
        "collisions": collisions_by_variant,
        "composition": composition,
        "funnel": funnel,
        "ground_truth_used": False,
    }
    ledger_path = pair_output / "candidate_ledger.json"
    save_json(ledger_path, ledger)
    return {
        "id": pair_id,
        "status": "prediction_frozen_pending_gt",
        "predictions": prediction_records,
        "sentinel_funnel": funnel,
        "composition": composition,
        "ledger": str(ledger_path.resolve()),
        "artifacts": str(pair_output.resolve()),
        "geometry_artifacts": str(paths["geometry_artifact"]),
        "tracking_cache_hit": tracking_cache_hit,
    }


def _freeze(output: Path, pair_ids: Sequence[str], records: Mapping[str, dict], execution_hash: str) -> dict:
    freeze = {
        "schema_version": 1,
        "execution_sha256": execution_hash,
        "selection": list(pair_ids),
        "variants": list(VARIANTS),
        "prediction_count": len(pair_ids) * len(VARIANTS),
        "ground_truth_opened_before_freeze": False,
        "pairs": [
            {
                "id": pair_id,
                "ground_truth_used": False,
                "predictions": records[pair_id]["predictions"],
            }
            for pair_id in pair_ids
        ],
    }
    save_json(output / "predictions_frozen.json", freeze)
    _verify_freeze(freeze, output)
    return freeze


def _verify_freeze(freeze: dict, output: Path) -> None:
    for pair in freeze["pairs"]:
        for variant in freeze["variants"]:
            record = pair["predictions"][variant]
            path = output / record["relative_path"]
            if _sha256_file(path) != record["file_sha256"]:
                raise RuntimeError(f"{pair['id']}/{variant}: prediction changed after freeze")
            labels = np.asarray(Image.open(path), dtype=np.uint8)
            if _sha256_array(labels) != record["array_sha256"]:
                raise RuntimeError(f"{pair['id']}/{variant}: prediction array changed after freeze")


def _prepare_output(output: Path, config_hash: str) -> None:
    marker = output / "experiment.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():
            raise RuntimeError(f"refusing non-empty unowned output: {output}")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_hash:
            raise RuntimeError("output belongs to a different experiment config")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs").mkdir(exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if tuple(config.get("variants", ())) != VARIANTS:
        raise ValueError(f"config must declare frozen O0-O3 order exactly: {list(VARIANTS)}")
    _enable_sam3_imports(config)
    context = _validate_roots(config)
    pair_ids = _selected_ids(args, context["selection"])
    output = (
        args.output.resolve()
        if args.output is not None
        else _resolve_path(config["recommended_output"])
    )
    for root in context["roots"].values():
        if output == root or root in output.parents:
            raise ValueError(f"output must be outside immutable parent: {root}")
    config_hash = _sha256_json(config)
    _prepare_output(output, config_hash)
    implementation_hash = _implementation_hash()
    execution_hash = _sha256_json(
        {
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "selection": pair_ids,
            "parents": {
                name: _sha256_file(root / "report.json")
                for name, root in context["roots"].items()
            },
        }
    )
    save_json(output / "selection.json", {"ids": pair_ids})
    save_json(
        output / "experiment.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "config": str(config_path),
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "selection": pair_ids,
            "identity_input_domain": "real_i0_vs_real_i1",
            "targeted_absence_search_implemented": True,
            "ground_truth_used_in_inference": False,
        },
    )
    for pair_id in pair_ids:
        _validate_pair_inputs(config, context, pair_id)
    if args.validate_only:
        print(f"Validated {len(pair_ids)} pairs without GPU inference or GT.", flush=True)
        return 0

    _generate_real_i0_caches(
        config, context, output, pair_ids, force=args.force_real_i0
    )
    if args.cache_only:
        save_json(
            output / "cache_phase.json",
            {
                "pairs": list(pair_ids),
                "pairs_cached": len(pair_ids),
                "ground_truth_used": False,
                "complete": True,
            },
        )
        print("Real-I0 proposal/feature cache phase complete.", flush=True)
        return 0

    inference_records: dict[str, dict] = {}
    failures: list[dict] = []
    tracker: Sam2MaskTracker | None = None
    started = time.perf_counter()
    try:
        # Cache hits do not require a GPU tracker. Instantiate lazily only if a
        # pair proves that its exact targeted cache is absent or invalid.
        for number, pair_id in enumerate(pair_ids, 1):
            pair_started = time.perf_counter()
            try:
                if tracker is None:
                    # _run_pair_inference requests a tracker only after it has
                    # checked the pair's targeted cache. A lightweight proxy
                    # creates the actual model on the first cache miss.
                    class _LazyTracker:
                        def __init__(self) -> None:
                            self.inner: Sam2MaskTracker | None = None

                        def track(self, masks, source_image, target_image):
                            if self.inner is None:
                                paths = _pair_paths(config, context, pair_id)
                                baseline = json.loads(
                                    paths["baseline_config"].read_text(encoding="utf-8")
                                )
                                self.inner = Sam2MaskTracker(baseline)
                            return self.inner.track(masks, source_image, target_image)

                        def release(self) -> None:
                            if self.inner is not None:
                                self.inner.release()

                    tracker = _LazyTracker()  # type: ignore[assignment]
                row = _run_pair_inference(
                    config,
                    context,
                    output,
                    pair_id,
                    tracker,
                    force_tracking=args.force_targeted_tracking,
                )
                row["elapsed_seconds"] = time.perf_counter() - pair_started
                inference_records[pair_id] = row
                save_json(output / "pairs" / pair_id / "inference.json", row)
                print(
                    f"[inference {number}/{len(pair_ids)}] {pair_id}: "
                    f"O3 +{row['composition'][VARIANTS[3]].get('added_pixels', 0)} "
                    f"added / +{row['composition'][VARIANTS[3]].get('removed_pixels', 0)} removed px",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "id": pair_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                save_json(output / "pairs" / pair_id / "failure.json", failure)
                if not args.continue_on_error:
                    raise
    finally:
        if tracker is not None:
            tracker.release()
    if failures:
        save_json(output / "failures.json", failures)
        raise RuntimeError(
            f"{len(failures)} prediction pairs failed; no partial metric report was produced"
        )

    freeze = _freeze(output, pair_ids, inference_records, execution_hash)
    if args.predictions_only:
        print("Predictions frozen; ground truth was not opened.", flush=True)
        return 0

    # Ground truth enters for the first time below this line.
    _verify_freeze(freeze, output)
    manifest = {pair.pair_id: pair for pair in load_manifest(_resolve_path(config["manifest"]))}
    missing = [pair_id for pair_id in pair_ids if pair_id not in manifest]
    if missing:
        raise RuntimeError(f"selected pairs are missing from manifest: {missing}")
    accumulators = {variant: MetricAccumulator() for variant in VARIANTS}
    pair_rows: list[dict] = []
    aggregate_funnel: Counter[str] = Counter()
    for pair_id in pair_ids:
        target = normalize_target(manifest[pair_id].target)
        confusions: dict[str, list[list[int]]] = {}
        pair_table3: dict[str, dict] = {}
        for variant in VARIANTS:
            record = inference_records[pair_id]["predictions"][variant]
            prediction = np.asarray(Image.open(output / record["relative_path"]), np.uint8)
            pair_metric = MetricAccumulator()
            pair_metric.add(prediction, target)
            accumulators[variant].add_confusion(pair_metric.confusion)
            confusions[variant] = pair_metric.confusion.tolist()
            pair_table3[variant] = _table3(pair_metric.compute())
        _add_funnel(aggregate_funnel, inference_records[pair_id]["sentinel_funnel"])
        pair_rows.append(
            {
                **inference_records[pair_id],
                "status": "ok",
                "confusion": confusions,
                "table3_iou_percent": pair_table3,
            }
        )
    variants = {
        variant: {
            "description": VARIANT_DESCRIPTIONS[variant],
            "metrics": accumulators[variant].compute(),
            "table3_iou_percent": _table3(accumulators[variant].compute()),
        }
        for variant in VARIANTS
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "elapsed_seconds": time.perf_counter() - started,
        "failures": [],
        "variants": variants,
        "sentinel_funnel": dict(sorted(aggregate_funnel.items())),
        "candidate_funnel": dict(sorted(aggregate_funnel.items())),
        "pairs": pair_rows,
        "protocol": {
            "pairs_selected": len(pair_ids),
            "pairs_succeeded": len(pair_rows),
            "ground_truth_used_in_inference": False,
            "predictions_frozen_before_current_gt_evaluation": True,
            "ground_truth_opened_before_freeze": False,
            "identity_input_domain": "real_i0_vs_real_i1",
            "targeted_absence_search_implemented": True,
            "targeted_absence_qualifying_outcome": "object_absent_only",
            "parent_changed_pixels_are_immutable": True,
            "selection_role": config["protocol"]["selection_role"],
        },
        "provenance": {
            "config": str(config_path),
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "prediction_freeze_sha256": _sha256_file(output / "predictions_frozen.json"),
            "identity_input_domain": "real_i0_vs_real_i1",
            "synthetic_render_used_for_identity": False,
            "targeted_absence_search_implemented": True,
        },
    }
    save_json(output / "report.json", report)
    print(json.dumps({key: value["table3_iou_percent"] for key, value in variants.items()}, indent=2))
    print(f"Report written to {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
