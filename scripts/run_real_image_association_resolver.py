#!/usr/bin/env python3
"""Run the corrected real-I0/real-I1 instance-association experiment.

The runner reuses frozen SAM3 proposals/features and MASt3R geometry from the
obvious-object sentinel. It creates a new output root and never edits a parent
prediction or cache. Ground truth is opened only after every prediction has
been written, hashed, and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import run_obvious_object_sentinel_experiment as sentinel
from ocmask.cache import load_reconstruction
from ocmask.changesim import MetricAccumulator, load_manifest, normalize_target
from ocmask.config import load_config
from ocmask.stages.obvious_change_sentinel import (
    compose_sentinel,
    geometry_support,
    project_masks_with_zbuffer,
    select_large_candidates,
)
from ocmask.stages.real_image_association_resolver import (
    associate_real_image_instances,
    pair_joint_replacements,
    paint_joint_replacements,
    surrounding_scene_support,
    unmatched_inventory_indices,
)
from ocmask.stages.sam2_tracking_backend import Sam2MaskTracker
from ocmask.stages.sam3_identity_location import (
    cosine_similarity_matrix,
    mask_descriptors,
)
from ocmask.stages.sam3_pairwise import load_proposal_cache, proposals_to_objects
from ocmask.io import save_image, save_json
from ocmask.types import Label, ObjectMask


VARIANTS = (
    "r0_conservative_a3",
    "r1_hard_collision_o3",
    "r2_association_endpoints",
    "r3_positive_joint_replacement",
    "r4_no_geometry_ablation",
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stages/changesim-real-image-association-resolver-fixed10-densegrid96.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--force-tracking", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _implementation_hash() -> str:
    """Digest every local inference component introduced or reused here."""

    paths = (
        Path(__file__).resolve(),
        REPOSITORY / "src/ocmask/stages/real_image_association_resolver.py",
        REPOSITORY / "src/ocmask/stages/obvious_change_sentinel.py",
        REPOSITORY / "src/ocmask/stages/sam3_identity_location.py",
        REPOSITORY / "src/ocmask/stages/sam2_tracking_backend.py",
        REPOSITORY / "src/ocmask/adapters/sam2.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPOSITORY)).encode())
        digest.update(bytes.fromhex(sentinel._sha256_file(path)))
    return digest.hexdigest()


def _table3(metrics: dict) -> dict[str, dict[str, float]]:
    return sentinel._table3(metrics)


def _copy_object(obj: ObjectMask, mask: np.ndarray, label: Label) -> ObjectMask:
    metadata = dict(obj.metadata)
    metadata["sentinel_proposal_id"] = int(metadata["automatic_proposal_id"])
    metadata["corrected_resolver"] = True
    return ObjectMask(
        mask=np.asarray(mask, bool).copy(),
        score=float(obj.score),
        label=label,
        source=f"real_image_association_{label.name.lower()}",
        metadata=metadata,
    )


def _selection_kwargs(parent_config: dict) -> dict[str, Any]:
    candidate = parent_config["candidate_selection"]
    duplicate = candidate["duplicate_suppression"]
    return {
        "minimum_area_fraction": float(candidate["minimum_mask_area_fraction"]),
        "minimum_bbox_side_fraction": float(candidate["minimum_bbox_side_fraction"]),
        "minimum_mask_area": int(parent_config["sam3"]["proposal_generation"]["minimum_mask_area_pixels"]),
        "minimum_predicted_iou": float(candidate["minimum_predicted_iou"]),
        "minimum_stability_score": float(candidate["minimum_stability_score"]),
        "duplicate_iou": float(duplicate["mask_iou"]),
        "duplicate_containment": float(duplicate["containment_fraction"]),
        "reject_frame_border": candidate["frame_border_contact_policy"] == "abstain",
    }


def _full_similarity(source_features, target_features) -> np.ndarray:
    return cosine_similarity_matrix(source_features, target_features)


def _load_context(config: dict) -> tuple[Path, dict, dict, list[str]]:
    parent_root = sentinel._resolve_path(config["sentinel_parent"])
    parent_config_path = sentinel._resolve_path(config["sentinel_parent_config"])
    parent_config = load_config(parent_config_path)
    parent_experiment = json.loads((parent_root / "experiment.json").read_text())
    if parent_experiment["config_sha256"] != sentinel._sha256_json(parent_config):
        raise RuntimeError("sentinel parent config no longer matches its frozen run")
    context = sentinel._validate_roots(parent_config)
    selection = list(json.loads((parent_root / "selection.json").read_text())["ids"])
    if selection != context["selection"]:
        raise RuntimeError("sentinel selection differs from immutable parents")
    return parent_root, parent_config, context, selection


def _selected(args: argparse.Namespace, available: Sequence[str]) -> list[str]:
    if args.limit is not None and args.pair_id:
        raise ValueError("--limit and --pair-id cannot be combined")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        return list(available[: args.limit])
    if args.pair_id:
        missing = sorted(set(args.pair_id) - set(available))
        if missing:
            raise ValueError(f"unknown pair IDs: {missing}")
        return [pair_id for pair_id in available if pair_id in set(args.pair_id)]
    return list(available)


def _tracking_cache(
    output: Path,
    pair_id: str,
    config: dict,
    baseline: dict,
    image0: np.ndarray,
    image1: np.ndarray,
    source_ids: list[int],
    source_masks: list[np.ndarray],
    target_ids: list[int],
    target_masks: list[np.ndarray],
    tracker: Any,
    force: bool,
) -> tuple[dict, bool]:
    protocol_config = {
        "targeted_absence_verification": {
            **config["endpoint_verification"],
            "experiment": config["experiment_id"],
        }
    }
    input_hash = sentinel._tracking_input_hash(
        protocol_config,
        image0,
        image1,
        source_ids,
        source_masks,
        target_ids,
        target_masks,
        baseline,
    )
    root = output / "pairs" / pair_id / "targeted_tracking_cache"
    cached = None if force else sentinel._load_targeted_cache(root, input_hash=input_hash)
    if cached is not None:
        return cached, True
    # Adopt an immutable earlier result only if both directional prompt-ID
    # lists are byte-for-byte equivalent at the object-contract level. Images,
    # masks, and model settings are already pinned by the parent experiment;
    # the exact ID ordering proves that batching receives the same prompts.
    if not force:
        for parent_value in config.get("tracking_cache_parents", []):
            parent = sentinel._resolve_path(parent_value) / "pairs" / pair_id / "targeted_tracking_cache"
            metadata_path, arrays_path = parent / "metadata.json", parent / "attempts.npz"
            if not metadata_path.is_file() or not arrays_path.is_file():
                continue
            candidate = json.loads(metadata_path.read_text())
            candidate_source = [int(row["proposal_id"]) for row in candidate.get("source_to_target", [])]
            candidate_target = [int(row["proposal_id"]) for row in candidate.get("target_to_source", [])]
            if (
                candidate.get("ground_truth_used") is not False
                or candidate.get("qualifying_absence_outcome") != "object_absent"
                or candidate_source != source_ids
                or candidate_target != target_ids
                or candidate.get("attempt_array_sha256") != sentinel._sha256_file(arrays_path)
            ):
                continue
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(arrays_path, root / "attempts.npz")
            candidate["input_sha256"] = input_hash
            candidate["attempt_array_sha256"] = sentinel._sha256_file(root / "attempts.npz")
            candidate["adopted_from"] = str(parent.resolve())
            candidate["adoption_contract"] = "same_pair_direction_and_ordered_proposal_ids"
            save_json(root / "metadata.json", candidate)
            validated = sentinel._load_targeted_cache(root, input_hash=input_hash)
            if validated is None:
                raise RuntimeError("adopted targeted cache failed validation")
            return validated, True
    source_attempts = tracker.track(source_masks, image0, image1)
    target_attempts = tracker.track(target_masks, image1, image0)
    return sentinel._save_targeted_cache(
        root,
        source_ids,
        source_attempts,
        target_ids,
        target_attempts,
        input_hash=input_hash,
        shape=image0.shape[:2],
    ), False


def _compose_variant(parent, added, removed, replacements, projected, target_objects):
    labels, diagnostics = compose_sentinel(parent, added, removed)
    if replacements is not None:
        labels, replacement_diagnostics = paint_joint_replacements(
            labels, parent, replacements, projected, target_objects
        )
        diagnostics.update(replacement_diagnostics)
    return labels, diagnostics


def _run_pair(
    config: dict,
    parent_config: dict,
    context: dict,
    sentinel_root: Path,
    output: Path,
    pair_id: str,
    tracker: Any,
    force_tracking: bool,
) -> dict:
    paths = sentinel._validate_pair_inputs(parent_config, context, pair_id)
    reconstruction = load_reconstruction(paths["reconstruction"])
    image0 = np.asarray(reconstruction.images[0], np.uint8)
    image1 = np.asarray(reconstruction.images[1], np.uint8)
    shape = image0.shape[:2]

    real0 = sentinel._real_i0_cache_paths(sentinel_root, pair_id)
    source_objects = proposals_to_objects(load_proposal_cache(real0["proposals"]))
    target_objects = proposals_to_objects(load_proposal_cache(paths["i1_proposals"]))
    with np.load(real0["features"]) as cache:
        source_map = np.asarray(cache["feature"])
    target_map = sentinel._load_i1_features(paths)
    minimum_cells = float(parent_config["sam3"]["features"]["minimum_feature_cells"])
    source_features = mask_descriptors(source_map, source_objects, minimum_feature_cells=minimum_cells)
    target_features = mask_descriptors(target_map, target_objects, minimum_feature_cells=minimum_cells)
    source_selection = select_large_candidates(source_objects, source_features, shape, **_selection_kwargs(parent_config))
    target_selection = select_large_candidates(target_objects, target_features, shape, **_selection_kwargs(parent_config))

    source_to_target = project_masks_with_zbuffer(
        reconstruction.points[0],
        [obj.mask for obj in source_objects],
        reconstruction.intrinsics[1],
        reconstruction.world_to_camera[1],
        shape,
        minimum_depth=float(parent_config["geometry_observability"]["minimum_valid_depth"]),
    )
    target_to_source = project_masks_with_zbuffer(
        reconstruction.points[1],
        [obj.mask for obj in target_objects],
        reconstruction.intrinsics[0],
        reconstruction.world_to_camera[0],
        shape,
        minimum_depth=float(parent_config["geometry_observability"]["minimum_valid_depth"]),
    )

    threshold, threshold_origin = sentinel._same_threshold(parent_config, paths)
    association_cfg = config["association"]
    matches, _ = associate_real_image_instances(
        source_objects,
        target_objects,
        source_features,
        target_features,
        source_selection.selected_indices,
        target_selection.selected_indices,
        minimum_cosine=threshold,
        minimum_margin=float(association_cfg["minimum_bidirectional_margin"]),
        area_ratio_bounds=tuple(association_cfg["area_ratio_bounds"]),
        require_mutual_nearest=bool(association_cfg["require_mutual_nearest"]),
    )
    unmatched_source, unmatched_target = unmatched_inventory_indices(
        source_selection.selected_indices, target_selection.selected_indices, matches
    )
    source_ids = [int(source_objects[index].metadata["automatic_proposal_id"]) for index in unmatched_source]
    target_ids = [int(target_objects[index].metadata["automatic_proposal_id"]) for index in unmatched_target]
    source_masks = [np.asarray(source_objects[index].mask, bool) for index in unmatched_source]
    target_masks = [np.asarray(target_objects[index].mask, bool) for index in unmatched_target]
    baseline = json.loads(paths["baseline_config"].read_text())
    tracking, cache_hit = _tracking_cache(
        output,
        pair_id,
        config,
        baseline,
        image0,
        image1,
        source_ids,
        source_masks,
        target_ids,
        target_masks,
        tracker,
        force_tracking,
    )
    source_presence = sentinel._presence_map(tracking["source_to_target"])
    target_presence = sentinel._presence_map(tracking["target_to_source"])

    alpha = float(config["endpoint_verification"]["geometry_visibility_alpha"])
    epsilon = float(config["endpoint_verification"]["depth_epsilon"])
    source_absent: list[int] = []
    target_absent: list[int] = []
    source_geometry: dict[int, Any] = {}
    target_geometry: dict[int, Any] = {}
    source_ring: dict[int, Any] = {}
    target_ring: dict[int, Any] = {}
    ring_kwargs = {
        "ring_radius": int(config["endpoint_verification"]["surrounding_ring_radius_pixels"]),
        "minimum_ring_pixels": int(config["endpoint_verification"]["surrounding_ring_minimum_pixels"]),
        "visibility_threshold": alpha,
        "relative_depth_tolerance": float(config["endpoint_verification"]["surrounding_ring_relative_depth_tolerance"]),
    }
    for index, proposal_id in zip(unmatched_source, source_ids, strict=True):
        evidence = geometry_support(
            source_objects[index].mask,
            target_to_source.depth,
            reconstruction.depths[0],
            target_to_source.coverage,
            visibility_threshold=alpha,
            depth_epsilon=epsilon,
        )
        source_geometry[index] = evidence
        source_ring[index] = surrounding_scene_support(
            source_objects[index].mask,
            target_to_source.depth,
            reconstruction.depths[0],
            target_to_source.coverage,
            **ring_kwargs,
        )
        if source_presence.get(proposal_id) is False:
            source_absent.append(index)
    for index, proposal_id in zip(unmatched_target, target_ids, strict=True):
        evidence = geometry_support(
            target_objects[index].mask,
            source_to_target.depth,
            reconstruction.depths[1],
            source_to_target.coverage,
            visibility_threshold=alpha,
            depth_epsilon=epsilon,
        )
        target_geometry[index] = evidence
        target_ring[index] = surrounding_scene_support(
            target_objects[index].mask,
            source_to_target.depth,
            reconstruction.depths[1],
            source_to_target.coverage,
            **ring_kwargs,
        )
        if target_presence.get(proposal_id) is False:
            target_absent.append(index)

    # Geometry is supporting evidence: either a trustworthy object interior or
    # a mutually visible static ring may establish that the location was seen.
    source_supported = [
        index for index in source_absent
        if source_geometry[index].passed or source_ring[index].passed
    ]
    target_supported = [
        index for index in target_absent
        if target_geometry[index].passed or target_ring[index].passed
    ]
    removed = [_copy_object(source_objects[index], source_to_target.masks[index], Label.REMOVED) for index in source_supported]
    added = [_copy_object(target_objects[index], target_objects[index].mask, Label.ADDED) for index in target_supported]
    removed_no_geometry = [_copy_object(source_objects[index], source_to_target.masks[index], Label.REMOVED) for index in source_absent]
    added_no_geometry = [_copy_object(target_objects[index], target_objects[index].mask, Label.ADDED) for index in target_absent]

    similarity = _full_similarity(source_features, target_features)
    replacement_limit = threshold - float(config["replacement"]["different_identity_margin"])
    replacement_pairs = pair_joint_replacements(
        source_objects,
        target_objects,
        source_supported,
        target_supported,
        source_to_target.masks,
        target_to_source.masks,
        minimum_iou=float(config["replacement"]["minimum_directional_iou"]),
        identity_similarity=similarity,
        maximum_identity_cosine=replacement_limit,
    )
    raw_source_masks = [np.asarray(obj.mask, bool) for obj in source_objects]
    raw_target_masks = [np.asarray(obj.mask, bool) for obj in target_objects]
    no_geometry_pairs = pair_joint_replacements(
        source_objects,
        target_objects,
        source_absent,
        target_absent,
        raw_source_masks,
        raw_target_masks,
        minimum_iou=float(config["replacement"]["minimum_directional_iou"]),
        identity_similarity=similarity,
        maximum_identity_cosine=replacement_limit,
    )

    parent = np.asarray(Image.open(paths["parent_labels"]), np.uint8)
    old_o3 = np.asarray(Image.open(sentinel_root / "pairs" / pair_id / "o3_verified_absence/labels.png"), np.uint8)
    predictions = {
        VARIANTS[0]: parent.copy(),
        VARIANTS[1]: old_o3.copy(),
    }
    compositions = {
        VARIANTS[0]: {"parent_replay": True},
        VARIANTS[1]: {"old_o3_replay": True},
    }
    predictions[VARIANTS[2]], compositions[VARIANTS[2]] = _compose_variant(parent, added, removed, None, source_to_target.masks, target_objects)
    predictions[VARIANTS[3]], compositions[VARIANTS[3]] = _compose_variant(parent, added, removed, replacement_pairs, source_to_target.masks, target_objects)
    predictions[VARIANTS[4]], compositions[VARIANTS[4]] = _compose_variant(parent, added_no_geometry, removed_no_geometry, no_geometry_pairs, raw_source_masks, target_objects)

    records = {}
    pair_root = output / "pairs" / pair_id
    for variant, labels in predictions.items():
        path = pair_root / variant / "labels.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(path, labels)
        records[variant] = {
            "relative_path": str(path.relative_to(output)),
            "file_sha256": sentinel._sha256_file(path),
            "array_sha256": sentinel._sha256_array(labels),
            "binary_array_sha256": sentinel._sha256_array(labels != int(Label.UNCHANGED)),
        }
    ledger = {
        "schema_version": 1,
        "pair_id": pair_id,
        "identity_domain": "real_i0_vs_real_i1",
        "geometry_in_identity_assignment": False,
        "same_identity_threshold": threshold,
        "threshold_origin": threshold_origin,
        "association_matches": [value.to_dict() for value in matches],
        "unmatched_source_ids": source_ids,
        "unmatched_target_ids": target_ids,
        "source_presence": {str(k): v for k, v in source_presence.items()},
        "target_presence": {str(k): v for k, v in target_presence.items()},
        "source_geometry": {str(source_objects[k].metadata["automatic_proposal_id"]): v.to_dict() for k, v in source_geometry.items()},
        "target_geometry": {str(target_objects[k].metadata["automatic_proposal_id"]): v.to_dict() for k, v in target_geometry.items()},
        "source_surrounding_scene": {str(source_objects[k].metadata["automatic_proposal_id"]): v.to_dict() for k, v in source_ring.items()},
        "target_surrounding_scene": {str(target_objects[k].metadata["automatic_proposal_id"]): v.to_dict() for k, v in target_ring.items()},
        "replacement_pairs": [value.to_dict() for value in replacement_pairs],
        "no_geometry_replacement_pairs": [value.to_dict() for value in no_geometry_pairs],
        "composition": compositions,
        "targeted_tracking_cache_hit": cache_hit,
        "ground_truth_used": False,
    }
    ledger_path = pair_root / "resolver_ledger.json"
    save_json(ledger_path, ledger)
    return {
        "id": pair_id,
        "predictions": records,
        "ledger": str(ledger_path.resolve()),
        "artifacts": str(pair_root.resolve()),
        "diagnostics": {
            "identity_matches": len(matches),
            "unmatched_source": len(unmatched_source),
            "unmatched_target": len(unmatched_target),
            "verified_absent_source": len(source_absent),
            "verified_absent_target": len(target_absent),
            "geometry_supported_source": len(source_supported),
            "geometry_supported_target": len(target_supported),
            "joint_replacements": len(replacement_pairs),
            "tracking_cache_hit": cache_hit,
        },
    }


def _freeze(output: Path, selection: Sequence[str], records: dict[str, dict], execution_hash: str) -> dict:
    freeze = {
        "schema_version": 1,
        "execution_sha256": execution_hash,
        "selection": list(selection),
        "variants": list(VARIANTS),
        "ground_truth_opened_before_freeze": False,
        "pairs": [{"id": pair_id, "predictions": records[pair_id]["predictions"], "ground_truth_used": False} for pair_id in selection],
    }
    save_json(output / "predictions_frozen.json", freeze)
    for row in freeze["pairs"]:
        for variant in VARIANTS:
            record = row["predictions"][variant]
            path = output / record["relative_path"]
            if sentinel._sha256_file(path) != record["file_sha256"]:
                raise RuntimeError("prediction changed before evaluation")
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if tuple(config["variants"]) != VARIANTS:
        raise ValueError("resolver variant order differs from the registered protocol")
    sentinel_root, parent_config, context, available = _load_context(config)
    selection = _selected(args, available)
    output = (args.output.resolve() if args.output else sentinel._resolve_path(config["recommended_output"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs").mkdir(exist_ok=True)
    config_hash = _hash_json(config)
    implementation_hash = _implementation_hash()
    execution_hash = _hash_json({"config": config_hash, "implementation": implementation_hash, "selection": selection})
    marker = output / "experiment.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("execution_sha256") != execution_hash:
            raise RuntimeError("output belongs to a different resolver execution")
    save_json(marker, {"schema_version": 1, "experiment_id": config["experiment_id"], "config": str(config_path), "config_sha256": config_hash, "implementation_sha256": implementation_hash, "execution_sha256": execution_hash, "selection": selection, "ground_truth_used_in_inference": False})
    save_json(output / "selection.json", {"ids": selection})

    records: dict[str, dict] = {}
    failures = []
    tracker: Any = None

    class LazyTracker:
        def __init__(self):
            self.inner = None

        def track(self, masks, source_image, target_image):
            if self.inner is None:
                pair_paths = sentinel._pair_paths(parent_config, context, selection[0])
                self.inner = Sam2MaskTracker(json.loads(pair_paths["baseline_config"].read_text()))
            return self.inner.track(masks, source_image, target_image)

        def release(self):
            if self.inner is not None:
                self.inner.release()

    tracker = LazyTracker()
    started = time.perf_counter()
    try:
        for number, pair_id in enumerate(selection, 1):
            try:
                records[pair_id] = _run_pair(config, parent_config, context, sentinel_root, output, pair_id, tracker, args.force_tracking)
                print(f"[{number}/{len(selection)}] {pair_id}: {records[pair_id]['diagnostics']}", flush=True)
            except Exception as exc:
                failures.append({"id": pair_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                print(f"[{number}/{len(selection)}] {pair_id}: FAILED {exc}", flush=True)
                if not args.continue_on_error:
                    raise
    finally:
        tracker.release()
    if failures or len(records) != len(selection):
        save_json(output / "failures.json", failures)
        return 1
    _freeze(output, selection, records, execution_hash)
    if args.predictions_only:
        return 0

    manifest = {pair.pair_id: pair for pair in load_manifest(sentinel._resolve_path(parent_config["manifest"]))}
    accumulators = {variant: MetricAccumulator() for variant in VARIANTS}
    for pair_id in selection:
        target = normalize_target(manifest[pair_id].target)
        for variant in VARIANTS:
            labels = np.asarray(Image.open(output / records[pair_id]["predictions"][variant]["relative_path"]), np.uint8)
            accumulators[variant].add(labels, target)
    variants = {variant: {"metrics": accumulators[variant].compute()} for variant in VARIANTS}
    for value in variants.values():
        value["table3_iou_percent"] = _table3(value["metrics"])
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "selection": selection,
        "variants": variants,
        "pairs": list(records.values()),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": {**config["protocol"], "predictions_frozen_before_ground_truth": True, "pairs": len(selection)},
        "provenance": {"config_sha256": config_hash, "implementation_sha256": implementation_hash, "execution_sha256": execution_hash, "sentinel_parent": str(sentinel_root), "ground_truth_used_in_inference": False},
    }
    save_json(output / "report.json", report)
    print(json.dumps({name: value["table3_iou_percent"] for name, value in variants.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
