#!/usr/bin/env python3
"""Run object-slot replacement inference and measure actual IoU against R4."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ocmask.cache import load_reconstruction
from ocmask.changesim import MetricAccumulator, normalize_target
from ocmask.config import load_config
from ocmask.stages.branch_b2 import ConsolidationSettings, consolidate_hypotheses
from ocmask.stages.sam3_pairwise import load_cached_inputs, load_proposal_cache, proposals_to_objects
from ocmask.stages.slot_inconsistency import (
    MaskGeometry,
    SlotMatch,
    SlotSettings,
    aligned_patch_retention,
    arbitrate_target_object_classes,
    color_intersection,
    decide_slot_replacement,
    descriptor_cosine,
    find_identity_elsewhere,
    floor_aligned_added_components,
    frontmost_replacement_ownership,
    mask_geometry,
    match_object_slots,
    rasterize_replacement_labels,
    relabel_changed_target_fragments,
    replacement_cleanup_evidence,
    replacement_companion_evidence,
    replacement_object_plausibility,
    support_surface_evidence,
)
from ocmask.artifacts import load_feature_maps, manifest_targets, parent_records, resolve_path
from ocmask.geometry import render_points
from ocmask.ground_contact import detect_floor_plane, up_vector_from_world_to_camera
from ocmask.io import save_image, save_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sam3_moved_association_experiment import load_tracking_cache, recover_changed_candidates  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/stages/stage11_object_consistent_masks.yaml"
DEFAULT_OUTPUT = REPOSITORY / "outputs/s11_object_consistent_masks"


def _resolve(path: str | Path) -> Path:
    return resolve_path(path, REPOSITORY)


PALETTE = np.asarray([[0, 0, 0], [30, 210, 80], [235, 70, 50], [255, 165, 20], [30, 160, 230], [205, 60, 235]], np.uint8)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", default="fixed10,new15")
    parser.add_argument("--pairs", default=None)
    return parser.parse_args()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlay(image: np.ndarray, mask: np.ndarray, color=(205, 60, 235)) -> np.ndarray:
    output = np.asarray(image, np.uint8).copy(); selected = np.asarray(mask, bool)
    output[selected] = (0.45 * output[selected] + 0.55 * np.asarray(color)).astype(np.uint8)
    boundary = selected ^ ndimage.binary_erosion(selected)
    output[boundary] = 255
    return output


def _crop_box(mask: np.ndarray, padding: int = 10) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if not len(xs):
        return 0, 0, width, height
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
        min(height, int(ys.max()) + padding + 1),
    )


def _save_crop(path: Path, image: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    crop = np.asarray(image)[y0:y1, x0:x1]
    if not crop.size:
        crop = np.zeros((8, 8, 3), np.uint8)
    height, width = crop.shape[:2]
    scale = min(6, max(1, int(np.ceil(220 / max(height, width, 1)))))
    if scale > 1:
        crop = np.asarray(
            Image.fromarray(np.asarray(crop, np.uint8)).resize(
                (width * scale, height * scale), Image.Resampling.NEAREST
            )
        )
    save_image(path, crop)


def _resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return labels if labels.shape == shape else np.asarray(Image.fromarray(np.asarray(labels, np.uint8)).resize(shape[::-1], Image.Resampling.NEAREST), np.uint8)


def _candidate_truth(mask: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(truth[mask], return_counts=True)
    names = ["unchanged", "added", "removed", "moved", "warped", "replaced"]
    return {
        "distribution": {names[int(v)]: int(c) for v, c in zip(values, counts, strict=True)},
        "dominant": names[int(values[np.argmax(counts)])] if len(values) else "unchanged",
        "purity": float(counts.max() / counts.sum()) if len(counts) else 0.0,
    }


def _replacement_depth_ownership(
    artifact: Path,
    baseline: np.ndarray,
    target_masks: list[np.ndarray],
    identity_override_target_masks: list[np.ndarray],
    settings: SlotSettings,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not target_masks:
        empty = np.zeros(baseline.shape, bool)
        return empty, empty, {
            "candidate_pixels": 0,
            "candidate_depth_known_pixels": 0,
            "overlap_pixels": 0,
            "frontmost_promoted_overlap_pixels": 0,
            "forced_nonremoved_unification_pixels": 0,
            "identity_override_removed_overlap_pixels": 0,
            "depth_front_removed_overlap_pixels": 0,
            "source_front_removed_overlap_pixels": 0,
            "unknown_depth_removed_overlap_pixels": 0,
            "cleanup_protected_removed_pixels": 0,
            "preserved_existing_overlap_pixels": 0,
            "promoted_overlap_by_baseline_label": {},
            "preserved_overlap_by_baseline_label": {},
            "baseline_component_count": 0,
            "unified_target_objects": True,
            "unified_object_pixels": 0,
            "depth_tie_margin_m": settings.depth_ownership_tie_margin_m,
            "targets": [],
        }
    reconstruction = load_reconstruction(artifact / "reconstruction.npz")
    geometry_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))["geometry"]
    _, source_depth, _ = render_points(
        reconstruction.points[0],
        reconstruction.images[0],
        reconstruction.intrinsics[1],
        reconstruction.world_to_camera[1],
        reconstruction.depths[1].shape,
        z_epsilon=geometry_config["z_buffer_epsilon"],
        splat_radius=geometry_config.get("splat_radius", 0),
        fill_holes=geometry_config.get("hole_fill_enabled", False),
        hole_fill_min_neighbors=geometry_config.get("hole_fill_min_neighbors", 5),
        hole_fill_max_relative_depth=geometry_config.get("hole_fill_max_relative_depth", 0.02),
    )
    return frontmost_replacement_ownership(
        baseline,
        target_masks,
        reconstruction.depths[1],
        source_depth,
        depth_tie_margin_m=settings.depth_ownership_tie_margin_m,
        minimum_valid_pixels=settings.depth_ownership_minimum_pixels,
        minimum_valid_fraction=settings.depth_ownership_minimum_fraction,
        unify_target_objects=True,
        identity_override_target_masks=identity_override_target_masks,
    )


def _floor_suppression(
    artifact: Path,
    baseline: np.ndarray,
    settings: SlotSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    reconstruction = load_reconstruction(artifact / "reconstruction.npz")
    with np.load(artifact / "geometry.npz") as geometry:
        keep0 = np.asarray(geometry["keep0"], bool)
        keep1 = np.asarray(geometry["keep1"], bool)
    static_points = np.concatenate([
        reconstruction.points[0][keep0],
        reconstruction.points[1][keep1],
    ])
    static_colors = np.concatenate([
        reconstruction.images[0][keep0],
        reconstruction.images[1][keep1],
    ])
    up = up_vector_from_world_to_camera(reconstruction.world_to_camera[1])
    floor = detect_floor_plane(static_points, static_colors, up)
    if floor is None:
        return np.zeros(baseline.shape, bool), {
            "floor_found": False,
            "suppressed_component_count": 0,
            "suppressed_pixels": 0,
        }
    suppressed, summary = floor_aligned_added_components(
        baseline,
        reconstruction.points[1],
        floor.point,
        floor.normal,
        distance_tolerance_m=settings.floor_distance_tolerance_m,
        minimum_component_pixels=settings.floor_component_minimum_pixels,
        minimum_floor_fraction=settings.floor_component_minimum_fraction,
    )
    return suppressed, {
        "floor_found": True,
        "floor_inlier_count": floor.inlier_count,
        "floor_residual_rms_m": floor.residual_rms,
        "floor_up_agreement_cosine": floor.up_agreement_cosine,
        **summary,
    }


def _target_change_evidence(
    artifact: Path,
    targets: list,
) -> tuple[list[bool], dict[str, Any]]:
    """Map explicit target→clean absence evidence onto consolidated objects."""

    ledger_path = artifact / "tracking_attempts.json"
    if not ledger_path.is_file():
        return [False] * len(targets), {
            "available": False,
            "absent_proposal_ids": [],
            "eligible_target_count": 0,
        }
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    attempts = ledger.get("stages", {}).get("target_to_clean", {}).get("attempts", [])
    absent_ids = {
        int(row["proposal_id"])
        for row in attempts
        if "object_absent" in row.get("tracker_rejection_reasons", [])
    }
    flags = [
        any(int(proposal_id) in absent_ids for proposal_id in target.proposal_ids)
        for target in targets
    ]
    return flags, {
        "available": True,
        "absent_proposal_ids": sorted(absent_ids),
        "eligible_target_count": int(sum(flags)),
    }


def _inference_pair(
    pair_id: str,
    split: dict[str, Any],
    config: dict[str, Any],
    record: dict[str, Any],
    baseline_record: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    settings = SlotSettings(**config["slot_inconsistency"])
    parent = Path(record["artifacts"]); artifact = Path(record["parent_artifacts"])
    proposal_cache = _resolve(split["proposal_cache_parent"]) / "pairs" / pair_id / "proposal_cache"
    tracking = _resolve(split["moved_tracking_root"]) / "pairs" / pair_id / "tracking_cache"
    source_changed, _, _ = recover_changed_candidates(parent, proposal_cache)
    metadata = json.loads((tracking / "metadata.json").read_text(encoding="utf-8"))
    forward, _, metadata = load_tracking_cache(tracking, input_sha256=metadata["input_sha256"])
    consolidation = ConsolidationSettings(**config["consolidation"])
    sources = consolidate_hypotheses(source_changed, forward, metadata["source_to_target"], settings=consolidation)
    target_inventory = proposals_to_objects(load_proposal_cache(proposal_cache / "target.npz"))
    neutral_rows = [{"rejection_reasons": []} for _ in target_inventory]
    targets = consolidate_hypotheses(target_inventory, [None] * len(target_inventory), neutral_rows, settings=consolidation)
    source_sam, target_sam = load_feature_maps(_resolve(split["feature_cache_root"]), pair_id)
    source_dino, target_dino = load_feature_maps(_resolve(split["dino_feature_cache_root"]), pair_id)
    inputs = load_cached_inputs(artifact)
    matches = match_object_slots(sources, targets, settings=settings)
    decisions = []
    for match in matches:
        source, target = sources[match.source_index], targets[match.target_index]
        retention = aligned_patch_retention(source_dino, target_dino, source.mask, target.mask, settings=settings)
        sam = descriptor_cosine(source_sam, source.mask, target_sam, target.mask)
        dino = descriptor_cosine(source_dino, source.mask, target_dino, target.mask)
        color = color_intersection(inputs.source_render, source.mask, inputs.target_image, target.mask)
        target_geometry = mask_geometry(target.mask)
        support_surface = support_surface_evidence(target_geometry, settings=settings)
        elsewhere = find_identity_elsewhere(
            match.source_index, match.target_index, sources, targets,
            source_sam, target_sam, source_dino, target_dino, settings=settings,
        )
        decision = decide_slot_replacement(
            match, retention, sam_cosine=sam, dino_cosine=dino,
            color_similarity=color, identity_elsewhere=bool(elsewhere["found"]),
            target_support_surface=bool(support_surface["is_support_surface"]),
            settings=settings,
        )
        decisions.append({
            "source_index": match.source_index, "target_index": match.target_index,
            "source_proposal_ids": list(source.proposal_ids), "target_proposal_ids": list(target.proposal_ids),
            "target_area_pixels": target.area_pixels, "match": match.summary(),
            "patch_retention": retention.summary(), "sam_cosine": sam, "dino_cosine": dino,
            "color_intersection": color, "identity_elsewhere": elsewhere,
            "target_geometry": target_geometry.summary(),
            "support_surface": support_surface,
            "decision": decision.summary(),
        })
    # Multiple source fragments can select the same target hypothesis. Keep the
    # strongest replacement evidence once, then emit the target mask itself.
    promoted_by_target: dict[int, dict[str, Any]] = {}
    for row in decisions:
        if row["decision"]["verdict"] != "replaced":
            continue
        score = float(row["patch_retention"]["lost_fraction"] or 0.0) + 0.25 * len(row["decision"]["mismatch_signals"]) + row["match"]["score"]
        previous = promoted_by_target.get(row["target_index"])
        if previous is None or score > previous["promotion_score"]:
            promoted_by_target[row["target_index"]] = {**row, "promotion_score": score}
    baseline = np.asarray(Image.open(baseline_record["path"]), np.uint8)
    native_shape = sources[0].mask.shape if sources else inputs.target_image.shape[:2]
    promoted_native = np.zeros(native_shape, bool)
    candidate_rows = [promoted_by_target[index] for index in sorted(promoted_by_target)]
    promoted_rows = []
    plausibility_rejected_rows = []
    cleanup_rows = []
    trusted_target_rows = []
    for row in candidate_rows:
        row["cleanup"] = replacement_cleanup_evidence(
            SlotMatch(**row["match"]),
            len(row["decision"]["mismatch_signals"]),
            baseline,
            targets[row["target_index"]].mask,
            settings=settings,
        )
        row["plausibility"] = replacement_object_plausibility(
            SlotMatch(**row["match"]),
            MaskGeometry(**row["target_geometry"]),
            cleanup_eligible=bool(row["cleanup"]["eligible"]),
            mismatch_signal_count=len(row["decision"]["mismatch_signals"]),
            added_removed_fraction=float(row["cleanup"]["added_removed_fraction"]),
            settings=settings,
        )
        if not row["plausibility"]["eligible"]:
            plausibility_rejected_rows.append(row)
            continue
        promoted_rows.append(row)
        if row["cleanup"]["source_cleanup_eligible"]:
            cleanup_rows.append(row)
        if row["cleanup"]["eligible"]:
            trusted_target_rows.append(row)
    cleanup_sources = [sources[row["source_index"]].mask for row in cleanup_rows]
    force_cleanup_rows = [
        row
        for row in cleanup_rows
        if row["cleanup"]["one_to_one_geometry"]
        and len(row["decision"]["mismatch_signals"])
        >= settings.minimum_cleanup_mismatch_signals
    ]
    force_cleanup_sources = [
        sources[row["source_index"]].mask for row in force_cleanup_rows
    ]
    force_cleanup_targets = [
        targets[row["target_index"]].mask for row in force_cleanup_rows
    ]
    trusted_targets = [targets[row["target_index"]].mask for row in trusted_target_rows]
    selected_target_indices = {row["target_index"] for row in promoted_rows}
    companion_rows: list[dict[str, Any]] = []
    reconstruction = load_reconstruction(artifact / "reconstruction.npz")
    for anchor_row in promoted_rows:
        anchor_index = anchor_row["target_index"]
        anchor_mask = targets[anchor_index].mask
        for candidate_index, candidate_target in enumerate(targets):
            if candidate_index in selected_target_indices:
                continue
            candidate_mask = candidate_target.mask
            gap = float(ndimage.distance_transform_edt(~anchor_mask)[candidate_mask].min())
            area_ratio = int(candidate_mask.sum()) / max(int(anchor_mask.sum()), 1)
            if (
                gap > settings.companion_maximum_gap_pixels
                or not settings.companion_minimum_area_ratio <= area_ratio <= settings.companion_maximum_area_ratio
                or mask_geometry(candidate_mask).compactness < settings.plausible_object_minimum_compactness
            ):
                continue
            evidence = replacement_companion_evidence(
                anchor_mask,
                candidate_mask,
                reconstruction.depths[1],
                sam_cosine=descriptor_cosine(target_sam, anchor_mask, target_sam, candidate_mask),
                dino_cosine=descriptor_cosine(target_dino, anchor_mask, target_dino, candidate_mask),
                color_intersection=color_intersection(
                    inputs.target_image, anchor_mask, inputs.target_image, candidate_mask
                ),
                settings=settings,
            )
            if not evidence["eligible"]:
                continue
            selected_target_indices.add(candidate_index)
            companion_rows.append({
                "anchor_target_index": anchor_index,
                "target_index": candidate_index,
                "target_proposal_ids": list(candidate_target.proposal_ids),
                "evidence": evidence,
            })
    companion_targets = [targets[row["target_index"]].mask for row in companion_rows]
    trusted_targets.extend(companion_targets)
    promoted_targets = [targets[row["target_index"]].mask for row in promoted_rows] + companion_targets
    target_change_flags, target_change_summary = _target_change_evidence(parent, targets)
    consensus_baseline, consensus_changed, consensus_summary = arbitrate_target_object_classes(
        baseline,
        [target.mask for target in targets],
        settings=settings,
        object_change_evidence=target_change_flags,
    )
    for row, target in zip(consensus_summary["objects"], targets, strict=True):
        row["proposal_ids"] = list(target.proposal_ids)
    consensus_summary["target_change_evidence"] = target_change_summary
    removal_fragment_rows = [
        row
        for row in plausibility_rejected_rows
        if row["cleanup"]["added_removed_fraction"]
        < settings.weak_identity_minimum_added_removed_fraction
    ]
    consensus_baseline, removal_fragment_changed = relabel_changed_target_fragments(
        consensus_baseline,
        [sources[row["source_index"]].mask for row in removal_fragment_rows],
        label=2,
    )
    replacement_ownership, protected_removed, depth_ownership = _replacement_depth_ownership(
        artifact,
        consensus_baseline,
        promoted_targets,
        force_cleanup_targets,
        settings,
    )
    candidate_native = np.zeros(native_shape, bool)
    for row in candidate_rows:
        candidate_native |= targets[row["target_index"]].mask
    for target_mask in promoted_targets:
        promoted_native |= target_mask
    promoted_mask = np.asarray(
        Image.fromarray(promoted_native.astype(np.uint8) * 255).resize(
            baseline.shape[::-1], Image.Resampling.NEAREST
        ),
        np.uint8,
    ) > 0
    full_labels, _, full_dropped_removed = rasterize_replacement_labels(
        consensus_baseline,
        cleanup_sources,
        promoted_targets,
        promote_only_changed=False,
        replacement_ownership_mask=replacement_ownership,
        protected_removed_mask=protected_removed,
        force_cleanup_source_masks=force_cleanup_sources,
        expand_connected_removed=True,
    )
    guarded_labels, guarded_mask, dropped_removed = rasterize_replacement_labels(
        consensus_baseline,
        cleanup_sources,
        promoted_targets,
        promote_only_changed=True,
        trusted_target_masks=trusted_targets,
        replacement_ownership_mask=replacement_ownership,
        protected_removed_mask=protected_removed,
        force_cleanup_source_masks=force_cleanup_sources,
        expand_connected_removed=True,
    )
    floor_suppression, floor_summary = _floor_suppression(
        artifact,
        baseline,
        settings,
    )
    full_floor_suppression = floor_suppression & (full_labels == 1)
    guarded_floor_suppression = floor_suppression & (guarded_labels == 1)
    full_labels[full_floor_suppression] = 0
    guarded_labels[guarded_floor_suppression] = 0
    output.mkdir(parents=True, exist_ok=True)
    save_image(output / "baseline_labels.png", baseline)
    save_image(output / "consensus_labels.png", consensus_baseline)
    save_image(output / "labels_full_mask.png", full_labels)
    save_image(output / "labels_guarded.png", guarded_labels)
    save_image(output / "labels.png", guarded_labels)
    save_image(output / "target.png", inputs.target_image)
    save_image(output / "source_render.png", inputs.source_render)
    source_slots = np.zeros(native_shape, bool)
    for source in sources:
        source_slots |= source.mask
    save_image(output / "source_slots.png", _overlay(inputs.source_render, source_slots, color=(30, 200, 235)))
    save_image(output / "replacement_candidate_overlay.png", _overlay(inputs.target_image, candidate_native, color=(255, 165, 20)))
    save_image(output / "replacement_overlay.png", _overlay(inputs.target_image, promoted_native))
    ownership_native = np.asarray(
        Image.fromarray(replacement_ownership.astype(np.uint8)).resize(
            promoted_native.shape[::-1], Image.Resampling.NEAREST
        ),
        bool,
    )
    save_image(output / "replacement_depth_owned_overlay.png", _overlay(inputs.target_image, promoted_native & ownership_native))
    ownership_color = np.zeros((*baseline.shape, 3), np.uint8)
    ownership_color[promoted_mask & replacement_ownership] = (30, 210, 80)
    ownership_color[promoted_mask & ~replacement_ownership] = (235, 70, 50)
    save_image(output / "depth_ownership_color.png", ownership_color)
    floor_native = np.asarray(
        Image.fromarray(floor_suppression.astype(np.uint8)).resize(
            inputs.target_image.shape[:2][::-1], Image.Resampling.NEAREST
        ),
        bool,
    )
    save_image(output / "floor_suppression_overlay.png", _overlay(inputs.target_image, floor_native, color=(255, 210, 30)))
    save_image(output / "baseline_labels_color.png", PALETTE[np.clip(baseline, 0, 5)])
    save_image(output / "consensus_labels_color.png", PALETTE[np.clip(consensus_baseline, 0, 5)])
    save_image(output / "labels_full_mask_color.png", PALETTE[np.clip(full_labels, 0, 5)])
    save_image(output / "labels_color.png", PALETTE[np.clip(guarded_labels, 0, 5)])
    inference = {
        "pair_id": pair_id, "ground_truth_used": False,
        "source_hypotheses": len(sources), "target_inventory_proposals": len(target_inventory), "target_hypotheses": len(targets),
        "slot_matches": len(matches), "decisions": decisions,
        "replacement_candidates": candidate_rows,
        "replacement_candidate_count": len(candidate_rows),
        "plausibility_rejected": plausibility_rejected_rows,
        "plausibility_rejected_count": len(plausibility_rejected_rows),
        "promoted": promoted_rows, "promoted_target_count": len(promoted_rows),
        "replacement_companions": companion_rows,
        "replacement_companion_count": len(companion_rows),
        "rasterized_target_count": len(promoted_rows) + len(companion_rows),
        "object_class_consensus": consensus_summary,
        "object_consensus_changed_pixels": int(consensus_changed.sum()),
        "removal_fragment_changed_pixels": int(removal_fragment_changed.sum()),
        "removal_fragment_targets": [
            {
                "source_index": row["source_index"],
                "target_index": row["target_index"],
                "target_proposal_ids": row["target_proposal_ids"],
                "reason": "rejected_target_lacks_new_object_support",
            }
            for row in removal_fragment_rows
        ],
        "cleanup_eligible_target_count": len(trusted_target_rows),
        "source_cleanup_eligible_target_count": len(cleanup_rows),
        "force_cleanup_target_count": len(force_cleanup_rows),
        "force_cleanup_targets": [
            {
                "source_index": row["source_index"],
                "target_index": row["target_index"],
                "reason": "three_mismatch_one_to_one_obsolete_source",
            }
            for row in force_cleanup_rows
        ],
        "trusted_full_target_count": len(trusted_targets),
        "promoted_pixels": int(promoted_mask.sum()), "guarded_promoted_pixels": int(guarded_mask.sum()),
        "dropped_removed_pixels": int(dropped_removed.sum()),
        "full_dropped_removed_pixels": int(full_dropped_removed.sum()),
        "depth_ownership": depth_ownership,
        "floor_suppression": floor_summary,
        "full_floor_suppressed_pixels": int(full_floor_suppression.sum()),
        "guarded_floor_suppressed_pixels": int(guarded_floor_suppression.sum()),
        "baseline_sha256": _sha256_array(baseline),
        "baseline_source": {
            "evaluation": baseline_record["evaluation"],
            "variant": baseline_record["variant"],
            "relative_path": baseline_record["relative_path"],
            "file_sha256": baseline_record["file_sha256"],
            "array_sha256": baseline_record["array_sha256"],
        },
        "evidence_parent_artifacts": str(parent),
        "full_prediction_sha256": _sha256_array(full_labels),
        "guarded_prediction_sha256": _sha256_array(guarded_labels),
        "prediction_sha256": _sha256_array(guarded_labels),
        "full_prediction_changed_pixels": int(np.sum(full_labels != baseline)),
        "prediction_changed_pixels": int(np.sum(guarded_labels != baseline)),
    }
    save_json(output / "inference.json", inference)
    return {
        "pair_id": pair_id,
        "output": output,
        "inference": inference,
        "labels": guarded_labels,
        "full_labels": full_labels,
        "baseline": baseline,
        "sources": sources,
        "targets": targets,
        "promoted_mask": promoted_mask,
    }


def _metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        "multiclass_miou": candidate["multiclass_miou"] - baseline["multiclass_miou"],
        "binary_miou": candidate["binary_miou"] - baseline["binary_miou"],
        "replaced_iou": candidate["multiclass"]["replaced"]["iou"] - baseline["multiclass"]["replaced"]["iou"],
        "replaced_precision": candidate["multiclass"]["replaced"]["precision"] - baseline["multiclass"]["replaced"]["precision"],
        "replaced_recall": candidate["multiclass"]["replaced"]["recall"] - baseline["multiclass"]["replaced"]["recall"],
    }


def _raster_baseline_records(
    root: Path,
    variant: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load and validate the frozen final-R4 prediction inventory."""
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "predictions_frozen.json").read_text(encoding="utf-8"))
    selection = [str(value) for value in report["selection"]]
    if report.get("failures") or selection != [str(value) for value in frozen["selection"]]:
        raise RuntimeError(f"incomplete or mismatched raster baseline: {root}")
    if report.get("provenance", {}).get("ground_truth_used_in_inference") is not False:
        raise RuntimeError(f"raster baseline used ground truth during inference: {root}")
    if report.get("protocol", {}).get("predictions_frozen_before_ground_truth") is not True:
        raise RuntimeError(f"raster baseline predictions were not frozen before scoring: {root}")

    report_rows = {str(row["id"]): row for row in report["pairs"]}
    frozen_rows = {str(row["id"]): row for row in frozen["pairs"]}
    if set(report_rows) != set(selection) or set(frozen_rows) != set(selection):
        raise RuntimeError(f"raster baseline pair inventory differs from selection: {root}")

    records: dict[str, dict[str, Any]] = {}
    for pair_id in selection:
        prediction = report_rows[pair_id]["predictions"].get(variant)
        frozen_prediction = frozen_rows[pair_id]["predictions"].get(variant)
        if prediction is None or frozen_prediction != prediction:
            raise RuntimeError(f"missing or unfrozen {variant} prediction for {pair_id}")
        path = (root / prediction["relative_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RuntimeError(f"invalid raster baseline path for {pair_id}: {path}")
        if _sha256_file(path) != prediction["file_sha256"]:
            raise RuntimeError(f"raster baseline file changed after freeze: {path}")
        records[pair_id] = {
            **prediction,
            "path": path,
            "evaluation": str(root),
            "variant": variant,
        }
    return selection, records


def _evaluate_split(rows: list[dict[str, Any]], targets: dict[str, Path], output: Path) -> dict[str, Any]:
    baseline_metric, full_metric, guarded_metric = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
    evaluated = []
    for row in rows:
        truth = _resize_labels(normalize_target(targets[row["pair_id"]]), row["labels"].shape)
        baseline_metric.add(row["baseline"], truth); full_metric.add(row["full_labels"], truth); guarded_metric.add(row["labels"], truth)
        base_pair, full_pair, guarded_pair = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
        base_pair.add(row["baseline"], truth); full_pair.add(row["full_labels"], truth); guarded_pair.add(row["labels"], truth)
        base_scores, full_scores, guarded_scores = base_pair.compute(), full_pair.compute(), guarded_pair.compute()
        candidate_truth = []
        native_shape = row["targets"][0].mask.shape if row["targets"] else truth.shape
        native_truth = _resize_labels(truth, native_shape)
        native_baseline = _resize_labels(row["baseline"], native_shape)
        native_full = _resize_labels(row["full_labels"], native_shape)
        native_guarded = _resize_labels(row["labels"], native_shape)
        source_render = np.asarray(Image.open(row["output"] / "source_render.png").convert("RGB"), np.uint8)
        target_image = np.asarray(Image.open(row["output"] / "target.png").convert("RGB"), np.uint8)
        example_root = row["output"] / "examples"
        example_root.mkdir(parents=True, exist_ok=True)
        for candidate in row["inference"]["replacement_candidates"]:
            native_mask = row["targets"][candidate["target_index"]].mask
            mask = np.asarray(Image.fromarray(native_mask.astype(np.uint8) * 255).resize(truth.shape[::-1], Image.Resampling.NEAREST), np.uint8) > 0
            source_mask = row["sources"][candidate["source_index"]].mask
            box = _crop_box(source_mask | native_mask)
            stem = f"target_{candidate['target_index']}_source_{candidate['source_index']}"
            images = {
                "source": f"examples/{stem}_source.png",
                "target": f"examples/{stem}_target.png",
                "full": f"examples/{stem}_full.png",
                "guarded": f"examples/{stem}_guarded.png",
                "truth": f"examples/{stem}_truth.png",
            }
            _save_crop(row["output"] / images["source"], _overlay(source_render, source_mask, color=(30, 200, 235)), box)
            _save_crop(row["output"] / images["target"], _overlay(target_image, native_mask), box)
            _save_crop(row["output"] / images["full"], PALETTE[np.clip(native_full, 0, 5)], box)
            _save_crop(row["output"] / images["guarded"], PALETTE[np.clip(native_guarded, 0, 5)], box)
            _save_crop(row["output"] / images["truth"], PALETTE[np.clip(native_truth, 0, 5)], box)
            candidate_truth.append({
                "target_index": candidate["target_index"],
                "source_index": candidate["source_index"],
                "rasterized": bool(candidate["plausibility"]["eligible"]),
                "plausibility": candidate["plausibility"],
                **_candidate_truth(mask, truth),
                "images": images,
                "indicators": {
                    "lost_fraction": candidate["patch_retention"]["lost_fraction"],
                    "sam_cosine": candidate["sam_cosine"],
                    "dino_cosine": candidate["dino_cosine"],
                    "color_intersection": candidate["color_intersection"],
                    "aligned_iou": candidate["match"]["aligned_iou"],
                    "track_iou": candidate["match"]["track_iou"],
                    "mismatch_signals": candidate["decision"]["mismatch_signals"],
                    "identity_elsewhere": candidate["identity_elsewhere"]["found"],
                },
            })
        save_image(row["output"] / "ground_truth_color.png", PALETTE[np.clip(truth, 0, 5)])
        error = np.zeros((*truth.shape, 3), np.uint8); error[(row["labels"] == truth)] = (35, 70, 45); error[(row["labels"] != truth)] = (235, 65, 55)
        save_image(row["output"] / "error.png", error)
        evaluated.append({
            "id": row["pair_id"], "inference": row["inference"],
            "baseline_metrics": base_scores, "full_mask_metrics": full_scores,
            "guarded_metrics": guarded_scores,
            "full_mask_delta": _metric_delta(full_scores, base_scores),
            "delta": _metric_delta(guarded_scores, base_scores),
            "replacement_candidate_truth": candidate_truth,
            "promoted_truth": [item for item in candidate_truth if item["rasterized"]],
        })
    baseline, full, guarded = baseline_metric.compute(), full_metric.compute(), guarded_metric.compute()
    groups: dict[str, list[dict[str, Any]]] = {"replaced-majority": [], "other-majority": []}
    for pair in evaluated:
        for item in pair["promoted_truth"]:
            groups["replaced-majority" if item["dominant"] == "replaced" else "other-majority"].append(item)
    indicator_statistics = {}
    for name, items in groups.items():
        signal_counts = Counter(
            signal for item in items for signal in item["indicators"]["mismatch_signals"]
        )
        indicator_statistics[name] = {
            "count": len(items),
            "mean_purity": float(np.mean([item["purity"] for item in items])) if items else None,
            "mean_lost_fraction": float(np.mean([item["indicators"]["lost_fraction"] for item in items])) if items else None,
            "mean_sam_cosine": float(np.mean([item["indicators"]["sam_cosine"] for item in items])) if items else None,
            "mean_dino_cosine": float(np.mean([item["indicators"]["dino_cosine"] for item in items])) if items else None,
            "mean_color_intersection": float(np.mean([item["indicators"]["color_intersection"] for item in items])) if items else None,
            "signal_counts": dict(signal_counts),
        }
    return {
        "baseline": baseline,
        "full_mask_candidate": full,
        "full_mask_delta": _metric_delta(full, baseline),
        "candidate": guarded,
        "guarded_candidate": guarded,
        "delta": _metric_delta(guarded, baseline),
        "guarded_delta": _metric_delta(guarded, baseline),
        "indicator_statistics": indicator_statistics,
        "pairs": evaluated,
    }


def _aggregate_pending(pending: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_metric = MetricAccumulator()
    full_metric = MetricAccumulator()
    guarded_metric = MetricAccumulator()
    pair_count = 0
    for value in pending.values():
        for row in value["rows"]:
            truth = _resize_labels(
                normalize_target(value["targets"][row["pair_id"]]),
                row["labels"].shape,
            )
            baseline_metric.add(row["baseline"], truth)
            full_metric.add(row["full_labels"], truth)
            guarded_metric.add(row["labels"], truth)
            pair_count += 1
    baseline = baseline_metric.compute()
    full = full_metric.compute()
    guarded = guarded_metric.compute()
    return {
        "pair_count": pair_count,
        "baseline": baseline,
        "full_mask_candidate": full,
        "full_mask_delta": _metric_delta(full, baseline),
        "candidate": guarded,
        "guarded_candidate": guarded,
        "delta": _metric_delta(guarded, baseline),
        "guarded_delta": _metric_delta(guarded, baseline),
    }


def _html(output: Path, report: dict[str, Any]) -> None:
    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    baseline_variants = sorted(
        {
            str(value["variant"])
            for value in report.get("raster_baseline", {}).values()
        }
    )
    baseline_variant = ", ".join(baseline_variants) or "unspecified"
    baseline_heading = f"Frozen baseline ({baseline_variant})"
    overall_html = ""
    if "overall" in report:
        overall = report["overall"]
        base = overall["baseline"]
        full = overall["full_mask_candidate"]
        guarded = overall["guarded_candidate"]
        delta = overall["guarded_delta"]
        comparison_rows = []
        for label, metrics in (
            (baseline_heading, base),
            ("Object-consistent full target masks", full),
            ("Object-consistent guarded output", guarded),
        ):
            comparison_rows.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td>{100 * metrics['binary']['changed']['iou']:.2f}</td>"
                f"<td>{100 * metrics['binary']['unchanged']['iou']:.2f}</td>"
                f"<td>{100 * metrics['binary_miou']:.2f}</td>"
                f"<td>{100 * metrics['multiclass']['added']['iou']:.2f}</td>"
                f"<td>{100 * metrics['multiclass']['removed']['iou']:.2f}</td>"
                f"<td>{100 * metrics['multiclass']['moved']['iou']:.2f}</td>"
                f"<td>{100 * metrics['multiclass']['replaced']['iou']:.2f}</td>"
                f"<td>{100 * metrics['multiclass_miou']:.2f}</td></tr>"
            )
        class_sections = []
        for family, title in (("binary", "Binary classes"), ("multiclass", "Multiclass classes")):
            rows = []
            for class_name, candidate_values in guarded[family].items():
                baseline_values = base[family][class_name]
                full_values = full[family][class_name]
                rows.append(
                    f"<tr><td>{html.escape(class_name)}</td>"
                    f"<td>{baseline_values['iou']:.6f}</td><td>{full_values['iou']:.6f}</td>"
                    f"<td>{candidate_values['iou']:.6f}</td>"
                    f"<td>{candidate_values['iou'] - baseline_values['iou']:+.6f}</td>"
                    f"<td>{candidate_values['precision']:.6f}</td>"
                    f"<td>{candidate_values['recall']:.6f}</td>"
                    f"<td>{candidate_values['f1']:.6f}</td><td>{candidate_values['support']}</td></tr>"
                )
            class_sections.append(
                f"<h3>{title}</h3><table><tr><th>class</th><th>baseline IoU</th>"
                f"<th>full-mask IoU</th><th>guarded IoU</th><th>guarded ΔIoU</th>"
                f"<th>guarded precision</th><th>guarded recall</th><th>guarded F1</th>"
                f"<th>support</th></tr>{''.join(rows)}</table>"
            )
        overall_html = f"""
        <section class='method'><h2>Combined {overall['pair_count']}-pair result</h2>
        <p>Pixel-count aggregate across all evaluated splits. The frozen raster baseline is
        <code>{html.escape(baseline_variant)}</code>. Tracking artifacts supply evidence only.</p>
        <h3>Table-3-style comparison (%)</h3>
        <table class='summary'><tr><th>method</th><th>changed</th><th>unchanged</th>
        <th>binary mIoU</th><th>added</th><th>removed</th><th>moved</th>
        <th>replaced</th><th>multi mIoU</th></tr>{''.join(comparison_rows)}</table>
        <p>The full-mask ablation has the higher binary and multiclass mIoU. Both variants
        reject incoherent replacement fragments, erase geometrically matched old REMOVED
        footprints, arbitrate inconsistent classes within independently changed target
        objects, and suppress floor-dominant
        ADDED components. Accepted SAM objects receive one class.
        WARPED is not a scored ChangeSim ground-truth class.</p>
        <table class='summary'><tr><th>metric</th><th>baseline</th><th>full target mask</th><th>guarded/trusted output</th><th>guarded delta</th></tr>
        <tr><td>multiclass mIoU</td><td>{base['multiclass_miou']:.6f}</td><td>{full['multiclass_miou']:.6f}</td><td>{guarded['multiclass_miou']:.6f}</td><td>{delta['multiclass_miou']:+.6f}</td></tr>
        <tr><td>binary mIoU</td><td>{base['binary_miou']:.6f}</td><td>{full['binary_miou']:.6f}</td><td>{guarded['binary_miou']:.6f}</td><td>{delta['binary_miou']:+.6f}</td></tr></table>
        {''.join(class_sections)}</section>"""
    sections = []
    for split_name, split in report["splits"].items():
        split_baseline_variant = str(
            report.get("raster_baseline", {}).get(split_name, {}).get("variant", "unspecified")
        )
        base, full, candidate, delta = split["baseline"], split["full_mask_candidate"], split["guarded_candidate"], split["guarded_delta"]
        cards = []
        for pair in split["pairs"]:
            inf = pair["inference"]; root = Path(split_name) / "pairs" / pair["id"]
            candidate_by_target = {
                item["target_index"]: item for item in inf["replacement_candidates"]
            }
            cases = []
            for item in pair["replacement_candidate_truth"]:
                indicators = item["indicators"]
                images = item["images"]
                candidate_row = candidate_by_target[item["target_index"]]
                cleanup = candidate_row.get("cleanup", {})
                plausibility = candidate_row.get("plausibility", {})
                cleanup_text = "allowed" if cleanup.get("source_cleanup_eligible") else "blocked"
                cleanup_reasons = ", ".join(cleanup.get("source_cleanup_reasons", [])) or "identity and slot geometry agree"
                plausibility_text = "accepted as one object" if plausibility.get("eligible") else "rejected before rasterization"
                plausibility_reasons = ", ".join(plausibility.get("reasons", [])) or "coherent target object"
                cases.append(f"""
                <section class='case'>
                  <h4>Source {item['source_index']} → target SAM hypothesis {item['target_index']}</h4>
                  <p class='verdict'>GT majority: <b>{html.escape(item['dominant'])}</b> ({item['purity']:.1%} purity) · {html.escape(str(item['distribution']))}</p>
                  <p>Object plausibility: <b>{html.escape(plausibility_text)}</b> · {html.escape(plausibility_reasons)}</p>
                  <p>Old-mask cleanup: <b>{cleanup_text}</b> · {html.escape(cleanup_reasons)}</p>
                  <div class='zoom-images'>
                    <figure><img src='{root}/{images['source']}'><figcaption>Old appearance (cyan source slot)</figcaption></figure>
                    <figure><img src='{root}/{images['target']}'><figcaption>New appearance (magenta target SAM)</figcaption></figure>
                    <figure><img src='{root}/{images['full']}'><figcaption>Object-consistent full output</figcaption></figure>
                    <figure><img src='{root}/{images['guarded']}'><figcaption>Object-consistent guarded output</figcaption></figure>
                    <figure><img src='{root}/{images['truth']}'><figcaption>Ground truth</figcaption></figure>
                  </div>
                  <table><tr><th>lost old identity ↑</th><th>SAM cosine ↓</th><th>DINO cosine ↓</th><th>color overlap ↓</th><th>aligned IoU</th><th>track IoU</th><th>triggered mismatches</th></tr>
                  <tr><td>{number(indicators['lost_fraction'])}</td><td>{number(indicators['sam_cosine'])}</td><td>{number(indicators['dino_cosine'])}</td><td>{number(indicators['color_intersection'])}</td><td>{number(indicators['aligned_iou'])}</td><td>{number(indicators['track_iou'])}</td><td>{html.escape(', '.join(indicators['mismatch_signals']))}</td></tr></table>
                </section>""")
            if not cases:
                cases.append("<p class='quiet'>No target hypothesis reached the object-plausibility check.</p>")
            depth = inf["depth_ownership"]
            floor = inf["floor_suppression"]
            cards.append(f"""<article><h3>{html.escape(pair['id'])}</h3><p>slot matches {inf['slot_matches']} · replacement candidates {inf['replacement_candidate_count']} · plausible objects {inf['promoted_target_count']} · companions {inf['replacement_companion_count']} · rejected fragments {inf['plausibility_rejected_count']} · target-object arbitration rewrites {inf['object_consensus_changed_pixels']} · suppressed floor pixels {inf['guarded_floor_suppressed_pixels']} · dropped old REMOVED pixels {inf['dropped_removed_pixels']} · full/guarded changed pixels {inf['full_prediction_changed_pixels']}/{inf['prediction_changed_pixels']} · guarded ΔmIoU {pair['delta']['multiclass_miou']:+.5f}</p><div class='images'><figure><img src='{root}/target.png'><figcaption>Target image</figcaption></figure><figure><img src='{root}/replacement_candidate_overlay.png'><figcaption>Raw identity-mismatch candidates</figcaption></figure><figure><img src='{root}/replacement_overlay.png'><figcaption>Plausible objects + companions</figcaption></figure><figure><img src='{root}/replacement_depth_owned_overlay.png'><figcaption>Final object ownership</figcaption></figure><figure><img src='{root}/floor_suppression_overlay.png'><figcaption>Implausible floor suppression</figcaption></figure><figure><img src='{root}/baseline_labels_color.png'><figcaption>Frozen raster baseline</figcaption></figure><figure><img src='{root}/consensus_labels_color.png'><figcaption>Target-object class arbitration</figcaption></figure><figure><img src='{root}/labels_full_mask_color.png'><figcaption>Object-consistent full output</figcaption></figure><figure><img src='{root}/labels_color.png'><figcaption>Object-consistent guarded output</figcaption></figure><figure><img src='{root}/ground_truth_color.png'><figcaption>Ground truth</figcaption></figure></div>{''.join(cases)}<details><summary>All slot decisions and raw metrics</summary><pre>{html.escape(json.dumps({'decisions': inf['decisions'], 'replacement_candidates': inf['replacement_candidates'], 'replacement_companions': inf['replacement_companions'], 'object_class_consensus': inf['object_class_consensus'], 'depth_ownership': depth, 'floor_suppression': floor}, indent=2))}</pre></details></article>""")
        representative = max(
            split["pairs"],
            key=lambda pair: (
                pair["inference"]["dropped_removed_pixels"],
                pair["inference"]["promoted_target_count"],
            ),
        )
        representative_root = Path(split_name) / "pairs" / representative["id"]
        statistic_rows = []
        for group_name, values in split["indicator_statistics"].items():
            statistic_rows.append(
                f"<tr><td>{html.escape(group_name)}</td><td>{values['count']}</td><td>{number(values['mean_purity'])}</td>"
                f"<td>{number(values['mean_lost_fraction'])}</td><td>{number(values['mean_sam_cosine'])}</td>"
                f"<td>{number(values['mean_dino_cosine'])}</td><td>{number(values['mean_color_intersection'])}</td>"
                f"<td>{html.escape(str(values['signal_counts']))}</td></tr>"
            )
        sections.append(f"""
        <h2>{split_name}</h2>
        <p>Identity mismatch first produces a candidate, not a raster edit. A coherent-object gate rejects weak or fragmented target masks. Accepted SAM objects receive one REPLACED class across their visible mask; an adjacent companion is included only when proximity, depth, SAM, DINO, and color all agree. When three identity mismatches and one-to-one slot geometry agree, the matched old REMOVED footprint is erased outside the new target object. Independently changed, coherent target objects are then arbitrated as a whole: ADDED+REMOVED means REPLACED, a REPLACED majority absorbs an interior MOVED fragment, and other classes require at least 75% consensus. This step relabels only existing change pixels and never expands binary support. Floor-dominant ADDED components are removed using a robust scene-floor fit.</p>
        <p>Frozen raster baseline: <code>{html.escape(split_baseline_variant)}</code>.</p>
        <table class='summary'><tr><th>metric</th><th>baseline</th><th>full target mask</th><th>guarded changed-pixel relabel</th><th>guarded delta</th></tr><tr><td>multiclass mIoU</td><td>{base['multiclass_miou']:.6f}</td><td>{full['multiclass_miou']:.6f}</td><td>{candidate['multiclass_miou']:.6f}</td><td>{delta['multiclass_miou']:+.6f}</td></tr><tr><td>binary mIoU</td><td>{base['binary_miou']:.6f}</td><td>{full['binary_miou']:.6f}</td><td>{candidate['binary_miou']:.6f}</td><td>{delta['binary_miou']:+.6f}</td></tr><tr><td>REPLACED IoU</td><td>{base['multiclass']['replaced']['iou']:.6f}</td><td>{full['multiclass']['replaced']['iou']:.6f}</td><td>{candidate['multiclass']['replaced']['iou']:.6f}</td><td>{delta['replaced_iou']:+.6f}</td></tr><tr><td>REPLACED precision</td><td>{base['multiclass']['replaced']['precision']:.6f}</td><td>{full['multiclass']['replaced']['precision']:.6f}</td><td>{candidate['multiclass']['replaced']['precision']:.6f}</td><td>{delta['replaced_precision']:+.6f}</td></tr><tr><td>REPLACED recall</td><td>{base['multiclass']['replaced']['recall']:.6f}</td><td>{full['multiclass']['replaced']['recall']:.6f}</td><td>{candidate['multiclass']['replaced']['recall']:.6f}</td><td>{delta['replaced_recall']:+.6f}</td></tr></table>
        <h3>Procedure shown on {html.escape(representative['id'])}</h3>
        <div class='pipeline'>
          <figure><span>1</span><img src='{representative_root}/source_render.png'><figcaption>Aligned source inference</figcaption></figure><b>→</b>
          <figure><span>2</span><img src='{representative_root}/source_slots.png'><figcaption>Consolidate old object slots</figcaption></figure><b>→</b>
          <figure><span>3</span><img src='{representative_root}/target.png'><figcaption>Look only at aligned target</figcaption></figure><b>→</b>
          <figure><span>4</span><img src='{representative_root}/replacement_candidate_overlay.png'><figcaption>Find identity-mismatch candidates</figcaption></figure><b>→</b>
          <figure><span>5</span><img src='{representative_root}/replacement_overlay.png'><figcaption>Keep coherent objects and companions</figcaption></figure><b>→</b>
          <figure><span>6</span><img src='{representative_root}/floor_suppression_overlay.png'><figcaption>Reject floor-dominant components</figcaption></figure><b>→</b>
          <figure><span>7</span><img src='{representative_root}/consensus_labels_color.png'><figcaption>Arbitrate classes inside each changed object</figcaption></figure><b>→</b>
          <figure><span>8</span><img src='{representative_root}/labels_color.png'><figcaption>Write one class per object pixel</figcaption></figure>
        </div>
        <h3>Indicator statistics for promoted hypotheses</h3>
        <table><tr><th>GT group (audit only)</th><th>count</th><th>mean GT purity</th><th>mean identity loss</th><th>mean SAM cosine</th><th>mean DINO cosine</th><th>mean color overlap</th><th>mismatch counts</th></tr>{''.join(statistic_rows)}</table>
        {''.join(cards)}""")
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Object-slot replacement IoU</title><style>
    body{{font:15px system-ui;max-width:1500px;margin:28px auto;background:#f3f6fa;color:#16202c;line-height:1.45}}article,.method{{background:white;border-radius:11px;padding:18px;margin:18px 0;box-shadow:0 2px 10px #20304014}}table{{border-collapse:collapse;width:100%;background:white;margin:10px 0 18px}}th,td{{padding:7px;border:1px solid #d8dfe8;text-align:left;vertical-align:top}}.summary th{{background:#203d5e;color:white}}.images{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}}.zoom-images{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;align-items:end}}figure{{margin:0}}img{{width:100%;display:block;background:#111;image-rendering:auto}}figcaption{{font-weight:600;padding-top:4px}}pre{{font-size:11px;white-space:pre-wrap}}.case{{border-top:3px solid #d9e1eb;margin-top:18px;padding-top:8px}}.verdict{{background:#f1f4f8;padding:7px}}.quiet{{color:#5b6977}}.pipeline{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr;gap:8px;align-items:center;background:#e7edf4;padding:12px;border-radius:9px}}.pipeline b{{font-size:24px;color:#48627e}}.pipeline span{{position:absolute;background:#203d5e;color:white;border-radius:50%;padding:3px 9px;margin:5px}}.legend span{{display:inline-block;padding:4px 9px;border-radius:5px;margin-right:6px;color:white}}@media(max-width:1000px){{.images,.zoom-images{{grid-template-columns:1fr 1fr}}.pipeline{{grid-template-columns:1fr}}.pipeline b{{display:none}}}}</style></head><body>
    <h1>Object-slot inconsistency replacement experiment</h1>
    <div class='method'><p><b>Question:</b> can replacement be found by looking for an old object whose identity disappears in the same aligned slot, without producing a new clean render?</p><p><b>Corrected baseline:</b> all label edits are applied on top of the frozen raster prediction <code>{html.escape(baseline_variant)}</code>. Tracking artifacts supply evidence only.</p><p><b>Object plausibility:</b> identity mismatch creates a candidate only. The candidate must be a compact object or a sufficiently aligned coherent fragment. When the target proposal is rejected and has no new-object support, the aligned source footprint—not the untrusted target fragment—is retained as REMOVED on existing changed pixels. Once accepted, a replacement's visible SAM mask receives one class. Adjacent masks join it only when distance, depth, SAM, DINO, and color independently agree.</p><p><b>Depth-authoritative ownership:</b> object unification may absorb ADDED/MOVED fragments but cannot normally override a closer REMOVED component. A replacement owns an overlapping red pixel only when target depth is closer. Source-front, tied, and unknown-depth REMOVED pixels are preserved. The one exception is explicitly identity-scoped: three identity mismatches plus one-to-one geometry prove that the paired source footprint is obsolete history, not a coexisting foreground object.</p><p><b>Replacement cleanup:</b> a strong identity-scoped pair removes only its own obsolete source footprint even if the historical source render is closer. Weak or asymmetric pairs remain depth-protected. Connected REMOVED components outside the direct overlap are deleted only when the same identity scope or overlap depth authorizes it; mask connectivity alone is insufficient.</p><p><b>Target-object arbitration:</b> every coherent SAM object is considered, not only a hand-picked failure or a rejected replacement candidate. At least one constituent proposal must independently fail target-to-clean tracking as <code>object_absent</code>. Within that object, ADDED+REMOVED is interpreted as REPLACED; a majority REPLACED label absorbs an interior MOVED fragment; otherwise a dominant class needs 75% support. Only pixels already marked changed are relabeled, so the binary mask cannot grow.</p><p><b>Floor:</b> large ADDED components are removed only when at least 70% of the component lies within 5 cm of a robust fitted floor plane.</p><p><b>Evaluation discipline:</b> predictions and SHA-256 hashes were frozen before ground truth was opened. Ground truth is used only below for scoring and failure analysis.</p><p class='legend'><span style='background:#1ed250'>ADDED</span><span style='background:#eb4632'>REMOVED</span><span style='background:#ffa514'>MOVED</span><span style='background:#1ea0e6'>WARPED</span><span style='background:#cd3ceb'>REPLACED</span></p></div>
    {overall_html}{''.join(sections)}</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = arguments(); config = load_config(args.config.resolve()); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    selected_splits = {item.strip() for item in args.splits.split(",") if item.strip()}; selected_pairs = None if args.pairs is None else {item.strip() for item in args.pairs.split(",") if item.strip()}
    pending: dict[str, dict[str, Any]] = {}; frozen = {"ground_truth_used": False, "splits": {}}
    for split_name, split in config["splits"].items():
        if split_name not in selected_splits: continue
        selection, records = parent_records(_resolve(split["parent_evaluation"])); targets = manifest_targets(_resolve(split["manifest"]))
        baseline_root = _resolve(split["raster_baseline_evaluation"])
        baseline_variant = str(split["raster_baseline_variant"])
        baseline_selection, baseline_records = _raster_baseline_records(baseline_root, baseline_variant)
        if selection != baseline_selection:
            raise RuntimeError(
                f"evidence and raster baseline selections differ for {split_name}: "
                f"{selection} != {baseline_selection}"
            )
        if selected_pairs is not None: selection = [item for item in selection if item in selected_pairs]
        rows = [
            _inference_pair(
                pair_id,
                split,
                config,
                records[pair_id],
                baseline_records[pair_id],
                output / split_name / "pairs" / pair_id,
            )
            for pair_id in selection
        ]
        pending[split_name] = {"rows": rows, "targets": targets}
        frozen["splits"][split_name] = {
            "baseline_evaluation": str(baseline_root),
            "baseline_variant": baseline_variant,
            "pairs": [
                {
                    "id": row["pair_id"],
                    "baseline_file_sha256": row["inference"]["baseline_source"]["file_sha256"],
                    "full_prediction_sha256": row["inference"]["full_prediction_sha256"],
                    "guarded_prediction_sha256": row["inference"]["guarded_prediction_sha256"],
                }
                for row in rows
            ],
        }
    save_json(output / "predictions_frozen.json", frozen)
    report = {
        "experiment_id": "slot_inconsistency_replacement_densegrid96",
        "prediction_edits": True,
        "ground_truth_after_prediction_freeze": True,
        "raster_baseline": {
            split_name: {
                "evaluation": value["baseline_evaluation"],
                "variant": value["baseline_variant"],
            }
            for split_name, value in frozen["splits"].items()
        },
        "splits": {},
    }
    for split_name, value in pending.items():
        report["splits"][split_name] = _evaluate_split(value["rows"], value["targets"], output / split_name)
    report["overall"] = _aggregate_pending(pending)
    save_json(output / "report.json", report); _html(output, report)
    print(json.dumps({name: value["delta"] for name, value in report["splits"].items()}, indent=2)); print(output / "index.html")


if __name__ == "__main__":
    main()
