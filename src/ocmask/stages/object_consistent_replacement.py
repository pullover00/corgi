"""Stage 11: the object-consistent replacement refinement.

This is the method's final, named stage: it refines the base pipeline's
per-pixel prediction (stages 1-10) by finding old objects whose identity
disappears in the same aligned image slot, gating candidates on whether they
are a plausible object, expanding to compatible neighboring fragments,
resolving depth-authoritative ownership against REMOVED pixels, cleaning up
obsolete source footprints, arbitrating class consensus across every
coherent target object, and suppressing floor-dominant ADDED components.

Every actual decision rule lives in :mod:`slot_inconsistency` and
:mod:`branch_b2`, already clean and unit-tested. This module is the
single-pair orchestration of those rules -- a faithful adaptation of
``scripts/run_slot_inconsistency_replacement_experiment.py``'s
``_inference_pair`` (and its three local helpers) from a disk-artifact,
multi-pair-loop shape to the in-memory objects a single ``run_pair`` call
already has on hand. The qualitative HTML/crop reporting that script also
built is not reproduced here; this module returns/saves only the prediction
rasters and the machine-readable decision trail (``inference.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from ..cache import load_reconstruction
from ..geometry import render_points
from ..ground_contact import detect_floor_plane, up_vector_from_world_to_camera
from ..io import save_image, save_json
from ..types import ObjectMask
from ..visualization import colorize
from .branch_b2 import ConsolidationSettings, ObjectHypothesis, consolidate_hypotheses
from .slot_inconsistency import (
    MaskGeometry,
    SlotMatch,
    SlotSettings,
    aligned_patch_retention,
    arbitrate_target_object_classes,
    color_intersection,
    decide_slot_replacement,
    descriptor_cosine,
    find_identity_elsewhere,
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


def _replacement_depth_ownership(
    reconstruction_artifact_dir: Path,
    baseline: np.ndarray,
    target_masks: list[np.ndarray],
    identity_override_target_masks: list[np.ndarray],
    settings: SlotSettings,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Which promoted-target pixels the depth-authoritative rule may own."""

    artifact = Path(reconstruction_artifact_dir)
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
    reconstruction_artifact_dir: Path,
    baseline: np.ndarray,
    settings: SlotSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Suppress floor-dominant ADDED components (a wall/floor plane misread)."""

    artifact = Path(reconstruction_artifact_dir)
    reconstruction = load_reconstruction(artifact / "reconstruction.npz")
    with np.load(artifact / "geometry.npz") as geometry:
        keep0 = np.asarray(geometry["keep0"], bool)
        keep1 = np.asarray(geometry["keep1"], bool)
    static_points = np.concatenate([reconstruction.points[0][keep0], reconstruction.points[1][keep1]])
    static_colors = np.concatenate([reconstruction.images[0][keep0], reconstruction.images[1][keep1]])
    up = up_vector_from_world_to_camera(reconstruction.world_to_camera[1])
    floor = detect_floor_plane(static_points, static_colors, up)
    if floor is None:
        return np.zeros(baseline.shape, bool), {
            "floor_found": False,
            "suppressed_component_count": 0,
            "suppressed_pixels": 0,
        }
    from .slot_inconsistency import floor_aligned_added_components

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


def target_change_evidence_from_ledger(
    tracking_attempts: dict[str, Any], targets: Sequence[ObjectHypothesis]
) -> tuple[list[bool], dict[str, Any]]:
    """Map stage 3's explicit target-vs-clean-render absence onto consolidated objects."""

    attempts = tracking_attempts.get("stages", {}).get("target_to_clean", {}).get("attempts", [])
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


def resolve_object_consistent_labels(
    reconstruction_artifact_dir: str | Path,
    baseline_labels: np.ndarray,
    source_changed_objects: Sequence[ObjectMask],
    forward_track_masks: Sequence[np.ndarray | None],
    forward_track_rows: Sequence[dict[str, Any]],
    target_inventory_objects: Sequence[ObjectMask],
    tracking_attempts: dict[str, Any],
    source_sam_map: np.ndarray,
    target_sam_map: np.ndarray,
    source_dino_map: np.ndarray,
    target_dino_map: np.ndarray,
    source_render: np.ndarray,
    target_image: np.ndarray,
    *,
    consolidation: ConsolidationSettings,
    settings: SlotSettings,
    output_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Refine ``baseline_labels`` (stage 10's output) into the object-consistent result.

    Returns ``(guarded_labels, full_mask_labels, diagnostics)``: both are the
    same decision applied at two rasterization footprints (a conservative
    trusted-target-only footprint vs. the full replacement footprint the
    method is named for); ``guarded_labels`` is the more conservative one.
    ``reconstruction_artifact_dir`` is stage 1's output directory (for
    ``reconstruction.npz``/``geometry.npz``/``config.json``);
    ``tracking_attempts`` is stage 3's ``tracking_attempts.json`` payload
    (used only for its ``target_to_clean`` explicit-absence evidence).
    ``source_changed_objects``/``forward_track_masks``/``forward_track_rows``
    are stage 6's changed-candidate proposals and their forward SAM2 tracks
    (aligned lists; a ``None`` track means that candidate's propagation
    failed). ``target_inventory_objects`` is stage 2's *raw, unfiltered*
    target proposal list.
    """

    artifact = Path(reconstruction_artifact_dir)
    sources = consolidate_hypotheses(
        source_changed_objects, forward_track_masks, forward_track_rows, settings=consolidation
    )
    target_inventory = list(target_inventory_objects)
    neutral_rows = [{"rejection_reasons": []} for _ in target_inventory]
    targets = consolidate_hypotheses(
        target_inventory, [None] * len(target_inventory), neutral_rows, settings=consolidation
    )
    matches = match_object_slots(sources, targets, settings=settings)
    decisions = []
    for match in matches:
        source, target = sources[match.source_index], targets[match.target_index]
        retention = aligned_patch_retention(
            source_dino_map, target_dino_map, source.mask, target.mask, settings=settings
        )
        sam = descriptor_cosine(source_sam_map, source.mask, target_sam_map, target.mask)
        dino = descriptor_cosine(source_dino_map, source.mask, target_dino_map, target.mask)
        color = color_intersection(source_render, source.mask, target_image, target.mask)
        target_geometry = mask_geometry(target.mask)
        support_surface = support_surface_evidence(target_geometry, settings=settings)
        elsewhere = find_identity_elsewhere(
            match.source_index,
            match.target_index,
            sources,
            targets,
            source_sam_map,
            target_sam_map,
            source_dino_map,
            target_dino_map,
            settings=settings,
        )
        decision = decide_slot_replacement(
            match,
            retention,
            sam_cosine=sam,
            dino_cosine=dino,
            color_similarity=color,
            identity_elsewhere=bool(elsewhere["found"]),
            target_support_surface=bool(support_surface["is_support_surface"]),
            settings=settings,
        )
        decisions.append(
            {
                "source_index": match.source_index,
                "target_index": match.target_index,
                "source_proposal_ids": list(source.proposal_ids),
                "target_proposal_ids": list(target.proposal_ids),
                "target_area_pixels": target.area_pixels,
                "match": match.summary(),
                "patch_retention": retention.summary(),
                "sam_cosine": sam,
                "dino_cosine": dino,
                "color_intersection": color,
                "identity_elsewhere": elsewhere,
                "target_geometry": target_geometry.summary(),
                "support_surface": support_surface,
                "decision": decision.summary(),
            }
        )

    # Multiple source fragments can select the same target hypothesis. Keep
    # the strongest replacement evidence once, then emit the target mask.
    promoted_by_target: dict[int, dict[str, Any]] = {}
    for row in decisions:
        if row["decision"]["verdict"] != "replaced":
            continue
        score = (
            float(row["patch_retention"]["lost_fraction"] or 0.0)
            + 0.25 * len(row["decision"]["mismatch_signals"])
            + row["match"]["score"]
        )
        previous = promoted_by_target.get(row["target_index"])
        if previous is None or score > previous["promotion_score"]:
            promoted_by_target[row["target_index"]] = {**row, "promotion_score": score}

    baseline = np.asarray(baseline_labels, np.uint8)
    native_shape = sources[0].mask.shape if sources else target_image.shape[:2]
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
        and len(row["decision"]["mismatch_signals"]) >= settings.minimum_cleanup_mismatch_signals
    ]
    force_cleanup_sources = [sources[row["source_index"]].mask for row in force_cleanup_rows]
    force_cleanup_targets = [targets[row["target_index"]].mask for row in force_cleanup_rows]
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
                sam_cosine=descriptor_cosine(target_sam_map, anchor_mask, target_sam_map, candidate_mask),
                dino_cosine=descriptor_cosine(target_dino_map, anchor_mask, target_dino_map, candidate_mask),
                color_intersection=color_intersection(target_image, anchor_mask, target_image, candidate_mask),
                settings=settings,
            )
            if not evidence["eligible"]:
                continue
            selected_target_indices.add(candidate_index)
            companion_rows.append(
                {
                    "anchor_target_index": anchor_index,
                    "target_index": candidate_index,
                    "target_proposal_ids": list(candidate_target.proposal_ids),
                    "evidence": evidence,
                }
            )
    companion_targets = [targets[row["target_index"]].mask for row in companion_rows]
    trusted_targets.extend(companion_targets)
    promoted_targets = [targets[row["target_index"]].mask for row in promoted_rows] + companion_targets
    target_change_flags, target_change_summary = target_change_evidence_from_ledger(tracking_attempts, targets)
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
        if row["cleanup"]["added_removed_fraction"] < settings.weak_identity_minimum_added_removed_fraction
    ]
    consensus_baseline, removal_fragment_changed = relabel_changed_target_fragments(
        consensus_baseline,
        [sources[row["source_index"]].mask for row in removal_fragment_rows],
        label=2,
    )
    replacement_ownership, protected_removed, depth_ownership = _replacement_depth_ownership(
        artifact, consensus_baseline, promoted_targets, force_cleanup_targets, settings
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
    floor_suppression, floor_summary = _floor_suppression(artifact, baseline, settings)
    full_floor_suppression = floor_suppression & (full_labels == 1)
    guarded_floor_suppression = floor_suppression & (guarded_labels == 1)
    full_labels[full_floor_suppression] = 0
    guarded_labels[guarded_floor_suppression] = 0

    diagnostics = {
        "ground_truth_used": False,
        "source_hypotheses": len(sources),
        "target_inventory_proposals": len(target_inventory),
        "target_hypotheses": len(targets),
        "slot_matches": len(matches),
        "replacement_candidate_count": len(candidate_rows),
        "plausibility_rejected_count": len(plausibility_rejected_rows),
        "promoted_target_count": len(promoted_rows),
        "replacement_companion_count": len(companion_rows),
        "rasterized_target_count": len(promoted_rows) + len(companion_rows),
        "object_class_consensus": consensus_summary,
        "object_consensus_changed_pixels": int(consensus_changed.sum()),
        "removal_fragment_changed_pixels": int(removal_fragment_changed.sum()),
        "cleanup_eligible_target_count": len(trusted_target_rows),
        "source_cleanup_eligible_target_count": len(cleanup_rows),
        "force_cleanup_target_count": len(force_cleanup_rows),
        "trusted_full_target_count": len(trusted_targets),
        "promoted_pixels": int(promoted_mask.sum()),
        "guarded_promoted_pixels": int(guarded_mask.sum()),
        "dropped_removed_pixels": int(dropped_removed.sum()),
        "full_dropped_removed_pixels": int(full_dropped_removed.sum()),
        "depth_ownership": depth_ownership,
        "floor_suppression": floor_summary,
        "full_floor_suppressed_pixels": int(full_floor_suppression.sum()),
        "guarded_floor_suppressed_pixels": int(guarded_floor_suppression.sum()),
        "full_prediction_changed_pixels": int(np.sum(full_labels != baseline)),
        "prediction_changed_pixels": int(np.sum(guarded_labels != baseline)),
        "decisions": decisions,
    }

    if output_dir is not None:
        # Note: ``full_labels``/``guarded_labels``/``baseline`` are at the
        # ChangeSim-native resolution stage 10 already resolved to, while
        # ``source_render``/``target_image`` (and therefore every mask this
        # function derives from proposals) stay at MASt3R's reconstruction
        # grid. A same-resolution RGB overlay needs the native-resolution
        # target image callers already have (e.g. stage 3's saved
        # ``target.png`` or the original ``image1`` load) -- produce it at
        # the ``run_pair`` level instead of guessing a resize here.
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_image(output / "baseline_labels.png", baseline)
        save_image(output / "labels_full_mask.png", full_labels)
        save_image(output / "labels_guarded.png", guarded_labels)
        save_image(output / "labels.png", guarded_labels)
        save_image(output / "labels_full_mask_color.png", colorize(full_labels))
        save_image(output / "labels_color.png", colorize(guarded_labels))
        save_json(output / "inference.json", diagnostics)

    return guarded_labels, full_labels, diagnostics
