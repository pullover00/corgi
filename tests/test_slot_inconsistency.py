from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

from ocmask.stages.branch_b2 import ObjectHypothesis
from ocmask.stages.slot_inconsistency import (
    PatchRetention,
    SlotMatch,
    SlotSettings,
    aligned_patch_retention,
    arbitrate_target_object_classes,
    decide_slot_replacement,
    dominant_changed_class_consensus,
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
from ocmask.types import ObjectMask

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.run_slot_inconsistency_replacement_experiment import (
    _raster_baseline_records,
    _sha256_file,
)


def _mask(y: int, x: int, size: int = 12) -> np.ndarray:
    value = np.zeros((48, 48), bool); value[y:y + size, x:x + size] = True
    return value


def _hyp(mask: np.ndarray, track=None) -> ObjectHypothesis:
    return ObjectHypothesis(0, (0,), (1,), mask, track, "present" if track is not None else "ambiguous", int(mask.sum()))


def test_slot_matching_uses_aligned_overlap_not_list_order() -> None:
    source = _mask(10, 10)
    matches = match_object_slots(
        [_hyp(source)], [ObjectMask(_mask(28, 28)), ObjectMask(_mask(11, 10))],
        settings=SlotSettings(),
    )
    assert len(matches) == 1
    assert matches[0].target_index == 1
    assert matches[0].eligible


def test_slot_matching_rejects_large_support_surface_around_small_object() -> None:
    source = _mask(18, 18, 6)
    floor = np.ones((48, 48), bool)
    matches = match_object_slots(
        [_hyp(source)], [ObjectMask(floor)], settings=SlotSettings()
    )
    assert matches == ()


def test_half_lost_object_has_high_lost_identity_fraction() -> None:
    channels, grid = 16, 4
    source = np.zeros((channels, grid, grid), np.float32)
    target = np.zeros_like(source)
    for y in range(grid):
        for x in range(grid):
            source[y * grid + x, y, x] = 1.0
            target[y * grid + x if x < 2 else (y * grid + x + 1) % channels, y, x] = 1.0
    mask = np.ones((40, 40), bool)
    evidence = aligned_patch_retention(source, target, mask, mask, settings=SlotSettings(same_patch_cosine=0.9))
    assert evidence.valid
    assert evidence.retained_fraction == 0.5
    assert evidence.lost_fraction == 0.5


def test_two_mismatches_and_identity_loss_produce_replacement() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.25, 0.7, 0.5, 0.01, 1.0, 0.6, True)
    retention = PatchRetention(10, 10, 8, 3, 7, 0.3, 0.7, 0.4, True)
    decision = decide_slot_replacement(
        match, retention, sam_cosine=0.5, dino_cosine=0.4,
        color_similarity=0.7, identity_elsewhere=False, settings=settings,
    )
    assert decision.verdict == "replaced"
    assert len(decision.mismatch_signals) == 2


def test_successful_track_does_not_block_replacement() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.4, 0.8, 0.9, 0.01, 1.0, 0.8, True)
    retention = PatchRetention(10, 10, 10, 2, 8, 0.2, 0.8, 0.3, True)
    result = decide_slot_replacement(
        match, retention, sam_cosine=0.4, dino_cosine=0.4,
        color_similarity=0.2, identity_elsewhere=False, settings=settings,
    )
    assert result.verdict == "replaced"


def test_identity_elsewhere_routes_away_from_replacement() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.3, 0.7, 0.2, 0.01, 1.0, 0.6, True)
    retention = PatchRetention(10, 10, 8, 2, 8, 0.2, 0.8, 0.3, True)
    result = decide_slot_replacement(
        match, retention, sam_cosine=0.4, dino_cosine=0.4,
        color_similarity=0.2, identity_elsewhere=True, settings=settings,
    )
    assert result.verdict == "moved_elsewhere"


def test_strong_track_and_color_route_moved_object_before_replacement() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.12, 0.3, 0.85, 0.04, 1.1, 0.5, True)
    retention = PatchRetention(20, 20, 5, 0, 20, 0.0, 1.0, 0.4, True)
    result = decide_slot_replacement(
        match,
        retention,
        sam_cosine=0.66,
        dino_cosine=0.55,
        color_similarity=0.52,
        identity_elsewhere=False,
        settings=settings,
    )
    assert result.verdict == "moved_elsewhere"
    assert "track_color_motion_rescue" in result.rescue_signals


def test_strong_patch_retention_rescues_same_identity() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.3, 0.7, 0.2, 0.01, 1.0, 0.6, True)
    retention = PatchRetention(10, 10, 9, 8, 2, 0.8, 0.2, 0.9, True)
    result = decide_slot_replacement(
        match, retention, sam_cosine=0.6, dino_cosine=0.5,
        color_similarity=0.2, identity_elsewhere=False, settings=settings,
    )
    assert result.verdict == "same_identity"
    assert "aligned_patch_identity_rescue" in result.rescue_signals


def test_support_surface_geometry_rejects_thin_shelf_and_fragmented_wall() -> None:
    settings = SlotSettings()
    shelf = np.zeros((64, 64), bool); shelf[30:34, 8:56] = True
    shelf_evidence = support_surface_evidence(mask_geometry(shelf), settings=settings)
    assert shelf_evidence["is_support_surface"]
    assert "thin_elongated_support_surface" in shelf_evidence["reasons"]

    wall = np.zeros((64, 64), bool)
    for y, x in ((4, 4), (4, 28), (4, 52), (52, 4), (52, 28), (52, 52)):
        wall[y:y + 4, x:x + 4] = True
    wall_evidence = support_surface_evidence(mask_geometry(wall), settings=settings)
    assert wall_evidence["is_support_surface"]
    assert "fragmented_low_fill_support_surface" in wall_evidence["reasons"]

    compact = _mask(16, 16, 16)
    assert not support_surface_evidence(mask_geometry(compact), settings=settings)["is_support_surface"]


def test_support_surface_cannot_be_promoted_to_replacement() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.3, 0.6, 0.0, 0.02, 1.0, 0.5, True)
    retention = PatchRetention(10, 10, 8, 1, 9, 0.1, 0.9, 0.3, True)
    result = decide_slot_replacement(
        match,
        retention,
        sam_cosine=0.3,
        dino_cosine=0.4,
        color_similarity=0.1,
        identity_elsewhere=False,
        target_support_surface=True,
        settings=settings,
    )
    assert result.verdict == "abstain"
    assert "target_is_support_surface" in result.reasons


def test_replacement_plausibility_requires_one_coherent_target_object() -> None:
    settings = SlotSettings()
    strong_match = SlotMatch(0, 0, 0.50, 0.70, 0.0, 0.01, 1.0, 0.55, True)
    weak_match = SlotMatch(0, 0, 0.18, 0.70, 0.0, 0.01, 1.0, 0.39, True)
    compact = mask_geometry(_mask(12, 12, 8))
    fragmented_mask = np.zeros((64, 64), bool)
    fragmented_mask[10:14, 8:40] = True
    fragmented_mask[42:46, 20:52] = True
    fragmented = mask_geometry(fragmented_mask)

    assert replacement_object_plausibility(
        strong_match, compact, cleanup_eligible=False, settings=settings
    )["eligible"]
    rejected = replacement_object_plausibility(
        weak_match, fragmented, cleanup_eligible=False, settings=settings
    )
    assert not rejected["eligible"]
    assert "target_mask_is_not_one_coherent_object" in rejected["reasons"]
    assert replacement_object_plausibility(
        weak_match, fragmented, cleanup_eligible=True, settings=settings
    )["eligible"]


def test_floor_plausibility_suppresses_only_floor_dominant_added_component() -> None:
    labels = np.zeros((10, 12), np.uint8)
    labels[7:10, 0:5] = 1
    labels[2:6, 8:11] = 1
    yy, xx = np.indices(labels.shape)
    points = np.stack([xx, np.full(labels.shape, 0.4), yy + 1.0], axis=-1).astype(float)
    points[7:10, 0:5, 1] = 0.0

    suppressed, summary = floor_aligned_added_components(
        labels,
        points,
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        minimum_component_pixels=8,
        minimum_floor_fraction=0.70,
    )

    assert suppressed[7:10, 0:5].all()
    assert not suppressed[2:6, 8:11].any()
    assert summary["suppressed_component_count"] == 1


def test_replacement_companion_requires_adjacent_same_depth_same_appearance() -> None:
    anchor = np.zeros((20, 30), bool); anchor[5:15, 4:12] = True
    companion = np.zeros_like(anchor); companion[5:15, 12:20] = True
    depth = np.full(anchor.shape, 3.0, np.float32)
    accepted = replacement_companion_evidence(
        anchor,
        companion,
        depth,
        sam_cosine=0.90,
        dino_cosine=0.95,
        color_intersection=0.80,
        settings=SlotSettings(),
    )
    assert accepted["eligible"]

    depth[companion] = 3.5
    rejected = replacement_companion_evidence(
        anchor,
        companion,
        depth,
        sam_cosine=0.90,
        dino_cosine=0.95,
        color_intersection=0.80,
        settings=SlotSettings(),
    )
    assert not rejected["eligible"]
    assert "companion_depth_is_incompatible" in rejected["reasons"]


def test_dominant_object_consensus_rewrites_only_high_agreement_fragments() -> None:
    labels = np.zeros((12, 18), np.uint8)
    coherent = np.zeros_like(labels, bool); coherent[1:6, 1:9] = True
    labels[coherent] = 5
    labels[1:3, 1:3] = 1
    ambiguous = np.zeros_like(labels, bool); ambiguous[7:11, 8:18] = True
    labels[ambiguous] = 2
    labels[7:9, 8:18] = 3

    output, rewritten, summary = dominant_changed_class_consensus(
        labels,
        [coherent, ambiguous],
        minimum_pixels=8,
        minimum_changed_fraction=0.50,
        minimum_dominant_fraction=0.75,
    )

    assert np.all(output[coherent] == 5)
    assert int(rewritten.sum()) == 4
    assert np.array_equal(output[ambiguous], labels[ambiguous])
    assert summary["eligible_object_count"] == 1


def test_target_object_arbitration_maps_added_removed_to_replaced() -> None:
    labels = np.zeros((20, 20), np.uint8)
    obj = np.zeros_like(labels, bool); obj[4:16, 4:16] = True
    labels[obj] = 1
    labels[4:16, 10:16] = 2

    output, rewritten, summary = arbitrate_target_object_classes(
        labels, [obj], settings=SlotSettings()
    )

    assert np.all(output[obj] == 5)
    assert int(rewritten.sum()) == int(obj.sum())
    assert summary["decision_counts"]["added_removed_implies_replaced"] == 1


def test_target_object_arbitration_removes_moved_stripe_from_replacement() -> None:
    labels = np.zeros((24, 24), np.uint8)
    obj = np.zeros_like(labels, bool); obj[3:21, 5:19] = True
    labels[obj] = 5
    labels[10:15, 5:19] = 3

    output, rewritten, summary = arbitrate_target_object_classes(
        labels, [obj], settings=SlotSettings()
    )

    assert np.all(output[obj] == 5)
    assert int(rewritten.sum()) == 5 * 14
    assert summary["decision_counts"]["replaced_majority_over_moved_fragment"] == 1


def test_target_object_arbitration_abstains_on_fragmented_or_weakly_changed_mask() -> None:
    labels = np.zeros((30, 30), np.uint8)
    fragmented = np.zeros_like(labels, bool)
    fragmented[2:4, 2:10] = True; fragmented[20:22, 18:28] = True
    labels[fragmented] = 5
    labels[2:4, 2:6] = 3
    sparse = np.zeros_like(labels, bool); sparse[8:20, 8:20] = True
    labels[8:10, 8:20] = 5; labels[10:11, 8:20] = 3
    original = labels.copy()

    output, rewritten, summary = arbitrate_target_object_classes(
        labels, [fragmented, sparse], settings=SlotSettings()
    )

    assert np.array_equal(output, original)
    assert not rewritten.any()
    assert summary["eligible_object_count"] == 0


def test_target_object_arbitration_requires_independent_change_evidence() -> None:
    labels = np.zeros((20, 20), np.uint8)
    obj = np.zeros_like(labels, bool); obj[4:16, 4:16] = True
    labels[obj] = 5; labels[9:12, 4:16] = 3

    output, rewritten, summary = arbitrate_target_object_classes(
        labels,
        [obj],
        settings=SlotSettings(),
        object_change_evidence=[False],
    )

    assert np.array_equal(output, labels)
    assert not rewritten.any()
    assert not summary["objects"][0]["independent_change_evidence"]


def test_replacement_raster_drops_old_removed_footprint_and_writes_new_last() -> None:
    baseline = np.zeros((20, 20), np.uint8)
    baseline[2:10, 2:10] = 2
    baseline[6:14, 6:14] = 1
    baseline[16:18, 16:18] = 2
    old_mask = np.zeros_like(baseline, bool); old_mask[2:10, 2:10] = True
    new_mask = np.zeros_like(baseline, bool); new_mask[6:14, 6:14] = True

    labels, promoted, dropped = rasterize_replacement_labels(
        baseline, [old_mask], [new_mask], promote_only_changed=True
    )

    assert np.all(labels[2:6, 2:10] == 0)
    assert np.all(labels[6:14, 6:14] == 5)
    assert np.all(labels[16:18, 16:18] == 2)
    assert int(promoted.sum()) == 64
    assert int(dropped.sum()) == 48


def test_trusted_replacement_writes_complete_target_even_over_unchanged_pixels() -> None:
    baseline = np.zeros((20, 20), np.uint8)
    baseline[2:10, 2:10] = 2
    old_mask = np.zeros_like(baseline, bool); old_mask[2:10, 2:10] = True
    new_mask = np.zeros_like(baseline, bool); new_mask[6:14, 6:14] = True
    labels, promoted, _ = rasterize_replacement_labels(
        baseline,
        [old_mask],
        [new_mask],
        promote_only_changed=True,
        trusted_target_masks=[new_mask],
    )
    assert np.all(labels[new_mask] == 5)
    assert int(promoted.sum()) == int(new_mask.sum())


def test_depth_ownership_preserves_nearer_and_equal_depth_existing_objects() -> None:
    baseline = np.zeros((12, 12), np.uint8)
    baseline[1:5, 1:5] = 2
    baseline[1:5, 5:9] = 3
    target_mask = np.zeros_like(baseline, bool); target_mask[2:6, 2:8] = True
    target_depth = np.full(baseline.shape, 4.0, np.float32)
    source_depth = np.full(baseline.shape, 6.0, np.float32)
    source_depth[1:5, 1:5] = 2.0

    ownership, protected_removed, summary = frontmost_replacement_ownership(
        baseline,
        [target_mask],
        target_depth,
        source_depth,
        minimum_valid_pixels=1,
        minimum_valid_fraction=0.0,
    )

    assert not ownership[2:5, 2:5].any()  # nearer source REMOVED object
    assert not ownership[2:5, 5:8].any()  # equal-depth target MOVED object
    assert ownership[5, 2:8].all()  # no existing changed owner
    assert protected_removed[1:5, 1:5].all()
    assert summary["preserved_existing_overlap_pixels"] == 18


def test_depth_ownership_uses_closest_promoted_mask_in_target_overlap() -> None:
    baseline = np.zeros((8, 8), np.uint8); baseline[1:6, 1:6] = 2
    far = np.zeros_like(baseline, bool); far[1:6, 1:6] = True
    near = np.zeros_like(baseline, bool); near[2:4, 2:4] = True
    target_depth = np.full(baseline.shape, 5.0, np.float32)
    target_depth[near] = 2.0
    source_depth = np.full(baseline.shape, 3.0, np.float32)

    ownership, _, _ = frontmost_replacement_ownership(
        baseline,
        [far, near],
        target_depth,
        source_depth,
        minimum_valid_pixels=1,
        minimum_valid_fraction=0.0,
    )
    assert ownership[near].all()
    assert not ownership[far & ~near].any()


def test_plausible_target_unifies_nonremoved_fragments_but_respects_front_removed() -> None:
    baseline = np.zeros((8, 8), np.uint8)
    baseline[1:6, 1:3] = 1
    baseline[1:6, 3:6] = 2
    target = np.zeros_like(baseline, bool); target[1:6, 1:6] = True
    target_depth = np.full(baseline.shape, 4.0, np.float32)
    source_depth = np.full(baseline.shape, 2.0, np.float32)

    ownership, protected_removed, summary = frontmost_replacement_ownership(
        baseline,
        [target],
        target_depth,
        source_depth,
        minimum_valid_pixels=1,
        minimum_valid_fraction=0.0,
        unify_target_objects=True,
    )

    assert ownership[1:6, 1:3].all()
    assert not ownership[1:6, 3:6].any()
    assert protected_removed[1:6, 3:6].all()
    assert summary["unified_target_objects"]


def test_strong_identity_pair_overrides_depth_only_for_its_replacement_target() -> None:
    baseline = np.zeros((10, 14), np.uint8)
    baseline[2:8, 2:7] = 2
    baseline[2:8, 9:13] = 2
    target = np.zeros_like(baseline, bool); target[3:9, 3:8] = True
    target_depth = np.full(baseline.shape, 4.0, np.float32)
    source_depth = np.full(baseline.shape, 2.0, np.float32)
    ownership, protected, summary = frontmost_replacement_ownership(
        baseline,
        [target],
        target_depth,
        source_depth,
        minimum_valid_pixels=1,
        minimum_valid_fraction=0.0,
        unify_target_objects=True,
        identity_override_target_masks=[target],
    )
    assert ownership[target].all()
    assert summary["identity_override_removed_overlap_pixels"] > 0
    # The override grants target ownership, while cleanup protection remains a
    # separate decision scoped by the paired source mask.
    assert protected[2:8, 2:7].all()
    assert protected[2:8, 9:13].all()


def test_protected_front_removed_component_survives_cleanup() -> None:
    baseline = np.zeros((12, 12), np.uint8)
    baseline[2:8, 2:8] = 2
    old = np.zeros_like(baseline, bool); old[2:8, 2:8] = True
    new = np.zeros_like(baseline, bool); new[4:10, 4:10] = True
    protected = baseline == 2
    ownership = new & (baseline == 0)
    labels, _, dropped = rasterize_replacement_labels(
        baseline,
        [old],
        [new],
        promote_only_changed=False,
        replacement_ownership_mask=ownership,
        protected_removed_mask=protected,
    )
    assert not dropped.any()
    assert np.all(labels[2:8, 2:8] == 2)

    forced_labels, _, forced_dropped = rasterize_replacement_labels(
        baseline,
        [old],
        [new],
        promote_only_changed=False,
        replacement_ownership_mask=new,
        protected_removed_mask=protected,
        force_cleanup_source_masks=[old],
    )
    assert forced_dropped[2:4, 2:8].all()
    assert np.all(forced_labels[new] == 5)


def test_rasterization_respects_frontmost_ownership_mask() -> None:
    baseline = np.zeros((10, 10), np.uint8); baseline[2:8, 2:8] = 2
    target = np.zeros_like(baseline, bool); target[3:9, 3:9] = True
    ownership = np.zeros_like(baseline, bool); ownership[6:9, 3:9] = True
    labels, promoted, _ = rasterize_replacement_labels(
        baseline,
        [],
        [target],
        promote_only_changed=False,
        replacement_ownership_mask=ownership,
    )
    assert np.all(labels[3:6, 3:8] == 2)
    assert np.all(labels[6:9, 3:9] == 5)
    assert np.array_equal(promoted, target & ownership)


def test_cleanup_requires_one_to_one_geometry_and_added_or_removed_support() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.55, 0.75, 0.0, 0.01, 1.0, 0.6, True)
    target = _mask(10, 10)
    baseline = np.zeros(target.shape, np.uint8)
    baseline[target] = 2
    allowed = replacement_cleanup_evidence(
        match, 3, baseline, target, settings=settings
    )
    assert allowed["eligible"]

    baseline[target] = 3
    blocked = replacement_cleanup_evidence(
        match, 3, baseline, target, settings=settings
    )
    assert not blocked["eligible"]
    assert blocked["source_cleanup_eligible"]
    assert "target_lacks_added_or_removed_support" in blocked["reasons"]
    assert not blocked["source_cleanup_reasons"]


def test_cleanup_rejects_weak_geometry_for_noncompact_target() -> None:
    settings = SlotSettings()
    weak_match = SlotMatch(0, 0, 0.2, 0.4, 0.0, 0.01, 1.0, 0.3, True)
    target = np.zeros((48, 48), bool); target[20:22, 5:35] = True
    baseline = np.zeros(target.shape, np.uint8)
    baseline[target] = 2
    evidence = replacement_cleanup_evidence(
        weak_match, 3, baseline, target, settings=settings
    )
    assert not evidence["eligible"]
    assert not evidence["source_cleanup_eligible"]
    assert "neither_one_to_one_nor_compact_foreground" in evidence["reasons"]


def test_cleanup_allows_compact_foreground_override_for_fragmented_old_mask() -> None:
    settings = SlotSettings()
    weak_match = SlotMatch(0, 0, 0.21, 0.55, 0.0, 0.02, 2.1, 0.36, True)
    target = _mask(10, 10)
    baseline = np.zeros(target.shape, np.uint8)
    baseline[target] = 2
    evidence = replacement_cleanup_evidence(
        weak_match, 3, baseline, target, settings=settings
    )
    assert evidence["eligible"]
    assert evidence["source_cleanup_eligible"]
    assert evidence["compact_foreground_override"]


def test_cleanup_allows_smaller_replacement_inside_obsolete_old_envelope() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.37, 0.99, 0.0, 0.02, 0.37, 0.52, True)
    target = _mask(10, 10)
    baseline = np.zeros(target.shape, np.uint8)
    evidence = replacement_cleanup_evidence(
        match, 2, baseline, target, settings=settings
    )
    assert evidence["asymmetric_old_envelope"]
    assert evidence["source_cleanup_eligible"]


def test_two_signal_fragment_without_change_support_is_not_replacement_object() -> None:
    settings = SlotSettings()
    match = SlotMatch(0, 0, 0.32, 0.70, 0.0, 0.01, 1.9, 0.45, True)
    mask = np.zeros((40, 60), bool)
    mask[10:18, 8:45] = True
    for x in range(8, 50, 6):
        mask[24, x] = True
    geometry = mask_geometry(mask)
    result = replacement_object_plausibility(
        match,
        geometry,
        cleanup_eligible=False,
        mismatch_signal_count=2,
        added_removed_fraction=0.0,
        settings=settings,
    )
    assert not result["eligible"]
    assert "two_signal_fragment_lacks_change_support" in result["reasons"]


def test_cleanup_completes_connected_removed_fringe_near_replacement() -> None:
    baseline = np.zeros((24, 24), np.uint8)
    baseline[4:14, 5:16] = 2
    old = np.zeros_like(baseline, bool); old[7:14, 5:16] = True
    new = np.zeros_like(baseline, bool); new[8:16, 6:17] = True
    labels, _, dropped = rasterize_replacement_labels(
        baseline,
        [old],
        [new],
        promote_only_changed=False,
        expand_connected_removed=True,
    )
    assert np.all(labels[4:8, 5:16] == 0)
    assert dropped[4:8, 5:16].all()


def test_rejected_target_fragment_becomes_removed_without_binary_expansion() -> None:
    baseline = np.zeros((12, 12), np.uint8)
    baseline[3:8, 3:8] = 5
    baseline[5:9, 6:10] = 3
    target = np.zeros_like(baseline, bool); target[2:10, 2:11] = True
    labels, rewritten = relabel_changed_target_fragments(
        baseline, [target], label=2
    )
    assert np.all(labels[baseline != 0] == 2)
    assert np.array_equal(labels != 0, baseline != 0)
    assert int(rewritten.sum()) == int((baseline != 0).sum())


def test_raster_baseline_must_match_the_frozen_resolver_prediction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resolver"
    labels_path = root / "pairs" / "pair-1" / "r4_no_geometry_ablation" / "labels.png"
    labels_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(labels_path)
    prediction = {
        "relative_path": str(labels_path.relative_to(root)),
        "file_sha256": _sha256_file(labels_path),
        "array_sha256": "frozen-array-hash",
        "binary_array_sha256": "frozen-binary-hash",
    }
    report = {
        "selection": ["pair-1"],
        "failures": [],
        "provenance": {"ground_truth_used_in_inference": False},
        "protocol": {"predictions_frozen_before_ground_truth": True},
        "pairs": [{"id": "pair-1", "predictions": {"r4_no_geometry_ablation": prediction}}],
    }
    frozen = {
        "selection": ["pair-1"],
        "pairs": [{"id": "pair-1", "predictions": {"r4_no_geometry_ablation": prediction}}],
    }
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "predictions_frozen.json").write_text(json.dumps(frozen), encoding="utf-8")

    selection, records = _raster_baseline_records(root, "r4_no_geometry_ablation")
    assert selection == ["pair-1"]
    assert records["pair-1"]["path"] == labels_path.resolve()

    labels_path.write_bytes(b"changed after freeze")
    with pytest.raises(RuntimeError, match="changed after freeze"):
        _raster_baseline_records(root, "r4_no_geometry_ablation")
