"""Association-first change resolver for the corrected sentinel experiment.

This module is intentionally separate from the pairwise GOLDILOCS pipeline.
It associates object proposals extracted from the two *real* RGB images before
consulting point-cloud geometry.  Geometry is then only supporting evidence
for endpoint observability and for pairing two independently absent endpoints
as a replacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation

from ..types import Label, ObjectMask, Reconstruction
from .obvious_change_sentinel import (
    compose_sentinel,
    mask_iou,
    project_masks_with_zbuffer,
    select_large_candidates,
)
from .sam3_identity_location import (
    FeatureDescriptorBatch,
    associate_identities,
    cosine_similarity_matrix,
    mask_descriptors,
)


@dataclass(frozen=True)
class RealImageAssociation:
    """One globally reserved identity match between real I0 and real I1."""

    source_index: int
    target_index: int
    source_proposal_id: int
    target_proposal_id: int
    cosine: float
    source_margin: float
    target_margin: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReplacementPair:
    """Two verified-absent endpoints occupying the same physical location."""

    source_index: int
    target_index: int
    source_proposal_id: int
    target_proposal_id: int
    source_to_target_iou: float
    target_to_source_iou: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SurroundingSceneSupport:
    """Mutual visibility of a static ring around a candidate object."""

    ring_pixels: int
    valid_pixels: int
    consistent_pixels: int
    valid_fraction: float
    consistent_fraction: float
    threshold: float
    relative_depth_tolerance: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _proposal_id(obj: ObjectMask) -> int:
    if "automatic_proposal_id" not in obj.metadata:
        raise ValueError("proposal lacks automatic_proposal_id")
    return int(obj.metadata["automatic_proposal_id"])


def _subset_features(
    features: FeatureDescriptorBatch, indices: Sequence[int]
) -> FeatureDescriptorBatch:
    selected = np.asarray(list(indices), dtype=np.int64)
    return FeatureDescriptorBatch(
        vectors=np.asarray(features.vectors[selected], dtype=np.float32),
        valid=np.asarray(features.valid[selected], dtype=bool),
        effective_cells=np.asarray(features.effective_cells[selected], dtype=np.float32),
    )


def associate_real_image_instances(
    source_objects: Sequence[ObjectMask],
    target_objects: Sequence[ObjectMask],
    source_features: FeatureDescriptorBatch,
    target_features: FeatureDescriptorBatch,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    *,
    minimum_cosine: float,
    minimum_margin: float = 0.0,
    area_ratio_bounds: tuple[float, float] = (0.25, 4.0),
    require_mutual_nearest: bool = True,
) -> tuple[list[RealImageAssociation], np.ndarray]:
    """Globally associate a deduplicated real-image object inventory.

    No projected-mask overlap is included in the association score.  This is
    the critical correction relative to the old location-first experiments:
    appearance establishes persistence, while geometry is considered later.
    """

    source_indices = tuple(int(value) for value in source_indices)
    target_indices = tuple(int(value) for value in target_indices)
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("source inventory contains duplicate indices")
    if len(target_indices) != len(set(target_indices)):
        raise ValueError("target inventory contains duplicate indices")
    if not source_indices or not target_indices:
        return [], np.empty((len(source_indices), len(target_indices)), np.float32)

    source_areas = np.asarray(
        [np.asarray(source_objects[index].mask, bool).sum() for index in source_indices],
        dtype=np.float32,
    )
    target_areas = np.asarray(
        [np.asarray(target_objects[index].mask, bool).sum() for index in target_indices],
        dtype=np.float32,
    )
    matches, similarity, _ = associate_identities(
        _subset_features(source_features, source_indices),
        _subset_features(target_features, target_indices),
        np.zeros((len(source_indices), len(target_indices)), dtype=np.float32),
        source_areas,
        target_areas,
        minimum_cosine=float(minimum_cosine),
        minimum_margin=float(minimum_margin),
        area_ratio_bounds=area_ratio_bounds,
        same_location_bonus=0.0,
        require_mutual_nearest=bool(require_mutual_nearest),
    )
    output = [
        RealImageAssociation(
            source_index=source_indices[match.source_index],
            target_index=target_indices[match.target_index],
            source_proposal_id=_proposal_id(source_objects[source_indices[match.source_index]]),
            target_proposal_id=_proposal_id(target_objects[target_indices[match.target_index]]),
            cosine=float(match.cosine),
            source_margin=float(match.source_margin),
            target_margin=float(match.target_margin),
        )
        for match in matches
    ]
    output.sort(key=lambda value: (value.source_proposal_id, value.target_proposal_id))
    return output, similarity


def unmatched_inventory_indices(
    selected_source: Sequence[int],
    selected_target: Sequence[int],
    matches: Sequence[RealImageAssociation],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return selected endpoints not consumed by the global assignment."""

    matched_source = {value.source_index for value in matches}
    matched_target = {value.target_index for value in matches}
    return (
        tuple(index for index in selected_source if index not in matched_source),
        tuple(index for index in selected_target if index not in matched_target),
    )


def surrounding_scene_support(
    mask: np.ndarray,
    opposing_depth: np.ndarray,
    observed_depth: np.ndarray,
    opposing_coverage: np.ndarray,
    *,
    ring_radius: int = 8,
    minimum_ring_pixels: int = 64,
    visibility_threshold: float = 0.80,
    relative_depth_tolerance: float = 0.10,
) -> SurroundingSceneSupport:
    """Check an object's observable context without trusting its own depth.

    Added and removed objects are precisely where monocular point depth may be
    unreliable.  The dilated ring excludes the object and instead asks whether
    nearby static scene points are rendered and depth-consistent in both views.
    """

    binary = np.asarray(mask, dtype=bool)
    opposing = np.asarray(opposing_depth, dtype=np.float32)
    observed = np.asarray(observed_depth, dtype=np.float32)
    coverage = np.asarray(opposing_coverage, dtype=bool)
    if any(value.shape != binary.shape for value in (opposing, observed, coverage)):
        raise ValueError("ring support inputs must share a shape")
    if ring_radius < 1 or minimum_ring_pixels < 1:
        raise ValueError("ring radius and minimum pixels must be positive")
    ring = binary_dilation(binary, iterations=int(ring_radius)) & ~binary
    ring_count = int(ring.sum())
    finite = (
        coverage
        & np.isfinite(opposing)
        & np.isfinite(observed)
        & (opposing > 0)
        & (observed > 0)
    )
    valid = ring & finite
    relative_error = np.abs(opposing - observed) / np.maximum(np.abs(observed), 1e-6)
    consistent = valid & (relative_error <= float(relative_depth_tolerance))
    valid_count = int(valid.sum())
    consistent_count = int(consistent.sum())
    valid_fraction = valid_count / max(ring_count, 1)
    consistent_fraction = consistent_count / max(ring_count, 1)
    reasons = []
    if ring_count < minimum_ring_pixels:
        reasons.append("insufficient_surrounding_ring_area")
    if valid_fraction < visibility_threshold:
        reasons.append("insufficient_surrounding_render_support")
    if consistent_fraction < visibility_threshold:
        reasons.append("insufficient_surrounding_depth_consistency")
    return SurroundingSceneSupport(
        ring_pixels=ring_count,
        valid_pixels=valid_count,
        consistent_pixels=consistent_count,
        valid_fraction=float(valid_fraction),
        consistent_fraction=float(consistent_fraction),
        threshold=float(visibility_threshold),
        relative_depth_tolerance=float(relative_depth_tolerance),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def pair_joint_replacements(
    source_objects: Sequence[ObjectMask],
    target_objects: Sequence[ObjectMask],
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    source_to_target_masks: Sequence[np.ndarray],
    target_to_source_masks: Sequence[np.ndarray],
    *,
    minimum_iou: float = 0.30,
    identity_similarity: np.ndarray | None = None,
    maximum_identity_cosine: float | None = None,
) -> list[ReplacementPair]:
    """Pair absent endpoints only when location agrees in both directions.

    Hungarian assignment maximizes the weaker of the two directional IoUs.
    Thus a single misleading projection cannot create a replacement by itself.
    """

    source_indices = tuple(int(value) for value in source_indices)
    target_indices = tuple(int(value) for value in target_indices)
    if not source_indices or not target_indices:
        return []
    scores = np.full((len(source_indices), len(target_indices)), -1.0, np.float32)
    if (identity_similarity is None) != (maximum_identity_cosine is None):
        raise ValueError("identity_similarity and maximum_identity_cosine are joint options")
    if identity_similarity is not None:
        identity_similarity = np.asarray(identity_similarity, dtype=np.float32)
        if identity_similarity.shape != (len(source_objects), len(target_objects)):
            raise ValueError("identity similarity matrix has the wrong shape")
    directional: dict[tuple[int, int], tuple[float, float]] = {}
    for row, source_index in enumerate(source_indices):
        projected_source = np.asarray(source_to_target_masks[source_index], bool)
        source_real = np.asarray(source_objects[source_index].mask, bool)
        for column, target_index in enumerate(target_indices):
            if identity_similarity is not None and float(
                identity_similarity[source_index, target_index]
            ) > float(maximum_identity_cosine):
                continue
            target_real = np.asarray(target_objects[target_index].mask, bool)
            projected_target = np.asarray(target_to_source_masks[target_index], bool)
            forward = mask_iou(projected_source, target_real)
            reverse = mask_iou(projected_target, source_real)
            score = min(forward, reverse)
            directional[(row, column)] = (forward, reverse)
            if score >= minimum_iou:
                scores[row, column] = score

    rows, columns = linear_sum_assignment(-scores)
    output: list[ReplacementPair] = []
    for row, column in zip(rows, columns):
        if scores[row, column] < minimum_iou:
            continue
        source_index, target_index = source_indices[row], target_indices[column]
        forward, reverse = directional[(row, column)]
        output.append(
            ReplacementPair(
                source_index=source_index,
                target_index=target_index,
                source_proposal_id=_proposal_id(source_objects[source_index]),
                target_proposal_id=_proposal_id(target_objects[target_index]),
                source_to_target_iou=float(forward),
                target_to_source_iou=float(reverse),
                score=float(scores[row, column]),
            )
        )
    output.sort(key=lambda value: (-value.score, value.source_proposal_id, value.target_proposal_id))
    return output


def paint_joint_replacements(
    labels: np.ndarray,
    parent_labels: np.ndarray,
    pairs: Sequence[ReplacementPair],
    source_to_target_masks: Sequence[np.ndarray],
    target_objects: Sequence[ObjectMask],
) -> tuple[np.ndarray, dict[str, int]]:
    """Relabel only newly writable source/target overlap as REPLACED."""

    from PIL import Image

    output = np.asarray(labels, dtype=np.uint8).copy()
    parent = np.asarray(parent_labels, dtype=np.uint8)
    replacement = np.zeros(output.shape, dtype=bool)
    for pair in pairs:
        mask = np.asarray(source_to_target_masks[pair.source_index], bool) & np.asarray(
            target_objects[pair.target_index].mask, bool
        )
        if mask.shape != output.shape:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    output.shape[::-1], Image.Resampling.NEAREST
                ),
                dtype=bool,
            )
        replacement |= mask
    writable = parent == int(Label.UNCHANGED)
    changed = replacement & writable
    output[changed] = int(Label.REPLACED)
    if not np.array_equal(output[~writable], labels[~writable]):
        raise AssertionError("replacement resolver modified frozen parent changes")
    return output, {
        "joint_replacement_pairs": len(pairs),
        "joint_replacement_pixels": int(changed.sum()),
    }


def _copy_endpoint(obj: ObjectMask, mask: np.ndarray, label: Label) -> ObjectMask:
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


@dataclass(frozen=True)
class AssociationResolverSettings:
    """Thresholds for the final real-image association/replacement resolver.

    This resolver skips geometry-support gating of endpoint absence
    entirely and relies only on a targeted SAM2 absence check plus
    same-place pairing (in the original research code's own internal
    ablation ladder, this composition was labeled "r4_no_geometry_ablation";
    its geometry-gated and parent-replay siblings fed comparison outputs
    this pipeline never uses and are not represented here).
    """

    minimum_valid_depth: float
    minimum_bidirectional_margin: float
    area_ratio_bounds: tuple[float, float]
    require_mutual_nearest: bool
    minimum_directional_iou: float
    different_identity_margin: float
    minimum_feature_cells: float
    minimum_mask_area_fraction: float
    minimum_bbox_side_fraction: float
    minimum_mask_area: int
    minimum_predicted_iou: float
    minimum_stability_score: float
    duplicate_iou: float
    duplicate_containment: float
    reject_frame_border: bool = True


class Sam2LikeTracker:
    """Structural type for the tracker this resolver needs: see adapters/sam2.py."""

    def track(
        self, masks: Sequence[np.ndarray], source_image: np.ndarray, target_image: np.ndarray
    ) -> list[Any]:  # pragma: no cover - protocol only
        raise NotImplementedError


def resolve_real_image_associations(
    reconstruction: Reconstruction,
    source_objects: Sequence[ObjectMask],
    target_objects: Sequence[ObjectMask],
    source_map: np.ndarray,
    target_map: np.ndarray,
    parent_labels: np.ndarray,
    identity_threshold: float,
    tracker: Sam2LikeTracker,
    image0: np.ndarray,
    image1: np.ndarray,
    *,
    settings: AssociationResolverSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine ``parent_labels`` (the feature-veto-gated direct-replacement
    raster) with identity/location reasoning over the two real images.

    ``source_objects``/``source_map`` are the *real*, un-warped T0 SAM3
    proposals/features (stage 9's generation); ``target_objects``/
    ``target_map`` are T1's (already real -- reused directly from stages
    2/4, no separate generation needed). This is a from-scratch extraction
    of the composition the original research code labeled
    "r4_no_geometry_ablation" in
    ``scripts/run_real_image_association_resolver.py``'s per-pair
    computation: its geometry-gated added/removed and parent-replay
    siblings are intentionally not computed at all.
    """

    shape = tuple(int(value) for value in image0.shape[:2])
    source_features = mask_descriptors(
        source_map, source_objects, minimum_feature_cells=settings.minimum_feature_cells
    )
    target_features = mask_descriptors(
        target_map, target_objects, minimum_feature_cells=settings.minimum_feature_cells
    )
    selection_kwargs = dict(
        minimum_area_fraction=settings.minimum_mask_area_fraction,
        minimum_bbox_side_fraction=settings.minimum_bbox_side_fraction,
        minimum_mask_area=settings.minimum_mask_area,
        minimum_predicted_iou=settings.minimum_predicted_iou,
        minimum_stability_score=settings.minimum_stability_score,
        duplicate_iou=settings.duplicate_iou,
        duplicate_containment=settings.duplicate_containment,
        reject_frame_border=settings.reject_frame_border,
    )
    source_selection = select_large_candidates(source_objects, source_features, shape, **selection_kwargs)
    target_selection = select_large_candidates(target_objects, target_features, shape, **selection_kwargs)

    # Only the source->target projection is needed: it supplies the painted
    # mask for a verified-REMOVED endpoint. added_no_geometry paints target
    # objects directly in their own (already target-aligned) pixel grid, and
    # this resolver never consults target->source geometry at all.
    source_to_target = project_masks_with_zbuffer(
        reconstruction.points[0],
        [obj.mask for obj in source_objects],
        reconstruction.intrinsics[1],
        reconstruction.world_to_camera[1],
        shape,
        minimum_depth=settings.minimum_valid_depth,
    )

    matches, _ = associate_real_image_instances(
        source_objects,
        target_objects,
        source_features,
        target_features,
        source_selection.selected_indices,
        target_selection.selected_indices,
        minimum_cosine=identity_threshold,
        minimum_margin=settings.minimum_bidirectional_margin,
        area_ratio_bounds=settings.area_ratio_bounds,
        require_mutual_nearest=settings.require_mutual_nearest,
    )
    unmatched_source, unmatched_target = unmatched_inventory_indices(
        source_selection.selected_indices, target_selection.selected_indices, matches
    )
    source_ids = [int(source_objects[index].metadata["automatic_proposal_id"]) for index in unmatched_source]
    target_ids = [int(target_objects[index].metadata["automatic_proposal_id"]) for index in unmatched_target]
    source_masks = [np.asarray(source_objects[index].mask, bool) for index in unmatched_source]
    target_masks = [np.asarray(target_objects[index].mask, bool) for index in unmatched_target]

    source_attempts = tracker.track(source_masks, image0, image1)
    target_attempts = tracker.track(target_masks, image1, image0)
    source_presence = {
        pid: _attempt_presence(attempt) for pid, attempt in zip(source_ids, source_attempts, strict=True)
    }
    target_presence = {
        pid: _attempt_presence(attempt) for pid, attempt in zip(target_ids, target_attempts, strict=True)
    }
    source_absent = [
        index for index, pid in zip(unmatched_source, source_ids, strict=True)
        if source_presence.get(pid) is False
    ]
    target_absent = [
        index for index, pid in zip(unmatched_target, target_ids, strict=True)
        if target_presence.get(pid) is False
    ]

    removed_no_geometry = [
        _copy_endpoint(source_objects[index], source_to_target.masks[index], Label.REMOVED)
        for index in source_absent
    ]
    added_no_geometry = [
        _copy_endpoint(target_objects[index], target_objects[index].mask, Label.ADDED)
        for index in target_absent
    ]

    similarity = cosine_similarity_matrix(source_features, target_features)
    replacement_limit = identity_threshold - settings.different_identity_margin
    raw_source_masks = [np.asarray(obj.mask, bool) for obj in source_objects]
    raw_target_masks = [np.asarray(obj.mask, bool) for obj in target_objects]
    no_geometry_pairs = pair_joint_replacements(
        source_objects,
        target_objects,
        source_absent,
        target_absent,
        raw_source_masks,
        raw_target_masks,
        minimum_iou=settings.minimum_directional_iou,
        identity_similarity=similarity,
        maximum_identity_cosine=replacement_limit,
    )

    labels, diagnostics = compose_sentinel(parent_labels, added_no_geometry, removed_no_geometry)
    labels, replacement_diagnostics = paint_joint_replacements(
        labels, parent_labels, no_geometry_pairs, raw_source_masks, target_objects
    )
    diagnostics.update(replacement_diagnostics)
    diagnostics.update(
        {
            "identity_matches": len(matches),
            "unmatched_source": len(unmatched_source),
            "unmatched_target": len(unmatched_target),
            "verified_absent_source": len(source_absent),
            "verified_absent_target": len(target_absent),
            "joint_replacements": len(no_geometry_pairs),
        }
    )
    return labels, diagnostics


def resolver_settings_from_config(config: Mapping[str, Any]) -> AssociationResolverSettings:
    """Build :class:`AssociationResolverSettings` from the merged pipeline config.

    ``config`` is the full ``pipeline.yaml`` mapping. This resolver reuses
    two sibling stages' settings verbatim rather than declaring its own
    copies: the large/high-quality endpoint candidate rule is stage 9's
    (``obvious_object_sentinel.candidate_selection``, the same rule that
    built its own real-I0 candidates) and the depth/feature floors are also
    stage 9's (``obvious_object_sentinel.geometry_observability.minimum_valid_depth``,
    ``...sam3.features.minimum_feature_cells``,
    ``...sam3.proposal_generation.minimum_mask_area_pixels``) -- matching
    ``scripts/run_real_image_association_resolver.py``'s original
    ``_selection_kwargs``, which reads its candidate-selection and depth
    settings from exactly that parent stage's config, not its own.
    """

    resolver = config["association_resolver"]
    sentinel = config["obvious_object_sentinel"]
    association = resolver["association"]
    replacement = resolver["replacement"]
    candidate_selection = sentinel["candidate_selection"]
    duplicate = candidate_selection["duplicate_suppression"]
    return AssociationResolverSettings(
        minimum_valid_depth=float(sentinel["geometry_observability"]["minimum_valid_depth"]),
        minimum_bidirectional_margin=float(association["minimum_bidirectional_margin"]),
        area_ratio_bounds=tuple(float(value) for value in association["area_ratio_bounds"]),
        require_mutual_nearest=bool(association["require_mutual_nearest"]),
        minimum_directional_iou=float(replacement["minimum_directional_iou"]),
        different_identity_margin=float(replacement["different_identity_margin"]),
        minimum_feature_cells=float(sentinel["sam3"]["features"]["minimum_feature_cells"]),
        minimum_mask_area_fraction=float(candidate_selection["minimum_mask_area_fraction"]),
        minimum_bbox_side_fraction=float(candidate_selection["minimum_bbox_side_fraction"]),
        minimum_mask_area=int(sentinel["sam3"]["proposal_generation"]["minimum_mask_area_pixels"]),
        minimum_predicted_iou=float(candidate_selection["minimum_predicted_iou"]),
        minimum_stability_score=float(candidate_selection["minimum_stability_score"]),
        duplicate_iou=float(duplicate["mask_iou"]),
        duplicate_containment=float(duplicate["containment_fraction"]),
        reject_frame_border=candidate_selection["frame_border_contact_policy"] == "abstain",
    )
