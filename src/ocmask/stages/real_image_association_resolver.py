"""Association-first change resolver for the corrected sentinel experiment.

This module is intentionally separate from the pairwise GOLDILOCS pipeline.
It associates object proposals extracted from the two *real* RGB images before
consulting point-cloud geometry.  Geometry is then only supporting evidence
for endpoint observability and for pairing two independently absent endpoints
as a replacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation

from ..types import Label, ObjectMask
from .obvious_change_sentinel import mask_iou
from .sam3_identity_location import (
    FeatureDescriptorBatch,
    associate_identities,
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
