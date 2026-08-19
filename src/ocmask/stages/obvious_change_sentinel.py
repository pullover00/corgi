"""High-precision early detection of visually obvious added/removed objects.

This module is an isolated experiment, not part of the production GOLDILOCS
pipeline.  It compares object proposals from the two *real* RGB images before
the geometry-clean-render classifier is used.  The detector is intentionally
selective: uncertain identity, a possible moved/replaced counterpart, missing
geometry, occlusion, or an inconclusive targeted search all cause abstention.

The functions here never read ground truth.  They are kept model-agnostic so a
runner can use cached SAM3 proposals/features and either SAM2 or SAM3.1 for the
targeted presence check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image

from ..geometry import project_points
from ..masks import mask_iou
from ..types import Label, ObjectMask
from .sam3_identity_location import FeatureDescriptorBatch


EndpointView = Literal["source", "target"]
IdentityBand = Literal[
    "present", "ambiguous", "different", "invalid", "no_comparison"
]


@dataclass(frozen=True)
class CandidateRecord:
    """Eligibility and deterministic duplicate handling for one proposal."""

    object_index: int
    proposal_id: int
    area: int
    area_fraction: float
    bbox_width_fraction: float
    bbox_height_fraction: float
    predicted_iou: float
    stability_score: float | None
    feature_valid: bool
    touches_frame_border: bool
    selected: bool
    duplicate_of: int | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSelection:
    """Large decision candidates plus the broader global identity-search pool."""

    selected_indices: tuple[int, ...]
    search_indices: tuple[int, ...]
    records: tuple[CandidateRecord, ...]

    @property
    def selected_ids(self) -> tuple[int, ...]:
        by_index = {record.object_index: record.proposal_id for record in self.records}
        return tuple(by_index[index] for index in self.selected_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_indices": list(self.selected_indices),
            "selected_ids": list(self.selected_ids),
            "search_indices": list(self.search_indices),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ProjectedMaskBatch:
    """Masks projected through the winning samples of one shared z-buffer."""

    masks: tuple[np.ndarray, ...]
    coverage: np.ndarray
    depth: np.ndarray
    owner_source_index: np.ndarray
    valid_source_point_count: int
    winner_count: int


@dataclass(frozen=True)
class FeatureAbsenceEvidence:
    """Global identity evidence for one possible absent object."""

    candidate_index: int
    candidate_id: int
    identity_band: IdentityBand
    absence_supported: bool
    comparable_count: int
    best_opposite_index: int | None
    best_opposite_id: int | None
    best_cosine: float | None
    same_threshold: float
    different_threshold: float
    replacement_veto: bool
    replacement_opposite_ids: tuple[int, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometrySupportEvidence:
    """How much of an object location is observed and not occluded."""

    area: int
    render_supported_pixels: int
    unoccluded_pixels: int
    occluded_pixels: int
    render_support_fraction: float
    unoccluded_support_fraction: float
    threshold: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SentinelDecision:
    """The O1/O2/O3 decision funnel for one endpoint proposal."""

    view: EndpointView
    proposal_id: int
    label: Label
    o1_large_candidate: bool
    o2_identity_absent: bool
    o3_verified_absent: bool
    targeted_presence: bool | None
    output_area_ratio: float | None
    feature: FeatureAbsenceEvidence | None
    geometry: GeometrySupportEvidence | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["label"] = self.label.name.lower()
        return value


@dataclass(frozen=True)
class EndpointSentinelResult:
    """Objects surviving successive sentinel stages for one temporal endpoint."""

    o1_objects: tuple[ObjectMask, ...]
    o2_objects: tuple[ObjectMask, ...]
    o3_objects: tuple[ObjectMask, ...]
    decisions: tuple[SentinelDecision, ...]

    def funnel(self) -> dict[str, int]:
        return {
            "selected_large": len(self.o1_objects),
            "identity_absent": len(self.o2_objects),
            "verified_absent": len(self.o3_objects),
            "identity_or_replacement_abstained": sum(
                decision.o1_large_candidate and not decision.o2_identity_absent
                for decision in self.decisions
            ),
            "geometry_or_targeted_abstained": sum(
                decision.o2_identity_absent and not decision.o3_verified_absent
                for decision in self.decisions
            ),
        }


def _proposal_id(obj: ObjectMask) -> int:
    """Return the stable proposal ID required for reorder determinism."""

    if "automatic_proposal_id" not in obj.metadata:
        raise ValueError("every sentinel proposal requires automatic_proposal_id")
    return int(obj.metadata["automatic_proposal_id"])


def _validate_feature_batch(
    objects: Sequence[ObjectMask], features: FeatureDescriptorBatch
) -> None:
    count = len(objects)
    if features.vectors.ndim != 2 or features.vectors.shape[0] != count:
        raise ValueError("feature descriptor count differs from proposal count")
    if features.valid.shape != (count,) or features.effective_cells.shape != (count,):
        raise ValueError("feature descriptor metadata differs from proposal count")


def _touches_border(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    return bool(
        np.any(mask[0])
        or np.any(mask[-1])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def select_large_candidates(
    objects: Sequence[ObjectMask],
    features: FeatureDescriptorBatch,
    image_shape: tuple[int, int],
    *,
    minimum_area_fraction: float = 0.01,
    minimum_bbox_side_fraction: float = 0.02,
    minimum_mask_area: int = 32,
    minimum_predicted_iou: float = 0.8,
    minimum_stability_score: float = 0.8,
    duplicate_iou: float = 0.8,
    duplicate_containment: float = 0.8,
    reject_frame_border: bool = True,
) -> CandidateSelection:
    """Select large, high-quality endpoints without pruning identity evidence.

    Duplicate suppression is applied only to large *decision candidates*.
    ``search_indices`` deliberately retains nested and duplicate proposals:
    additional opposite-view proposals can only veto a false absence claim.
    """

    objects = list(objects)
    _validate_feature_batch(objects, features)
    height, width = map(int, image_shape)
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    if not 0 <= minimum_area_fraction <= 1:
        raise ValueError("minimum_area_fraction must be in [0, 1]")
    if not 0 <= minimum_bbox_side_fraction <= 1:
        raise ValueError("minimum_bbox_side_fraction must be in [0, 1]")
    if minimum_mask_area < 1:
        raise ValueError("minimum_mask_area must be positive")
    for name, value in (
        ("minimum_predicted_iou", minimum_predicted_iou),
        ("minimum_stability_score", minimum_stability_score),
        ("duplicate_iou", duplicate_iou),
        ("duplicate_containment", duplicate_containment),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    ids = [_proposal_id(obj) for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("sentinel proposal IDs must be unique")
    masks = [np.asarray(obj.mask, dtype=bool) for obj in objects]
    if any(mask.shape != (height, width) for mask in masks):
        raise ValueError("proposal masks must match image_shape")

    base: dict[int, dict[str, Any]] = {}
    search_indices: list[int] = []
    large_indices: list[int] = []
    pixels = height * width
    for index, (obj, mask, proposal_id) in enumerate(zip(objects, masks, ids)):
        area = int(mask.sum())
        fraction = area / pixels
        ys, xs = np.nonzero(mask)
        bbox_width_fraction = (
            float(xs.max() - xs.min() + 1) / width if len(xs) else 0.0
        )
        bbox_height_fraction = (
            float(ys.max() - ys.min() + 1) / height if len(ys) else 0.0
        )
        stability_value = obj.metadata.get("stability_score")
        stability = None if stability_value is None else float(stability_value)
        quality_reasons: list[str] = []
        if area < minimum_mask_area:
            quality_reasons.append("below_minimum_mask_area")
        if float(obj.score) < minimum_predicted_iou:
            quality_reasons.append("low_predicted_iou")
        if stability is None:
            quality_reasons.append("missing_stability_score")
        elif stability < minimum_stability_score:
            quality_reasons.append("low_stability_score")
        if not quality_reasons and bool(features.valid[index]):
            search_indices.append(index)

        reasons = list(quality_reasons)
        if fraction < minimum_area_fraction:
            reasons.append("below_minimum_area_fraction")
        if min(bbox_width_fraction, bbox_height_fraction) < minimum_bbox_side_fraction:
            reasons.append("below_minimum_bbox_side_fraction")
        border = _touches_border(mask)
        if reject_frame_border and border:
            reasons.append("touches_frame_border")
        if not reasons:
            large_indices.append(index)
        base[index] = {
            "object_index": index,
            "proposal_id": proposal_id,
            "area": area,
            "area_fraction": fraction,
            "bbox_width_fraction": bbox_width_fraction,
            "bbox_height_fraction": bbox_height_fraction,
            "predicted_iou": float(obj.score),
            "stability_score": stability,
            "feature_valid": bool(features.valid[index]),
            "touches_frame_border": border,
            "selected": False,
            "duplicate_of": None,
            "reasons": reasons,
        }

    # Rank quality deterministically, then report final selections in proposal
    # ID order so input list order never affects serialized output.
    ranked = sorted(
        large_indices,
        key=lambda index: (
            -float(objects[index].score),
            -float(objects[index].metadata["stability_score"]),
            -int(masks[index].sum()),
            ids[index],
        ),
    )
    retained: list[int] = []
    for index in ranked:
        duplicates = []
        for other in retained:
            intersection = int(np.logical_and(masks[index], masks[other]).sum())
            containment = intersection / max(
                min(int(masks[index].sum()), int(masks[other].sum())), 1
            )
            if (
                mask_iou(masks[index], masks[other]) >= duplicate_iou
                or containment >= duplicate_containment
            ):
                duplicates.append(other)
        if duplicates:
            duplicate_of = min(duplicates, key=lambda other: ids[other])
            base[index]["duplicate_of"] = ids[duplicate_of]
            base[index]["reasons"].append("duplicate_large_candidate")
        else:
            retained.append(index)
            base[index]["selected"] = True

    records = tuple(
        CandidateRecord(
            **{
                **base[index],
                "reasons": tuple(base[index]["reasons"]),
            }
        )
        for index in sorted(range(len(objects)), key=lambda item: ids[item])
    )
    selected = tuple(sorted(retained, key=lambda index: ids[index]))
    search = tuple(sorted(search_indices, key=lambda index: ids[index]))
    return CandidateSelection(selected, search, records)


def project_masks_with_zbuffer(
    points: np.ndarray,
    masks: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    image_shape: tuple[int, int],
    *,
    source_valid: np.ndarray | None = None,
    minimum_depth: float = 1e-6,
) -> ProjectedMaskBatch:
    """Project masks using only points that own a target z-buffer pixel.

    The z-buffer is shared by the complete source pointmap, not constructed
    independently for each mask.  Consequently a point hidden behind another
    source point cannot make an object appear visible.  Overlapping/nested
    proposal masks may still share a winning point, which is desirable for a
    conservative presence veto.
    """

    points = np.asarray(points)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape H x W x 3")
    source_shape = points.shape[:2]
    binary_masks = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if any(mask.shape != source_shape for mask in binary_masks):
        raise ValueError("source masks and pointmap must share a shape")
    if source_valid is None:
        allowed = np.ones(source_shape, dtype=bool)
    else:
        allowed = np.asarray(source_valid, dtype=bool)
        if allowed.shape != source_shape:
            raise ValueError("source_valid and pointmap must share a shape")

    height, width = map(int, image_shape)
    projection = project_points(
        points, intrinsics, world_to_camera, (height, width), minimum_depth
    )
    flat_points = points.reshape(-1, 3)
    finite = np.isfinite(flat_points).all(axis=1)
    valid = projection.valid & finite & allowed.reshape(-1)
    valid_indices = np.flatnonzero(valid)
    coverage = np.zeros((height, width), dtype=bool)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int64)
    if len(valid_indices):
        uv = projection.uv[valid_indices]
        z = projection.depth[valid_indices]
        pixel = uv[:, 1] * width + uv[:, 0]
        # Same stable tie-break as geometry.render_points: nearest depth, then
        # the lower flattened source-point index for equal-depth collisions.
        order = np.lexsort((valid_indices, z, pixel))
        sorted_pixels = pixel[order]
        first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
        winning_order = order[first]
        winners = valid_indices[winning_order]
        winning_uv = uv[winning_order]
        winning_depth = z[winning_order]
        yy, xx = winning_uv[:, 1], winning_uv[:, 0]
        coverage[yy, xx] = True
        depth[yy, xx] = winning_depth.astype(np.float32)
        owner[yy, xx] = winners

    projected: list[np.ndarray] = []
    for mask in binary_masks:
        output = np.zeros((height, width), dtype=bool)
        if np.any(coverage):
            output[coverage] = mask.reshape(-1)[owner[coverage]]
        projected.append(output)
    return ProjectedMaskBatch(
        masks=tuple(projected),
        coverage=coverage,
        depth=depth,
        owner_source_index=owner,
        valid_source_point_count=int(len(valid_indices)),
        winner_count=int(coverage.sum()),
    )


def geometry_support(
    mask: np.ndarray,
    opposing_depth: np.ndarray,
    observed_depth: np.ndarray,
    opposing_coverage: np.ndarray,
    *,
    visibility_threshold: float = 0.8,
    depth_epsilon: float = 1e-3,
) -> GeometrySupportEvidence:
    """Require a location to be rendered and not hidden by a nearer surface.

    ``opposing_depth`` is the other time point rendered into the candidate's
    camera; ``observed_depth`` is the candidate time's own depth.  A rendered
    surface nearer than the candidate-time surface means the location was
    occluded in the other observation and absence cannot be established.
    """

    binary = np.asarray(mask, dtype=bool)
    opposing = np.asarray(opposing_depth, dtype=np.float32)
    observed = np.asarray(observed_depth, dtype=np.float32)
    coverage = np.asarray(opposing_coverage, dtype=bool)
    if opposing.shape != binary.shape or observed.shape != binary.shape or coverage.shape != binary.shape:
        raise ValueError("geometry support arrays must share the candidate-mask shape")
    if not 0 <= visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be in [0, 1]")
    if depth_epsilon < 0:
        raise ValueError("depth_epsilon must be non-negative")
    area = int(binary.sum())
    if not area:
        return GeometrySupportEvidence(
            0, 0, 0, 0, 0.0, 0.0, float(visibility_threshold), False, ("empty_mask",)
        )

    finite = np.isfinite(opposing) & np.isfinite(observed) & (opposing > 0) & (observed > 0)
    rendered = binary & coverage & finite
    unoccluded = rendered & (opposing >= observed - float(depth_epsilon))
    occluded = rendered & ~unoccluded
    rendered_count = int(rendered.sum())
    unoccluded_count = int(unoccluded.sum())
    render_fraction = rendered_count / area
    unoccluded_fraction = unoccluded_count / area
    reasons: list[str] = []
    if render_fraction < visibility_threshold:
        reasons.append("insufficient_valid_render_support")
    if unoccluded_fraction < visibility_threshold:
        reasons.append("insufficient_unoccluded_support")
    return GeometrySupportEvidence(
        area=area,
        render_supported_pixels=rendered_count,
        unoccluded_pixels=unoccluded_count,
        occluded_pixels=int(occluded.sum()),
        render_support_fraction=float(render_fraction),
        unoccluded_support_fraction=float(unoccluded_fraction),
        threshold=float(visibility_threshold),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_y, first_x = np.nonzero(first)
    second_y, second_x = np.nonzero(second)
    if not len(first_x) or not len(second_x):
        return math.inf
    height, width = first.shape
    return math.hypot(
        float(first_x.mean() - second_x.mean()) / max(width - 1, 1),
        float(first_y.mean() - second_y.mean()) / max(height - 1, 1),
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return None
    value = float(np.dot(first, second) / denominator)
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else None


def evaluate_global_feature_absence(
    candidate_index: int,
    objects: Sequence[ObjectMask],
    features: FeatureDescriptorBatch,
    opposite_objects: Sequence[ObjectMask],
    opposite_features: FeatureDescriptorBatch,
    opposite_aligned_masks: Sequence[np.ndarray],
    *,
    opposite_search_indices: Sequence[int] | None = None,
    same_threshold: float = 0.65,
    different_margin: float = 0.10,
    identity_area_ratio_bounds: tuple[float, float] = (0.25, 4.0),
    replacement_minimum_iou: float = 0.30,
    replacement_maximum_centroid_distance: float = 0.10,
) -> FeatureAbsenceEvidence:
    """Test identity globally and use same-place disagreement as a veto.

    The best similarity is computed over *all* eligible opposite proposals,
    not merely a one-to-one assignment.  Nested proposals therefore protect
    against declaring a still-present or moved object absent.
    """

    objects = list(objects)
    opposite_objects = list(opposite_objects)
    _validate_feature_batch(objects, features)
    _validate_feature_batch(opposite_objects, opposite_features)
    if not 0 <= candidate_index < len(objects):
        raise IndexError("candidate_index is out of range")
    if len(opposite_aligned_masks) != len(opposite_objects):
        raise ValueError("opposite aligned-mask count differs from proposal count")
    candidate_mask = np.asarray(objects[candidate_index].mask, dtype=bool)
    aligned = [np.asarray(mask, dtype=bool) for mask in opposite_aligned_masks]
    if any(mask.shape != candidate_mask.shape for mask in aligned):
        raise ValueError("opposite aligned masks must share the candidate frame")
    low_area, high_area = map(float, identity_area_ratio_bounds)
    if low_area <= 0 or high_area < low_area:
        raise ValueError("invalid identity area-ratio bounds")
    if not -1 <= same_threshold <= 1:
        raise ValueError("same_threshold must be in [-1, 1]")
    if different_margin < 0:
        raise ValueError("different_margin must be non-negative")
    different_threshold = float(same_threshold) - float(different_margin)
    candidate_id = _proposal_id(objects[candidate_index])
    if not bool(features.valid[candidate_index]):
        return FeatureAbsenceEvidence(
            candidate_index,
            candidate_id,
            "invalid",
            False,
            0,
            None,
            None,
            None,
            float(same_threshold),
            different_threshold,
            False,
            (),
            ("invalid_candidate_descriptor",),
        )

    if opposite_search_indices is None:
        search_indices = list(range(len(opposite_objects)))
    else:
        search_indices = [int(index) for index in opposite_search_indices]
    if len(search_indices) != len(set(search_indices)):
        raise ValueError("opposite_search_indices contains duplicates")
    if any(index < 0 or index >= len(opposite_objects) for index in search_indices):
        raise IndexError("opposite_search_indices contains an out-of-range index")

    candidate_area = int(candidate_mask.sum())
    comparisons: list[tuple[float, int, int]] = []
    for opposite_index in search_indices:
        if not bool(opposite_features.valid[opposite_index]):
            continue
        opposite_area = int(np.asarray(opposite_objects[opposite_index].mask, bool).sum())
        ratio = opposite_area / max(candidate_area, 1)
        if not low_area <= ratio <= high_area:
            continue
        similarity = _cosine(
            features.vectors[candidate_index], opposite_features.vectors[opposite_index]
        )
        if similarity is None:
            continue
        comparisons.append((similarity, _proposal_id(opposite_objects[opposite_index]), opposite_index))

    if not comparisons:
        return FeatureAbsenceEvidence(
            candidate_index,
            candidate_id,
            "no_comparison",
            False,
            0,
            None,
            None,
            None,
            float(same_threshold),
            different_threshold,
            False,
            (),
            ("no_comparable_opposite_descriptor",),
        )

    # Cosine descending, stable proposal ID ascending is invariant to list order.
    best_cosine, best_id, best_index = sorted(
        comparisons, key=lambda value: (-value[0], value[1])
    )[0]
    if best_cosine >= same_threshold or np.isclose(
        best_cosine, same_threshold, atol=1e-6, rtol=0
    ):
        band: IdentityBand = "present"
    elif best_cosine <= different_threshold or np.isclose(
        best_cosine, different_threshold, atol=1e-6, rtol=0
    ):
        band = "different"
    else:
        band = "ambiguous"

    comparison_by_index = {index: cosine for cosine, _, index in comparisons}
    replacement_ids: list[int] = []
    for opposite_index in sorted(comparison_by_index, key=lambda index: _proposal_id(opposite_objects[index])):
        if comparison_by_index[opposite_index] > different_threshold and not np.isclose(
            comparison_by_index[opposite_index],
            different_threshold,
            atol=1e-6,
            rtol=0,
        ):
            continue
        aligned_mask = aligned[opposite_index]
        aligned_area = int(aligned_mask.sum())
        aligned_ratio = aligned_area / max(candidate_area, 1)
        if not low_area <= aligned_ratio <= high_area:
            continue
        if mask_iou(candidate_mask, aligned_mask) < replacement_minimum_iou:
            continue
        if _centroid_distance(candidate_mask, aligned_mask) > replacement_maximum_centroid_distance:
            continue
        replacement_ids.append(_proposal_id(opposite_objects[opposite_index]))

    reasons: list[str] = []
    if band == "present":
        reasons.append("same_identity_present_or_moved_veto")
    elif band == "ambiguous":
        reasons.append("identity_ambiguity_abstain")
    if replacement_ids:
        reasons.append("same_place_different_identity_replacement_veto")
    supported = band == "different" and not replacement_ids
    return FeatureAbsenceEvidence(
        candidate_index=candidate_index,
        candidate_id=candidate_id,
        identity_band=band,
        absence_supported=supported,
        comparable_count=len(comparisons),
        best_opposite_index=best_index,
        best_opposite_id=best_id,
        best_cosine=float(best_cosine),
        same_threshold=float(same_threshold),
        different_threshold=different_threshold,
        replacement_veto=bool(replacement_ids),
        replacement_opposite_ids=tuple(replacement_ids),
        reasons=tuple(reasons),
    )


def _copy_sentinel_object(
    obj: ObjectMask,
    output_mask: np.ndarray,
    label: Label,
    decision: SentinelDecision,
) -> ObjectMask:
    metadata = dict(obj.metadata)
    metadata.update(
        {
            "sentinel_view": decision.view,
            "sentinel_proposal_id": decision.proposal_id,
            "sentinel_decision": decision.to_dict(),
        }
    )
    return ObjectMask(
        mask=np.asarray(output_mask, dtype=bool).copy(),
        score=float(obj.score),
        label=label,
        source=f"obvious_change_sentinel_{label.name.lower()}",
        metadata=metadata,
    )


def evaluate_endpoint_candidates(
    view: EndpointView,
    label: Label,
    objects: Sequence[ObjectMask],
    features: FeatureDescriptorBatch,
    selection: CandidateSelection,
    opposite_objects: Sequence[ObjectMask],
    opposite_features: FeatureDescriptorBatch,
    opposite_aligned_masks: Sequence[np.ndarray],
    candidate_output_masks: Sequence[np.ndarray],
    opposing_depth: np.ndarray,
    observed_depth: np.ndarray,
    opposing_coverage: np.ndarray,
    targeted_presence: Mapping[int, bool | None],
    *,
    same_threshold: float,
    opposite_search_indices: Sequence[int] | None = None,
    different_margin: float = 0.10,
    identity_area_ratio_bounds: tuple[float, float] = (0.25, 4.0),
    replacement_minimum_iou: float = 0.30,
    replacement_maximum_centroid_distance: float = 0.10,
    visibility_threshold: float = 0.80,
    depth_epsilon: float = 1e-3,
    output_area_ratio_bounds: tuple[float, float] = (0.25, 4.0),
) -> EndpointSentinelResult:
    """Evaluate the O1 large, O2 identity, and O3 verified-absence stages.

    ``targeted_presence[id]`` is tri-state: ``True`` found the object and
    vetoes absence, ``False`` is an executed search that found no object, and
    ``None`` (or a missing key) is inconclusive and therefore abstains.
    """

    if view not in {"source", "target"}:
        raise ValueError("view must be 'source' or 'target'")
    expected_label = Label.REMOVED if view == "source" else Label.ADDED
    if label != expected_label:
        raise ValueError(f"{view} endpoint must emit {expected_label.name.lower()}")
    objects = list(objects)
    if len(candidate_output_masks) != len(objects):
        raise ValueError("candidate output-mask count differs from proposals")
    output_masks = [np.asarray(mask, dtype=bool) for mask in candidate_output_masks]
    low_output, high_output = map(float, output_area_ratio_bounds)
    if low_output <= 0 or high_output < low_output:
        raise ValueError("invalid output area-ratio bounds")

    o1: list[ObjectMask] = []
    o2: list[ObjectMask] = []
    o3: list[ObjectMask] = []
    decisions: list[SentinelDecision] = []
    for index in selection.selected_indices:
        obj = objects[index]
        proposal_id = _proposal_id(obj)
        input_area = int(np.asarray(obj.mask, bool).sum())
        output_area = int(output_masks[index].sum())
        ratio = output_area / max(input_area, 1)
        o1_pass = output_area > 0 and low_output <= ratio <= high_output
        feature_evidence: FeatureAbsenceEvidence | None = None
        geometry_evidence: GeometrySupportEvidence | None = None
        o2_pass = False
        o3_pass = False
        reasons: list[str] = []
        if not o1_pass:
            reasons.append("invalid_or_distorted_output_projection")
        else:
            feature_evidence = evaluate_global_feature_absence(
                index,
                objects,
                features,
                opposite_objects,
                opposite_features,
                opposite_aligned_masks,
                opposite_search_indices=opposite_search_indices,
                same_threshold=same_threshold,
                different_margin=different_margin,
                identity_area_ratio_bounds=identity_area_ratio_bounds,
                replacement_minimum_iou=replacement_minimum_iou,
                replacement_maximum_centroid_distance=replacement_maximum_centroid_distance,
            )
            o2_pass = feature_evidence.absence_supported
            if not o2_pass:
                reasons.extend(feature_evidence.reasons)
            else:
                geometry_evidence = geometry_support(
                    obj.mask,
                    opposing_depth,
                    observed_depth,
                    opposing_coverage,
                    visibility_threshold=visibility_threshold,
                    depth_epsilon=depth_epsilon,
                )
                if not geometry_evidence.passed:
                    reasons.extend(geometry_evidence.reasons)
                presence = targeted_presence.get(proposal_id)
                if presence is True:
                    reasons.append("targeted_presence_found_veto")
                elif presence is None:
                    reasons.append("targeted_presence_unknown_abstain")
                o3_pass = geometry_evidence.passed and presence is False

        # Construct the immutable decision before copying it into object
        # metadata.  O1/O2/O3 lists are deliberately nested by construction.
        decision = SentinelDecision(
            view=view,
            proposal_id=proposal_id,
            label=label,
            o1_large_candidate=o1_pass,
            o2_identity_absent=o2_pass,
            o3_verified_absent=o3_pass,
            targeted_presence=targeted_presence.get(proposal_id),
            output_area_ratio=float(ratio) if input_area else None,
            feature=feature_evidence,
            geometry=geometry_evidence,
            reasons=tuple(dict.fromkeys(reasons)),
        )
        decisions.append(decision)
        if o1_pass:
            o1.append(_copy_sentinel_object(obj, output_masks[index], label, decision))
        if o2_pass:
            o2.append(_copy_sentinel_object(obj, output_masks[index], label, decision))
        if o3_pass:
            o3.append(_copy_sentinel_object(obj, output_masks[index], label, decision))

    return EndpointSentinelResult(tuple(o1), tuple(o2), tuple(o3), tuple(decisions))


def suppress_added_removed_collisions(
    added: Sequence[ObjectMask],
    removed: Sequence[ObjectMask],
    *,
    minimum_iou: float = 0.10,
) -> tuple[list[ObjectMask], list[ObjectMask], list[dict[str, Any]]]:
    """Symmetrically abstain on potential replacement pairs.

    No greedy direction is used: every endpoint participating in any
    above-threshold added/removed overlap is removed.  Swapping time therefore
    exchanges the two retained sets rather than changing their membership.
    """

    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be in [0, 1]")
    added = list(added)
    removed = list(removed)
    drop_added: set[int] = set()
    drop_removed: set[int] = set()
    records: list[dict[str, Any]] = []
    for added_index, added_obj in enumerate(added):
        for removed_index, removed_obj in enumerate(removed):
            first = np.asarray(added_obj.mask, bool)
            second = np.asarray(removed_obj.mask, bool)
            if first.shape != second.shape:
                raise ValueError("added and removed masks must share an output frame")
            overlap = mask_iou(first, second)
            if overlap < minimum_iou:
                continue
            drop_added.add(added_index)
            drop_removed.add(removed_index)
            records.append(
                {
                    "added_proposal_id": int(
                        added_obj.metadata["sentinel_proposal_id"]
                        if "sentinel_proposal_id" in added_obj.metadata
                        else _proposal_id(added_obj)
                    ),
                    "removed_proposal_id": int(
                        removed_obj.metadata["sentinel_proposal_id"]
                        if "sentinel_proposal_id" in removed_obj.metadata
                        else _proposal_id(removed_obj)
                    ),
                    "iou": float(overlap),
                    "decision": "abstain_potential_replacement",
                }
            )
    return (
        [obj for index, obj in enumerate(added) if index not in drop_added],
        [obj for index, obj in enumerate(removed) if index not in drop_removed],
        records,
    )


def _native_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.shape == shape:
        return binary
    return np.asarray(
        Image.fromarray(binary.astype(np.uint8)).resize(
            shape[::-1], Image.Resampling.NEAREST
        ),
        dtype=bool,
    )


def compose_sentinel(
    parent_labels: np.ndarray,
    added: Sequence[ObjectMask],
    removed: Sequence[ObjectMask],
) -> tuple[np.ndarray, dict[str, int]]:
    """Promote only parent-UNCHANGED pixels; abstain on class conflicts."""

    parent = np.asarray(parent_labels, dtype=np.uint8)
    if parent.ndim != 2:
        raise ValueError("parent_labels must be two-dimensional")
    shape = parent.shape
    added_union = np.zeros(shape, dtype=bool)
    removed_union = np.zeros(shape, dtype=bool)
    for obj in added:
        if obj.label != Label.ADDED:
            raise ValueError("added sentinel list contains another label")
        added_union |= _native_mask(obj.mask, shape)
    for obj in removed:
        if obj.label != Label.REMOVED:
            raise ValueError("removed sentinel list contains another label")
        removed_union |= _native_mask(obj.mask, shape)

    conflict = added_union & removed_union
    editable = parent == int(Label.UNCHANGED)
    add_pixels = added_union & ~conflict & editable
    remove_pixels = removed_union & ~conflict & editable
    output = parent.copy()
    output[add_pixels] = int(Label.ADDED)
    output[remove_pixels] = int(Label.REMOVED)
    # A sentinel may only add binary support; pre-existing semantic pixels are
    # an immutable parent prediction.
    if not np.array_equal(output[~editable], parent[~editable]):
        raise AssertionError("sentinel modified an existing parent change")
    if np.any((parent != int(Label.UNCHANGED)) & (output == int(Label.UNCHANGED))):
        raise AssertionError("sentinel removed parent binary support")
    return output, {
        "added_objects": len(added),
        "removed_objects": len(removed),
        "added_pixels": int(add_pixels.sum()),
        "removed_pixels": int(remove_pixels.sum()),
        "cross_class_conflict_pixels_abstained": int(conflict.sum()),
        "parent_changed_pixels_protected": int(
            ((added_union | removed_union) & ~editable).sum()
        ),
    }
