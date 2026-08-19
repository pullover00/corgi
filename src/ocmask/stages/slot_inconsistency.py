"""Object-slot inconsistency evidence for target-frame replacement masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from ..masks import mask_iou
from ..types import ObjectMask
from .branch_b2 import ObjectHypothesis


@dataclass(frozen=True)
class SlotSettings:
    minimum_aligned_iou: float = 0.04
    minimum_containment: float = 0.20
    maximum_centroid_distance_fraction: float = 0.08
    minimum_slot_score: float = 0.25
    minimum_target_to_source_area_ratio: float = 0.25
    maximum_target_to_source_area_ratio: float = 4.0
    minimum_patch_cells: int = 4
    same_patch_cosine: float = 0.68
    minimum_lost_identity_fraction: float = 0.35
    minimum_target_slot_coverage: float = 0.20
    sam_mismatch_maximum: float = 0.72
    dino_mismatch_maximum: float = 0.58
    color_mismatch_maximum: float = 0.35
    minimum_mismatch_signals: int = 2
    minimum_cleanup_mismatch_signals: int = 3
    minimum_cleanup_aligned_iou: float = 0.45
    minimum_cleanup_containment: float = 0.60
    minimum_cleanup_area_ratio: float = 0.65
    maximum_cleanup_area_ratio: float = 1.50
    minimum_cleanup_slot_score: float = 0.45
    minimum_cleanup_added_removed_fraction: float = 0.50
    fragmented_surface_maximum_compactness: float = 0.25
    fragmented_surface_maximum_bbox_fill: float = 0.25
    fragmented_surface_minimum_components: int = 5
    thin_surface_minimum_aspect_ratio: float = 3.5
    thin_surface_maximum_half_thickness: float = 5.0
    thin_surface_maximum_compactness: float = 0.45
    compact_cleanup_minimum_compactness: float = 0.80
    compact_cleanup_minimum_bbox_fill: float = 0.70
    compact_cleanup_maximum_aspect_ratio: float = 2.0
    compact_cleanup_maximum_components: int = 2
    asymmetric_cleanup_minimum_containment: float = 0.85
    asymmetric_cleanup_maximum_area_ratio: float = 0.65
    asymmetric_cleanup_minimum_slot_score: float = 0.45
    asymmetric_cleanup_maximum_centroid_distance_fraction: float = 0.04
    asymmetric_cleanup_minimum_largest_component_fraction: float = 0.80
    weak_identity_maximum_target_components: int = 6
    weak_identity_minimum_added_removed_fraction: float = 0.08
    sam_rescue_minimum: float = 0.84
    dino_rescue_minimum: float = 0.74
    retained_identity_rescue_fraction: float = 0.62
    confident_moved_track_iou_minimum: float = 0.70
    confident_moved_color_minimum: float = 0.45
    elsewhere_sam_minimum: float = 0.82
    elsewhere_dino_minimum: float = 0.70
    maximum_elsewhere_aligned_iou: float = 0.05
    depth_ownership_tie_margin_m: float = 0.0
    depth_ownership_minimum_pixels: int = 8
    depth_ownership_minimum_fraction: float = 0.25
    plausible_object_minimum_compactness: float = 0.45
    plausible_fragment_minimum_compactness: float = 0.33
    plausible_fragment_minimum_aligned_iou: float = 0.45
    plausible_object_minimum_slot_score: float = 0.33
    companion_maximum_gap_pixels: float = 6.0
    companion_minimum_area_ratio: float = 0.50
    companion_maximum_area_ratio: float = 2.0
    companion_maximum_depth_difference_m: float = 0.08
    companion_minimum_sam_cosine: float = 0.80
    companion_minimum_dino_cosine: float = 0.80
    companion_minimum_color_intersection: float = 0.50
    consensus_minimum_changed_fraction: float = 0.50
    consensus_minimum_dominant_fraction: float = 0.75
    consensus_minimum_pixels: int = 32
    arbitration_minimum_changed_fraction: float = 0.75
    arbitration_minimum_pixels: int = 32
    arbitration_minimum_compactness: float = 0.45
    arbitration_minimum_bbox_fill: float = 0.35
    arbitration_maximum_components: int = 6
    arbitration_minimum_largest_component_fraction: float = 0.80
    arbitration_minimum_dominant_fraction: float = 0.75
    arbitration_minimum_added_removed_fraction: float = 0.08
    arbitration_minimum_replaced_fraction_with_moved: float = 0.50
    floor_distance_tolerance_m: float = 0.05
    floor_component_minimum_pixels: int = 32
    floor_component_minimum_fraction: float = 0.70


@dataclass(frozen=True)
class SlotMatch:
    source_index: int
    target_index: int
    aligned_iou: float
    containment: float
    track_iou: float
    centroid_distance_fraction: float
    area_ratio: float
    score: float
    eligible: bool

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchRetention:
    source_cells: int
    target_cells: int
    overlapping_cells: int
    retained_cells: int
    lost_cells: int
    retained_fraction: float | None
    lost_fraction: float | None
    median_aligned_cosine: float | None
    valid: bool

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SlotDecision:
    verdict: str  # replaced | moved_elsewhere | same_identity | abstain
    mismatch_signals: tuple[str, ...]
    rescue_signals: tuple[str, ...]
    reasons: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaskGeometry:
    area_pixels: int
    image_fraction: float
    bbox_height: int
    bbox_width: int
    aspect_ratio: float
    bbox_fill: float
    compactness: float
    maximum_half_thickness: float
    component_count: int
    largest_component_fraction: float

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.asarray([xs.mean(), ys.mean()], float)


def _proposal_mask(value: ObjectMask | ObjectHypothesis) -> np.ndarray:
    return np.asarray(value.mask, bool)


def mask_geometry(mask: np.ndarray) -> MaskGeometry:
    """Describe whether a target mask resembles an object or a support surface."""

    value = np.asarray(mask, bool)
    area = int(value.sum())
    if not area:
        return MaskGeometry(0, 0.0, 0, 0, float("inf"), 0.0, 0.0, 0.0, 0, 0.0)
    ys, xs = np.nonzero(value)
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    aspect = max(height / max(width, 1), width / max(height, 1))
    boundary = value ^ ndimage.binary_erosion(value)
    perimeter = int(boundary.sum())
    compactness = float(4.0 * np.pi * area / max(perimeter * perimeter, 1))
    distance = ndimage.distance_transform_edt(value)
    components, count = ndimage.label(value)
    component_sizes = np.bincount(components.ravel())[1:]
    largest = int(component_sizes.max()) if len(component_sizes) else 0
    return MaskGeometry(
        area_pixels=area,
        image_fraction=area / value.size,
        bbox_height=height,
        bbox_width=width,
        aspect_ratio=float(aspect),
        bbox_fill=area / max(height * width, 1),
        compactness=compactness,
        maximum_half_thickness=float(distance.max()),
        component_count=int(count),
        largest_component_fraction=largest / area,
    )


def support_surface_evidence(
    geometry: MaskGeometry,
    *,
    settings: SlotSettings,
) -> dict[str, Any]:
    fragmented = (
        geometry.compactness <= settings.fragmented_surface_maximum_compactness
        and geometry.bbox_fill <= settings.fragmented_surface_maximum_bbox_fill
        and geometry.component_count >= settings.fragmented_surface_minimum_components
    )
    thin = (
        geometry.aspect_ratio >= settings.thin_surface_minimum_aspect_ratio
        and geometry.maximum_half_thickness <= settings.thin_surface_maximum_half_thickness
        and geometry.compactness <= settings.thin_surface_maximum_compactness
    )
    reasons = []
    if fragmented:
        reasons.append("fragmented_low_fill_support_surface")
    if thin:
        reasons.append("thin_elongated_support_surface")
    return {"is_support_surface": bool(reasons), "reasons": reasons}


def replacement_object_plausibility(
    match: SlotMatch,
    geometry: MaskGeometry,
    *,
    cleanup_eligible: bool,
    mismatch_signal_count: int | None = None,
    added_removed_fraction: float | None = None,
    settings: SlotSettings,
) -> dict[str, Any]:
    """Require one plausible target object before changing any class pixels.

    Identity mismatch says that the old appearance disappeared, but it does
    not prove that a fragmented shelf/background proposal is the replacing
    object. A target is rasterized only when it is compact, or when a mildly
    fragmented mask has strong direct alignment to the old slot. The strict
    cleanup route is already stronger than this shape gate and may pass it.
    """

    compact_object = geometry.compactness >= settings.plausible_object_minimum_compactness
    aligned_fragment = (
        geometry.compactness >= settings.plausible_fragment_minimum_compactness
        and match.aligned_iou >= settings.plausible_fragment_minimum_aligned_iou
    )
    strong_slot = match.score >= settings.plausible_object_minimum_slot_score
    reasons = []
    if not compact_object and not aligned_fragment:
        reasons.append("target_mask_is_not_one_coherent_object")
    if not strong_slot:
        reasons.append("replacement_slot_match_is_too_weak")
    weak_fragmented_identity = (
        mismatch_signal_count is not None
        and mismatch_signal_count < settings.minimum_cleanup_mismatch_signals
        and geometry.component_count > settings.weak_identity_maximum_target_components
        and (added_removed_fraction or 0.0)
        < settings.weak_identity_minimum_added_removed_fraction
    )
    if weak_fragmented_identity:
        reasons.append("two_signal_fragment_lacks_change_support")
    eligible = cleanup_eligible or not reasons
    return {
        "eligible": bool(eligible),
        "compact_object": bool(compact_object),
        "aligned_fragment": bool(aligned_fragment),
        "weak_fragmented_identity": bool(weak_fragmented_identity),
        "strong_cleanup_override": bool(cleanup_eligible),
        "reasons": [] if eligible else reasons,
    }


def replacement_companion_evidence(
    anchor_mask: np.ndarray,
    candidate_mask: np.ndarray,
    target_depth: np.ndarray,
    *,
    sam_cosine: float | None,
    dino_cosine: float | None,
    color_intersection: float | None,
    settings: SlotSettings,
) -> dict[str, Any]:
    """Decide whether an adjacent target proposal is the same replacement group.

    This is deliberately strict: the masks must touch or nearly touch, have
    compatible sizes and depths, and independently agree in SAM, DINO, and
    color. It recovers repeated neighboring objects (for example a pair of
    barrels) without expanding an accepted replacement onto its shelf.
    """

    anchor = np.asarray(anchor_mask, bool)
    candidate = np.asarray(candidate_mask, bool)
    depth = np.asarray(target_depth, float)
    if anchor.shape != candidate.shape or anchor.shape != depth.shape:
        raise ValueError("companion masks and target depth must have equal shapes")
    anchor_area, candidate_area = int(anchor.sum()), int(candidate.sum())
    if not anchor_area or not candidate_area:
        return {"eligible": False, "reasons": ["empty_companion_mask"]}
    gap_pixels = float(ndimage.distance_transform_edt(~anchor)[candidate].min())
    area_ratio = candidate_area / anchor_area
    valid_depth = np.isfinite(depth) & (depth > 0)
    anchor_valid = anchor & valid_depth
    candidate_valid = candidate & valid_depth
    anchor_depth = float(np.median(depth[anchor_valid])) if anchor_valid.any() else None
    candidate_depth = float(np.median(depth[candidate_valid])) if candidate_valid.any() else None
    depth_difference = (
        abs(candidate_depth - anchor_depth)
        if anchor_depth is not None and candidate_depth is not None else None
    )
    candidate_geometry = mask_geometry(candidate)
    reasons = []
    if gap_pixels > settings.companion_maximum_gap_pixels:
        reasons.append("companion_is_not_adjacent")
    if not settings.companion_minimum_area_ratio <= area_ratio <= settings.companion_maximum_area_ratio:
        reasons.append("companion_area_is_incompatible")
    if depth_difference is None or depth_difference > settings.companion_maximum_depth_difference_m:
        reasons.append("companion_depth_is_incompatible")
    if sam_cosine is None or sam_cosine < settings.companion_minimum_sam_cosine:
        reasons.append("companion_sam_identity_is_incompatible")
    if dino_cosine is None or dino_cosine < settings.companion_minimum_dino_cosine:
        reasons.append("companion_dino_identity_is_incompatible")
    if color_intersection is None or color_intersection < settings.companion_minimum_color_intersection:
        reasons.append("companion_color_is_incompatible")
    if candidate_geometry.compactness < settings.plausible_object_minimum_compactness:
        reasons.append("companion_is_not_a_coherent_object")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "gap_pixels": gap_pixels,
        "area_ratio": float(area_ratio),
        "anchor_median_depth_m": anchor_depth,
        "candidate_median_depth_m": candidate_depth,
        "depth_difference_m": depth_difference,
        "sam_cosine": sam_cosine,
        "dino_cosine": dino_cosine,
        "color_intersection": color_intersection,
        "candidate_compactness": candidate_geometry.compactness,
    }


def floor_aligned_added_components(
    labels: np.ndarray,
    target_points: np.ndarray,
    floor_point: np.ndarray,
    floor_normal: np.ndarray,
    *,
    distance_tolerance_m: float = 0.05,
    minimum_component_pixels: int = 32,
    minimum_floor_fraction: float = 0.70,
    added_label: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find implausible ADDED components that are predominantly floor plane.

    The decision is component-level: isolated contact pixels on a real object
    are retained, while a component that is itself mostly the fitted floor is
    removed as a unit. This avoids cutting holes into valid foreground masks.
    """

    values = np.asarray(labels, np.uint8)
    points = np.asarray(target_points, float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("target points must have H x W x 3 shape")
    if distance_tolerance_m <= 0:
        raise ValueError("floor distance tolerance must be positive")
    normal = np.asarray(floor_normal, float).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 1e-9:
        raise ValueError("floor normal must be finite and non-zero")
    normal /= normal_norm
    point = np.asarray(floor_point, float).reshape(3)
    valid = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-9)
    floor_native = valid & (np.abs((points - point) @ normal) <= distance_tolerance_m)
    if floor_native.shape != values.shape:
        floor_pixels = np.asarray(
            Image.fromarray(floor_native.astype(np.uint8)).resize(
                values.shape[::-1], Image.Resampling.NEAREST
            ),
            bool,
        )
    else:
        floor_pixels = floor_native

    components, count = ndimage.label(values == added_label)
    suppressed = np.zeros(values.shape, bool)
    component_rows = []
    for component_index in range(1, count + 1):
        component = components == component_index
        area = int(component.sum())
        aligned_pixels = int((component & floor_pixels).sum())
        aligned_fraction = aligned_pixels / max(area, 1)
        selected = (
            area >= minimum_component_pixels
            and aligned_fraction >= minimum_floor_fraction
        )
        if selected:
            suppressed |= component
        if selected or aligned_pixels:
            component_rows.append({
                "component_index": component_index,
                "area_pixels": area,
                "floor_aligned_pixels": aligned_pixels,
                "floor_aligned_fraction": float(aligned_fraction),
                "suppressed": bool(selected),
            })
    return suppressed, {
        "floor_pixel_count": int(floor_pixels.sum()),
        "added_component_count": int(count),
        "suppressed_component_count": int(sum(row["suppressed"] for row in component_rows)),
        "suppressed_pixels": int(suppressed.sum()),
        "components_with_floor_support": component_rows,
    }


def dominant_changed_class_consensus(
    labels: np.ndarray,
    object_masks: Sequence[np.ndarray],
    *,
    minimum_changed_fraction: float = 0.50,
    minimum_dominant_fraction: float = 0.75,
    minimum_pixels: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fold small base-pipeline class fragments into a strong per-object class vote.

    This is used only for object hypotheses that failed replacement
    plausibility. It cannot invent a changed object: at least half of the SAM
    mask must already be changed, and at least three quarters of those pixels
    must agree on one non-zero class. Overlapping consensus masks are resolved
    by the stronger vote, not loop order.
    """

    original = np.asarray(labels, np.uint8)
    output = original.copy()
    owner_strength = np.full(original.shape, -np.inf, float)
    owner_label = np.zeros(original.shape, np.uint8)
    rows = []
    for index, object_mask in enumerate(object_masks):
        mask = np.asarray(object_mask, bool)
        if mask.shape != original.shape:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    original.shape[::-1], Image.Resampling.NEAREST
                ),
                bool,
            )
        changed = mask & (original != 0)
        changed_pixels = int(changed.sum())
        changed_fraction = changed_pixels / max(int(mask.sum()), 1)
        values, counts = np.unique(original[changed], return_counts=True)
        dominant_label = int(values[np.argmax(counts)]) if len(values) else 0
        dominant_pixels = int(counts.max()) if len(counts) else 0
        dominant_fraction = dominant_pixels / max(changed_pixels, 1)
        eligible = (
            changed_pixels >= minimum_pixels
            and changed_fraction >= minimum_changed_fraction
            and dominant_fraction >= minimum_dominant_fraction
        )
        if eligible:
            stronger = changed & (dominant_fraction > owner_strength)
            owner_strength[stronger] = dominant_fraction
            owner_label[stronger] = dominant_label
        rows.append({
            "index": index,
            "mask_pixels": int(mask.sum()),
            "changed_pixels": changed_pixels,
            "changed_fraction": float(changed_fraction),
            "dominant_label": dominant_label,
            "dominant_pixels": dominant_pixels,
            "dominant_fraction": float(dominant_fraction),
            "eligible": bool(eligible),
        })
    rewritten = owner_strength > -np.inf
    changed_class = rewritten & (original != owner_label)
    output[rewritten] = owner_label[rewritten]
    return output, changed_class, {
        "eligible_object_count": int(sum(row["eligible"] for row in rows)),
        "rewritten_pixels": int(changed_class.sum()),
        "objects": rows,
    }


def arbitrate_target_object_classes(
    labels: np.ndarray,
    object_masks: Sequence[np.ndarray],
    *,
    settings: SlotSettings,
    object_change_evidence: Sequence[bool] | None = None,
    added_label: int = 1,
    removed_label: int = 2,
    moved_label: int = 3,
    replaced_label: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve impossible class mixtures inside coherent target objects.

    The rule is object-structural and ground-truth blind:

    * co-spatial ADDED and REMOVED evidence describes replacement;
    * a coherent object split only between MOVED and REPLACED becomes
      REPLACED when replacement already owns the majority; and
    * otherwise a conventional dominant-class fold needs 75% agreement.

    Only already-changed pixels are relabelled, so this stage cannot invent a
    changed region.  Shape and occupancy checks exclude floors, walls, shelf
    spans, and weak proposal fragments.  Overlapping hypotheses are resolved
    by rule strength and confidence rather than iteration order.
    """

    original = np.asarray(labels, np.uint8)
    if object_change_evidence is None:
        change_evidence = [True] * len(object_masks)
    else:
        change_evidence = [bool(value) for value in object_change_evidence]
        if len(change_evidence) != len(object_masks):
            raise ValueError("object change evidence must align with object masks")
    output = original.copy()
    owner_strength = np.full(original.shape, -np.inf, float)
    owner_label = np.zeros(original.shape, np.uint8)
    rows: list[dict[str, Any]] = []

    def resize(mask: np.ndarray) -> np.ndarray:
        value = np.asarray(mask, bool)
        if value.shape == original.shape:
            return value
        return np.asarray(
            Image.fromarray(value.astype(np.uint8)).resize(
                original.shape[::-1], Image.Resampling.NEAREST
            ),
            bool,
        )

    for index, native_mask in enumerate(object_masks):
        geometry = mask_geometry(native_mask)
        mask = resize(native_mask)
        changed = mask & (original != 0)
        mask_pixels = int(mask.sum())
        changed_pixels = int(changed.sum())
        changed_fraction = changed_pixels / max(mask_pixels, 1)
        counts = np.bincount(original[changed], minlength=6)
        nonzero_labels = set(np.flatnonzero(counts).tolist())
        dominant_label = int(np.argmax(counts)) if changed_pixels else 0
        dominant_pixels = int(counts[dominant_label]) if changed_pixels else 0
        dominant_fraction = dominant_pixels / max(changed_pixels, 1)
        added_fraction = int(counts[added_label]) / max(changed_pixels, 1)
        removed_fraction = int(counts[removed_label]) / max(changed_pixels, 1)
        replaced_fraction = int(counts[replaced_label]) / max(changed_pixels, 1)
        coherent = (
            geometry.compactness >= settings.arbitration_minimum_compactness
            and geometry.bbox_fill >= settings.arbitration_minimum_bbox_fill
            and geometry.component_count <= settings.arbitration_maximum_components
            and geometry.largest_component_fraction
            >= settings.arbitration_minimum_largest_component_fraction
        )
        occupied = (
            changed_pixels >= settings.arbitration_minimum_pixels
            and changed_fraction >= settings.arbitration_minimum_changed_fraction
        )
        decision = "abstain"
        selected_label = 0
        strength = -np.inf
        independently_changed = change_evidence[index]
        if coherent and occupied and independently_changed and len(nonzero_labels) >= 2:
            if (
                added_fraction >= settings.arbitration_minimum_added_removed_fraction
                and removed_fraction >= settings.arbitration_minimum_added_removed_fraction
            ):
                decision = "added_removed_implies_replaced"
                selected_label = replaced_label
                strength = 3.0 + min(added_fraction, removed_fraction)
            elif (
                nonzero_labels.issubset({moved_label, replaced_label})
                and moved_label in nonzero_labels
                and replaced_label in nonzero_labels
                and replaced_fraction
                >= settings.arbitration_minimum_replaced_fraction_with_moved
            ):
                decision = "replaced_majority_over_moved_fragment"
                selected_label = replaced_label
                strength = 2.0 + replaced_fraction
            elif dominant_fraction >= settings.arbitration_minimum_dominant_fraction:
                decision = "dominant_class_consensus"
                selected_label = dominant_label
                strength = dominant_fraction
        if selected_label:
            stronger = changed & (strength > owner_strength)
            owner_strength[stronger] = strength
            owner_label[stronger] = selected_label
        rows.append({
            "index": index,
            "mask_pixels": mask_pixels,
            "changed_pixels": changed_pixels,
            "changed_fraction": float(changed_fraction),
            "class_pixels": {
                str(label): int(counts[label]) for label in range(1, len(counts))
                if counts[label]
            },
            "dominant_label": dominant_label,
            "dominant_fraction": float(dominant_fraction),
            "selected_label": int(selected_label),
            "decision": decision,
            "coherent": bool(coherent),
            "independent_change_evidence": bool(independently_changed),
            "geometry": geometry.summary(),
        })

    rewritten = owner_strength > -np.inf
    changed_class = rewritten & (original != owner_label)
    output[rewritten] = owner_label[rewritten]
    decision_counts = {
        decision: sum(row["decision"] == decision for row in rows)
        for decision in (
            "added_removed_implies_replaced",
            "replaced_majority_over_moved_fragment",
            "dominant_class_consensus",
        )
    }
    return output, changed_class, {
        "eligible_object_count": int(sum(row["selected_label"] != 0 for row in rows)),
        "rewritten_pixels": int(changed_class.sum()),
        "decision_counts": decision_counts,
        "objects": rows,
    }


def match_object_slots(
    sources: Sequence[ObjectHypothesis],
    targets: Sequence[ObjectMask | ObjectHypothesis],
    *,
    settings: SlotSettings,
) -> tuple[SlotMatch, ...]:
    """Choose one best target proposal/hypothesis for each aligned source slot."""

    output = []
    if not targets:
        return ()
    diagonal = float(np.hypot(*sources[0].mask.shape)) if sources else 1.0
    for source_index, source in enumerate(sources):
        source_mask = np.asarray(source.mask, bool)
        source_area = int(source_mask.sum())
        candidates = []
        for target_index, target in enumerate(targets):
            target_mask = _proposal_mask(target); target_area = int(target_mask.sum())
            intersection = int(np.logical_and(source_mask, target_mask).sum())
            aligned = mask_iou(source_mask, target_mask)
            containment = intersection / max(min(source_area, target_area), 1)
            track_iou = 0.0 if source.propagated_mask is None else mask_iou(source.propagated_mask, target_mask)
            centroid = float(np.linalg.norm(_centroid(source_mask) - _centroid(target_mask)) / max(diagonal, 1.0))
            area_ratio = target_area / max(source_area, 1)
            area_compatibility = float(np.exp(-abs(np.log(max(area_ratio, 1e-6)))))
            score = 0.35 * containment + 0.25 * track_iou + 0.20 * aligned + 0.10 * area_compatibility + 0.10 * max(0.0, 1.0 - centroid / max(settings.maximum_centroid_distance_fraction, 1e-6))
            eligible = (
                (aligned >= settings.minimum_aligned_iou or containment >= settings.minimum_containment)
                and centroid <= settings.maximum_centroid_distance_fraction
                and score >= settings.minimum_slot_score
                and settings.minimum_target_to_source_area_ratio <= area_ratio
                and area_ratio <= settings.maximum_target_to_source_area_ratio
            )
            candidates.append(SlotMatch(source_index, target_index, aligned, containment, track_iou, centroid, area_ratio, score, eligible))
        eligible = [item for item in candidates if item.eligible]
        if eligible:
            output.append(max(eligible, key=lambda item: (item.score, -item.target_index)))
    return tuple(output)


def _grid_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(mask, np.float32), mode="F").resize(shape[::-1], Image.Resampling.BOX), np.float32)


def aligned_patch_retention(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    settings: SlotSettings,
) -> PatchRetention:
    """Measure how much old identity survives at the same aligned object slot."""

    source = np.asarray(source_features, np.float32); target = np.asarray(target_features, np.float32)
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("source and target feature maps must share C x H x W shape")
    _, height, width = source.shape
    source_cells = _grid_mask(source_mask, (height, width)) >= 0.25
    target_cells = _grid_mask(target_mask, (height, width)) >= 0.25
    source_flat = source.reshape(source.shape[0], -1); target_flat = target.reshape(target.shape[0], -1)
    source_flat /= np.maximum(np.linalg.norm(source_flat, axis=0, keepdims=True), 1e-12)
    target_flat /= np.maximum(np.linalg.norm(target_flat, axis=0, keepdims=True), 1e-12)
    cosine = (source_flat * target_flat).sum(axis=0).reshape(height, width)
    overlapping = source_cells & target_cells
    retained = overlapping & (cosine >= settings.same_patch_cosine)
    source_count, target_count = int(source_cells.sum()), int(target_cells.sum())
    overlap_count, retained_count = int(overlapping.sum()), int(retained.sum())
    valid = source_count >= settings.minimum_patch_cells and target_count >= settings.minimum_patch_cells
    retained_fraction = retained_count / source_count if valid else None
    lost_fraction = 1.0 - retained_fraction if retained_fraction is not None else None
    median = float(np.median(cosine[overlapping])) if overlap_count else None
    return PatchRetention(source_count, target_count, overlap_count, retained_count,
                          source_count - retained_count, retained_fraction,
                          lost_fraction, median, valid)


def pooled_descriptor(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    features = np.asarray(feature_map, np.float32); _, height, width = features.shape
    weights = _grid_mask(mask, (height, width))
    if weights.sum() < 1.0:
        return None
    vector = (features * weights[None]).reshape(features.shape[0], -1).sum(axis=1) / weights.sum()
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 1e-9 else None


def descriptor_cosine(feature_a: np.ndarray, mask_a: np.ndarray, feature_b: np.ndarray, mask_b: np.ndarray) -> float | None:
    a, b = pooled_descriptor(feature_a, mask_a), pooled_descriptor(feature_b, mask_b)
    return None if a is None or b is None else float(a @ b)


def color_intersection(image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray, bins: int = 8) -> float | None:
    def histogram(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        pixels = np.asarray(image, np.uint8)[np.asarray(mask, bool)]
        if len(pixels) < 8:
            return None
        value, _ = np.histogramdd(pixels, bins=(np.linspace(0, 256, bins + 1),) * 3)
        return value.reshape(-1) / max(float(value.sum()), 1.0)
    left, right = histogram(image_a, mask_a), histogram(image_b, mask_b)
    return None if left is None or right is None else float(np.minimum(left, right).sum())


def frontmost_replacement_ownership(
    baseline: np.ndarray,
    target_masks: Sequence[np.ndarray],
    target_depth: np.ndarray,
    source_depth_in_target: np.ndarray,
    *,
    depth_tie_margin_m: float = 0.0,
    minimum_valid_pixels: int = 8,
    minimum_valid_fraction: float = 0.25,
    unify_target_objects: bool = False,
    identity_override_target_masks: Sequence[np.ndarray] = (),
    removed_label: int = 2,
    replacement_label: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Give each overlap to the front-most object instead of paint order.

    Target proposals use their robust median target-camera depth. Existing
    REMOVED components use source geometry rendered into the target camera;
    other changed components use target depth. Exact/near ties preserve the
    existing base-pipeline owner. Unknown depth also preserves an existing
    owner.
    """

    labels = np.asarray(baseline, np.uint8)
    target_depth = np.asarray(target_depth, float)
    source_depth = np.asarray(source_depth_in_target, float)
    if target_depth.shape != source_depth.shape:
        raise ValueError("source and target depth maps must have equal shapes")
    if depth_tie_margin_m < 0:
        raise ValueError("depth_tie_margin_m must be non-negative")

    def valid_depth(depth: np.ndarray) -> np.ndarray:
        return np.isfinite(depth) & (depth > 0)

    def resize_mask(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, bool)
        if mask.shape == labels.shape:
            return mask
        return np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                labels.shape[::-1], Image.Resampling.NEAREST
            ),
            bool,
        )

    def resize_depth(depth: np.ndarray) -> np.ndarray:
        if depth.shape == labels.shape:
            return depth
        return np.asarray(
            Image.fromarray(np.asarray(depth, np.float32), mode="F").resize(
                labels.shape[::-1], Image.Resampling.NEAREST
            ),
            float,
        )

    candidate_depth = np.full(labels.shape, np.inf, float)
    candidate_union = np.zeros(labels.shape, bool)
    target_summaries = []
    unified_objects = np.zeros(labels.shape, bool)
    identity_override = np.zeros(labels.shape, bool)
    for index, target_mask in enumerate(target_masks):
        native_mask = np.asarray(target_mask, bool)
        if native_mask.shape != target_depth.shape:
            raise ValueError("target masks and depth maps must have equal shapes")
        native_valid = native_mask & valid_depth(target_depth)
        valid_count = int(native_valid.sum())
        valid_fraction = valid_count / max(int(native_mask.sum()), 1)
        median_depth = None
        resized = resize_mask(native_mask)
        candidate_union |= resized
        if valid_count >= minimum_valid_pixels and valid_fraction >= minimum_valid_fraction:
            median_depth = float(np.median(target_depth[native_valid]))
            candidate_depth[resized] = np.minimum(candidate_depth[resized], median_depth)
            if unify_target_objects:
                unified_objects |= resized
        target_summaries.append({
            "index": index,
            "area_pixels": int(native_mask.sum()),
            "valid_depth_pixels": valid_count,
            "valid_depth_fraction": float(valid_fraction),
            "median_depth_m": median_depth,
        })
    for target_mask in identity_override_target_masks:
        identity_override |= resize_mask(np.asarray(target_mask, bool))

    target_depth_resized = resize_depth(target_depth)
    source_depth_resized = resize_depth(source_depth)
    baseline_owner_depth = np.full(labels.shape, np.inf, float)
    baseline_owner_known = np.zeros(labels.shape, bool)
    component_count = 0
    for label_value in sorted(int(value) for value in np.unique(labels) if value != 0):
        components, count = ndimage.label(labels == label_value)
        component_depth = source_depth_resized if label_value == removed_label else target_depth_resized
        for component_index in range(1, count + 1):
            component = components == component_index
            valid = component & valid_depth(component_depth)
            valid_count = int(valid.sum())
            valid_fraction = valid_count / max(int(component.sum()), 1)
            component_count += 1
            if valid_count < minimum_valid_pixels or valid_fraction < minimum_valid_fraction:
                continue
            baseline_owner_depth[component] = float(np.median(component_depth[valid]))
            baseline_owner_known[component] = True

    candidate_known = np.isfinite(candidate_depth)
    existing_changed = labels != 0
    already_replaced = labels == replacement_label
    candidate_front = (
        candidate_known
        & baseline_owner_known
        & (candidate_depth + depth_tie_margin_m < baseline_owner_depth)
    )
    depth_ownership = candidate_union & candidate_known & (
        ~existing_changed | already_replaced | candidate_front
    )
    # Object consistency may absorb target-frame ADDED/MOVED class fragments,
    # but it must never overrule a closer source-frame REMOVED owner. This keeps
    # one class across the visible replacement without painting through an old
    # object that is physically in front of it.
    unified_nonremoved = (
        unified_objects
        & candidate_known
        & (labels != removed_label)
    )
    identity_override_ownership = identity_override & candidate_known
    ownership = depth_ownership | unified_nonremoved | identity_override_ownership

    # Cleanup is stricter than raster ownership: a REMOVED component may be
    # deleted only if overlap depth demonstrates that the replacement is in
    # front. Unknown depth, ties, and source-front depth protect the complete
    # connected red component, including portions outside the target mask.
    protected_removed = np.zeros(labels.shape, bool)
    removed_components, removed_count = ndimage.label(labels == removed_label)
    depth_front_removed_overlap = np.zeros(labels.shape, bool)
    source_front_removed_overlap = np.zeros(labels.shape, bool)
    unknown_removed_overlap = np.zeros(labels.shape, bool)
    for component_index in range(1, removed_count + 1):
        component = removed_components == component_index
        overlap = component & candidate_union
        if not overlap.any():
            protected_removed |= component
            continue
        valid = overlap & candidate_known & baseline_owner_known
        unknown_removed_overlap |= overlap & ~valid
        if not valid.any():
            protected_removed |= component
            continue
        target_front_votes = valid & (
            candidate_depth + depth_tie_margin_m < baseline_owner_depth
        )
        source_front_votes = valid & ~target_front_votes
        depth_front_removed_overlap |= target_front_votes
        source_front_removed_overlap |= source_front_votes
        # Require a majority of valid overlap pixels to put the replacement in
        # front; ties conservatively preserve the REMOVED component.
        if int(target_front_votes.sum()) * 2 <= int(valid.sum()):
            protected_removed |= component

    overlap = candidate_union & existing_changed & ~already_replaced
    preserved = overlap & ~ownership
    promoted_overlap = overlap & ownership
    forced_unification = overlap & unified_nonremoved & ~depth_ownership
    identity_override_removed = (
        overlap & (labels == removed_label) & identity_override_ownership
    )

    def counts_by_label(mask: np.ndarray) -> dict[str, int]:
        values, counts = np.unique(labels[mask], return_counts=True)
        return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}

    summary = {
        "candidate_pixels": int(candidate_union.sum()),
        "candidate_depth_known_pixels": int((candidate_union & candidate_known).sum()),
        "overlap_pixels": int(overlap.sum()),
        "frontmost_promoted_overlap_pixels": int((overlap & depth_ownership).sum()),
        "forced_nonremoved_unification_pixels": int(forced_unification.sum()),
        "identity_override_removed_overlap_pixels": int(identity_override_removed.sum()),
        "depth_front_removed_overlap_pixels": int(depth_front_removed_overlap.sum()),
        "source_front_removed_overlap_pixels": int(source_front_removed_overlap.sum()),
        "unknown_depth_removed_overlap_pixels": int(unknown_removed_overlap.sum()),
        "cleanup_protected_removed_pixels": int(protected_removed.sum()),
        "preserved_existing_overlap_pixels": int(preserved.sum()),
        "promoted_overlap_by_baseline_label": counts_by_label(promoted_overlap),
        "preserved_overlap_by_baseline_label": counts_by_label(preserved),
        "baseline_component_count": component_count,
        "unified_target_objects": bool(unify_target_objects),
        "unified_object_pixels": int((unified_objects & candidate_known).sum()),
        "depth_tie_margin_m": float(depth_tie_margin_m),
        "targets": target_summaries,
    }
    return ownership, protected_removed, summary


def rasterize_replacement_labels(
    baseline: np.ndarray,
    source_masks: Sequence[np.ndarray],
    target_masks: Sequence[np.ndarray],
    *,
    promote_only_changed: bool,
    trusted_target_masks: Sequence[np.ndarray] = (),
    replacement_ownership_mask: np.ndarray | None = None,
    protected_removed_mask: np.ndarray | None = None,
    force_cleanup_source_masks: Sequence[np.ndarray] = (),
    expand_connected_removed: bool = False,
    removed_expansion_radius_pixels: int = 4,
    removed_expansion_minimum_coverage: float = 0.50,
    removed_expansion_maximum_area_ratio: float = 3.0,
    removed_label: int = 2,
    replacement_label: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop old footprints and write one ownership-resolved replacement class."""

    labels = np.asarray(baseline, np.uint8).copy()
    if not source_masks and not target_masks and not trusted_target_masks:
        empty = np.zeros(labels.shape, bool)
        return labels, empty, empty
    reference_mask = (
        source_masks[0] if source_masks else
        target_masks[0] if target_masks else trusted_target_masks[0]
    )
    native_shape = np.asarray(reference_mask, bool).shape
    old_native = np.zeros(native_shape, bool)
    new_native = np.zeros(native_shape, bool)
    trusted_native = np.zeros(native_shape, bool)
    force_cleanup_native = np.zeros(native_shape, bool)
    for source_mask in source_masks:
        old_native |= np.asarray(source_mask, bool)
    for target_mask in target_masks:
        new_native |= np.asarray(target_mask, bool)
    for target_mask in trusted_target_masks:
        trusted_native |= np.asarray(target_mask, bool)
    for source_mask in force_cleanup_source_masks:
        force_cleanup_native |= np.asarray(source_mask, bool)

    def resize(mask: np.ndarray) -> np.ndarray:
        if mask.shape == labels.shape:
            return mask
        return np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                labels.shape[::-1], Image.Resampling.NEAREST
            ),
            np.uint8,
        ) > 0

    old_mask, new_mask = resize(old_native), resize(new_native)
    trusted_mask = resize(trusted_native)
    force_cleanup = resize(force_cleanup_native)
    original_changed = labels != 0
    removed_mask = labels == removed_label
    cleanup_envelope = old_mask.copy()
    if expand_connected_removed and old_mask.any() and new_mask.any():
        near_new = ndimage.binary_dilation(
            new_mask, iterations=max(int(removed_expansion_radius_pixels), 0)
        )
        components, count = ndimage.label(removed_mask)
        for component_index in range(1, count + 1):
            component = components == component_index
            source_overlap = int(np.sum(component & old_mask))
            if source_overlap == 0 or not np.any(component & near_new):
                continue
            component_area = int(component.sum())
            coverage = float(np.sum(component & (old_mask | near_new))) / max(component_area, 1)
            expansion_ratio = component_area / max(source_overlap, 1)
            if (
                coverage >= removed_expansion_minimum_coverage
                and expansion_ratio <= removed_expansion_maximum_area_ratio
            ):
                cleanup_envelope |= component
                if np.any(component & force_cleanup):
                    force_cleanup |= component
    protected_removed = np.zeros(labels.shape, bool)
    if protected_removed_mask is not None:
        protected_removed = np.asarray(protected_removed_mask, bool)
        if protected_removed.shape != labels.shape:
            raise ValueError("protected removed mask must match baseline labels")
    # A strong one-to-one, three-mismatch identity pair identifies its source
    # footprint as obsolete history rather than a coexisting foreground object.
    # Only that explicitly paired footprint may override depth protection.
    effective_protected = protected_removed & ~force_cleanup
    dropped_removed = cleanup_envelope & ~new_mask & removed_mask & ~effective_protected
    labels[dropped_removed] = 0
    promoted = new_mask & original_changed if promote_only_changed else new_mask
    promoted |= trusted_mask
    if replacement_ownership_mask is not None:
        ownership = np.asarray(replacement_ownership_mask, bool)
        if ownership.shape != labels.shape:
            raise ValueError("replacement ownership mask must match baseline labels")
        promoted &= ownership
    labels[promoted] = replacement_label
    return labels, promoted, dropped_removed


def relabel_changed_target_fragments(
    baseline: np.ndarray,
    target_masks: Sequence[np.ndarray],
    *,
    label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel existing changed support without expanding the binary mask."""

    labels = np.asarray(baseline, np.uint8).copy()
    rewritten = np.zeros(labels.shape, bool)
    for target_mask in target_masks:
        mask = np.asarray(target_mask, bool)
        if mask.shape != labels.shape:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    labels.shape[::-1], Image.Resampling.NEAREST
                ),
                np.uint8,
            ) > 0
        edit = mask & (labels != 0) & (labels != label)
        labels[edit] = label
        rewritten |= edit
    return labels, rewritten


def replacement_cleanup_evidence(
    match: SlotMatch,
    mismatch_signal_count: int,
    baseline: np.ndarray,
    target_mask: np.ndarray,
    *,
    settings: SlotSettings,
) -> dict[str, Any]:
    """Separate old-footprint cleanup from full-target replacement trust.

    Once a target has independently passed replacement plausibility, erasing
    the matched old REMOVED footprint needs strong identity and geometric
    correspondence.  It must not depend on how the base pipeline happened
    to label the new target mask: that semantic mixture is the artifact this stage is meant to
    repair.  The stricter ``eligible`` result is retained for allowing a full
    target-mask write and for overriding target-shape plausibility.
    """

    labels = np.asarray(baseline, np.uint8)
    mask = np.asarray(target_mask, bool)
    if mask.shape != labels.shape:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                labels.shape[::-1], Image.Resampling.NEAREST
            ),
            np.uint8,
        ) > 0
    changed_pixels = int(np.sum(mask & (labels != 0)))
    added_removed_pixels = int(np.sum(mask & ((labels == 1) | (labels == 2))))
    semantic_fraction = added_removed_pixels / max(changed_pixels, 1)
    geometry = mask_geometry(target_mask)
    one_to_one = (
        match.aligned_iou >= settings.minimum_cleanup_aligned_iou
        and match.containment >= settings.minimum_cleanup_containment
        and settings.minimum_cleanup_area_ratio <= match.area_ratio <= settings.maximum_cleanup_area_ratio
        and match.score >= settings.minimum_cleanup_slot_score
    )
    compact_foreground = (
        geometry.compactness >= settings.compact_cleanup_minimum_compactness
        and geometry.bbox_fill >= settings.compact_cleanup_minimum_bbox_fill
        and geometry.aspect_ratio <= settings.compact_cleanup_maximum_aspect_ratio
        and geometry.component_count <= settings.compact_cleanup_maximum_components
    )
    asymmetric_old_envelope = (
        match.area_ratio <= settings.asymmetric_cleanup_maximum_area_ratio
        and match.containment >= settings.asymmetric_cleanup_minimum_containment
        and match.score >= settings.asymmetric_cleanup_minimum_slot_score
        and match.centroid_distance_fraction
        <= settings.asymmetric_cleanup_maximum_centroid_distance_fraction
        and geometry.largest_component_fraction
        >= settings.asymmetric_cleanup_minimum_largest_component_fraction
    )
    identity_ok = mismatch_signal_count >= settings.minimum_cleanup_mismatch_signals
    asymmetric_identity_ok = mismatch_signal_count >= settings.minimum_mismatch_signals
    semantic_ok = semantic_fraction >= settings.minimum_cleanup_added_removed_fraction
    source_cleanup_reasons = []
    if not identity_ok and not (asymmetric_old_envelope and asymmetric_identity_ok):
        source_cleanup_reasons.append("insufficient_identity_mismatches")
    if not one_to_one and not compact_foreground and not asymmetric_old_envelope:
        source_cleanup_reasons.append("neither_one_to_one_nor_compact_foreground")
    reasons = list(source_cleanup_reasons)
    if not semantic_ok:
        reasons.append("target_lacks_added_or_removed_support")
    return {
        "eligible": not reasons,
        "source_cleanup_eligible": not source_cleanup_reasons,
        "source_cleanup_reasons": source_cleanup_reasons,
        "one_to_one_geometry": one_to_one,
        "compact_foreground_override": compact_foreground,
        "asymmetric_old_envelope": asymmetric_old_envelope,
        "changed_pixels": changed_pixels,
        "added_removed_pixels": added_removed_pixels,
        "added_removed_fraction": semantic_fraction,
        "target_geometry": geometry.summary(),
        "reasons": reasons,
    }


def find_identity_elsewhere(
    source_index: int,
    target_slot_index: int,
    sources: Sequence[ObjectHypothesis],
    targets: Sequence[ObjectMask | ObjectHypothesis],
    source_sam: np.ndarray,
    target_sam: np.ndarray,
    source_dino: np.ndarray,
    target_dino: np.ndarray,
    *,
    settings: SlotSettings,
) -> dict[str, Any]:
    source = sources[source_index]
    best = {"found": False, "target_index": None, "sam_cosine": None, "dino_cosine": None}
    best_score = -np.inf
    for index, target in enumerate(targets):
        if index == target_slot_index or mask_iou(source.mask, _proposal_mask(target)) > settings.maximum_elsewhere_aligned_iou:
            continue
        sam = descriptor_cosine(source_sam, source.mask, target_sam, _proposal_mask(target))
        dino = descriptor_cosine(source_dino, source.mask, target_dino, _proposal_mask(target))
        if sam is None or dino is None:
            continue
        score = min(sam / settings.elsewhere_sam_minimum, dino / settings.elsewhere_dino_minimum)
        if score > best_score:
            best_score = score
            best = {"found": bool(sam >= settings.elsewhere_sam_minimum and dino >= settings.elsewhere_dino_minimum), "target_index": index, "sam_cosine": sam, "dino_cosine": dino}
    return best


def decide_slot_replacement(
    match: SlotMatch,
    retention: PatchRetention,
    *,
    sam_cosine: float | None,
    dino_cosine: float | None,
    color_similarity: float | None,
    identity_elsewhere: bool,
    target_support_surface: bool = False,
    settings: SlotSettings,
) -> SlotDecision:
    mismatch = []
    if sam_cosine is not None and sam_cosine <= settings.sam_mismatch_maximum:
        mismatch.append("sam_identity_mismatch")
    if dino_cosine is not None and dino_cosine <= settings.dino_mismatch_maximum:
        mismatch.append("dino_identity_mismatch")
    if color_similarity is not None and color_similarity <= settings.color_mismatch_maximum:
        mismatch.append("color_identity_mismatch")
    rescue = []
    if sam_cosine is not None and sam_cosine >= settings.sam_rescue_minimum:
        rescue.append("sam_same_identity_rescue")
    if dino_cosine is not None and dino_cosine >= settings.dino_rescue_minimum:
        rescue.append("dino_same_identity_rescue")
    if retention.retained_fraction is not None and retention.retained_fraction >= settings.retained_identity_rescue_fraction:
        rescue.append("aligned_patch_identity_rescue")
    reasons = []
    target_coverage = max(match.containment, match.aligned_iou)
    if not retention.valid:
        reasons.append("insufficient_patch_evidence")
    if retention.lost_fraction is None or retention.lost_fraction < settings.minimum_lost_identity_fraction:
        reasons.append("old_identity_not_sufficiently_lost")
    if target_coverage < settings.minimum_target_slot_coverage:
        reasons.append("target_does_not_occupy_slot")
    if len(mismatch) < settings.minimum_mismatch_signals:
        reasons.append("fewer_than_two_identity_mismatches")
    if rescue:
        reasons.append("same_identity_rescue_present")
    if target_support_surface:
        return SlotDecision(
            "abstain",
            tuple(mismatch),
            tuple(rescue),
            tuple((*reasons, "target_is_support_surface")),
        )
    confident_motion = (
        match.track_iou >= settings.confident_moved_track_iou_minimum
        and color_similarity is not None
        and color_similarity >= settings.confident_moved_color_minimum
    )
    if confident_motion:
        return SlotDecision(
            "moved_elsewhere",
            tuple(mismatch),
            tuple((*rescue, "track_color_motion_rescue")),
            tuple((*reasons, "propagated_track_and_color_support_motion")),
        )
    if identity_elsewhere:
        return SlotDecision("moved_elsewhere", tuple(mismatch), tuple(rescue), tuple((*reasons, "old_identity_found_elsewhere")))
    if reasons:
        verdict = "same_identity" if rescue else "abstain"
        return SlotDecision(verdict, tuple(mismatch), tuple(rescue), tuple(reasons))
    return SlotDecision("replaced", tuple(mismatch), tuple(rescue), ())
