"""Object-level change detection over three images on one aligned pixel grid.

The three frames are:

``render_t0``
    The old scene rendered into the T1 camera (see ``reconstruction.py``).
``clean_render``
    The canonical point-cloud render in the same camera, with mutual
    depth-conflict filtering applied so add/remove transients are pruned.
``image_t1``
    The real current image, and the pixel grid the output is aligned to.

Each frame gets an independent SAM3 automatic-mask inventory with pooled
SAM3 + DINOv2 descriptors per object. A correspondence between two frames is
accepted as an identity only when both descriptors agree (or a SAM2-tracked
mask and both descriptors agree). The clean render is a third-view bridge
when the render/photo appearance gap makes a direct comparison fail.

Objects that end up unmatched are, by default, revisited once more before
being called removed/added: ``recover_unmatched_via_tracking`` reuses the
SAM2 track already computed for every object (regardless of whether it later
got a partner) and checks it against the opposite frame's own dense
SAM3/DINOv2 feature maps at the same identity thresholds used everywhere
else here. This recovers objects that SAM3's automatic mask generator
proposed in only one of the two frames, which is not evidence of a real
change.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, label as connected_components
from scipy.optimize import linear_sum_assignment

from .io import save_image, save_json
from .stages.sam2_tracking_backend import Sam2MaskTracker
from .stages.sam3_identity_location import (
    FeatureDescriptorBatch,
    cosine_similarity_matrix,
    mask_descriptors,
    pairwise_mask_iou,
)
from .stages.sam3_proposals import Sam3AutomaticMaskGenerator, Sam3Proposal
from .adapters.dinov2 import Dinov2FeatureExtractor
from .types import Label, ObjectMask
from .visualization import colorize, instance_overlay, overlay


@dataclass(frozen=True)
class ThreeImageSettings:
    """Thresholds for proposal cleanup, association, and classification."""

    minimum_mask_area: int = 16
    maximum_mask_area_fraction: float = 0.25
    background_border_sides: int = 1
    background_minimum_area_fraction: float = 0.0
    duplicate_iou: float = 0.80
    duplicate_containment: float = 0.65
    duplicate_minimum_area_ratio: float = 0.0
    part_maximum_area_ratio: float = 0.40
    part_bbox_margin_pixels: int = 3
    minimum_sam_feature_cells: float = 1.0
    minimum_dino_feature_cells: float = 1.0
    minimum_sam_cosine: float = 0.65
    minimum_dino_cosine: float = 0.60
    different_identity_margin: float = 0.08
    minimum_identity_margin: float = 0.02
    area_ratio_low: float = 0.25
    area_ratio_high: float = 4.0
    minimum_track_iou: float = 0.20
    same_location_iou: float = 0.45
    replacement_location_iou: float = 0.30
    tracking_batch_size: int = 16
    recover_unmatched_via_tracking: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ThreeImageSettings":
        values = config.get("three_image_comparison", {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("Unknown three_image_comparison settings: " + ", ".join(unknown))
        return cls(**values)


@dataclass(frozen=True)
class FrameInventory:
    """Object masks and two independent descriptor batches for one frame."""

    objects: tuple[ObjectMask, ...]
    sam: FeatureDescriptorBatch
    dino: FeatureDescriptorBatch


@dataclass(frozen=True)
class TrackingEvidence:
    """Bidirectional SAM2 tracks for the three pairwise frame combinations."""

    t0_to_t1: tuple[np.ndarray | None, ...]
    t1_to_t0: tuple[np.ndarray | None, ...]
    t0_to_clean: tuple[np.ndarray | None, ...]
    clean_to_t0: tuple[np.ndarray | None, ...]
    t1_to_clean: tuple[np.ndarray | None, ...]
    clean_to_t1: tuple[np.ndarray | None, ...]


@dataclass(frozen=True)
class ThreeImageResult:
    labels: np.ndarray
    objects: tuple[ObjectMask, ...]
    decisions: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    artifacts_dir: Path
    timings: dict[str, float]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = connected_components(np.asarray(mask, dtype=bool))
    if count <= 1:
        return np.asarray(mask, dtype=bool).copy()
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def _border_sides(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    return sum(bool(side.any()) for side in (binary[0], binary[-1], binary[:, 0], binary[:, -1]))


def select_object_proposals(
    proposals: Sequence[Sam3Proposal], settings: ThreeImageSettings
) -> list[ObjectMask]:
    """Turn overlapping automatic proposals into a compact object inventory.

    This removes empty/tiny masks, obvious image-spanning background regions
    (masks touching the frame border), disconnected residuals, and
    near-duplicate/nested proposals. It cannot infer semantic foreground
    classes; that would require labelled fine-tuning data or a
    text-conditioned detector.
    """
    candidates = []
    for index, proposal in enumerate(proposals, start=1):
        x0, y0, x1, y1 = proposal.crop_box_xyxy
        candidates.append(
            ObjectMask(
                mask=np.asarray(proposal.mask, dtype=bool),
                score=float(proposal.predicted_iou),
                source="sam3_automatic",
                metadata={
                    "automatic_proposal_id": index,
                    "proposal_backend": "sam3",
                    "stability_score": float(proposal.stability_score),
                    "point_coords": [list(proposal.point_xy)],
                    "crop_box_xywh": [x0, y0, x1 - x0, y1 - y0],
                },
            )
        )
    if not candidates:
        return []

    height, width = np.asarray(candidates[0].mask).shape
    image_area = height * width
    filtered: list[ObjectMask] = []
    for item in candidates:
        mask = _largest_component(item.mask)
        if mask.shape != (height, width):
            raise ValueError("all proposals in an inventory must share one shape")
        area = int(mask.sum())
        fraction = area / max(image_area, 1)
        sides = _border_sides(mask)
        if area < settings.minimum_mask_area:
            continue
        if fraction > settings.maximum_mask_area_fraction:
            continue
        if sides >= settings.background_border_sides and fraction >= settings.background_minimum_area_fraction:
            continue
        filtered.append(
            ObjectMask(
                mask=mask,
                score=item.score,
                source=item.source,
                metadata={**item.metadata, "area": area, "area_fraction": fraction, "border_sides": sides},
            )
        )

    def rank(item: ObjectMask) -> tuple[int, float, float, int]:
        # Every candidate already cleared SAM3's quality/stability gates. Prefer
        # the complete enclosing mask so a head, button, lid, or face does not
        # become another object beside the whole instance.
        return (
            int(np.asarray(item.mask, bool).sum()),
            float(item.score),
            float(item.metadata.get("stability_score", 0.0)),
            -int(item.metadata.get("automatic_proposal_id", 0)),
        )

    retained: list[ObjectMask] = []
    for candidate in sorted(filtered, key=rank, reverse=True):
        candidate_mask = np.asarray(candidate.mask, dtype=bool)
        candidate_area = int(candidate_mask.sum())
        duplicate = False
        for existing in retained:
            existing_mask = np.asarray(existing.mask, dtype=bool)
            existing_area = int(existing_mask.sum())
            intersection = int(np.logical_and(candidate_mask, existing_mask).sum())
            union = candidate_area + existing_area - intersection
            iou = intersection / max(union, 1)
            containment = intersection / max(min(candidate_area, existing_area), 1)
            area_ratio = min(candidate_area, existing_area) / max(candidate_area, existing_area, 1)
            if iou >= settings.duplicate_iou or (
                containment >= settings.duplicate_containment and area_ratio >= settings.duplicate_minimum_area_ratio
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)

    retained.sort(key=lambda item: int(item.metadata.get("automatic_proposal_id", 0)))
    for proposal_id, item in enumerate(retained, start=1):
        item.metadata["object_id"] = proposal_id
    return retained


def _track_iou(tracks: Sequence[np.ndarray | None], targets: Sequence[ObjectMask]) -> np.ndarray:
    output = np.zeros((len(tracks), len(targets)), dtype=np.float32)
    target_masks = [np.asarray(item.mask, dtype=bool) for item in targets]
    target_areas = np.asarray([mask.sum() for mask in target_masks], dtype=np.int64)
    for row, track in enumerate(tracks):
        if track is None:
            continue
        source = np.asarray(track, dtype=bool)
        source_area = int(source.sum())
        for column, target in enumerate(target_masks):
            intersection = int(np.logical_and(source, target).sum())
            union = source_area + int(target_areas[column]) - intersection
            if union:
                output[row, column] = intersection / union
    return output


def _bidirectional_track_score(
    forward: Sequence[np.ndarray | None],
    backward: Sequence[np.ndarray | None],
    source: Sequence[ObjectMask],
    target: Sequence[ObjectMask],
) -> tuple[np.ndarray, np.ndarray]:
    forward_iou = _track_iou(forward, target)
    backward_iou = _track_iou(backward, source).T
    return np.minimum(forward_iou, backward_iou), np.maximum(forward_iou, backward_iou)


def _feature_matrices(source: FrameInventory, target: FrameInventory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sam = cosine_similarity_matrix(source.sam, target.sam)
    dino = cosine_similarity_matrix(source.dino, target.dino)
    valid = (
        source.sam.valid[:, None]
        & target.sam.valid[None, :]
        & source.dino.valid[:, None]
        & target.dino.valid[None, :]
    )
    return sam, dino, valid


def _areas(objects: Sequence[ObjectMask]) -> np.ndarray:
    return np.asarray([np.asarray(item.mask, bool).sum() for item in objects], np.float32)


def _reciprocal_with_margin(scores: np.ndarray, feasible: np.ndarray, margin: float) -> np.ndarray:
    accepted = np.zeros_like(feasible, dtype=bool)
    if not feasible.size or not feasible.any():
        return accepted
    masked = np.where(feasible, scores, -np.inf)
    row_best = np.argmax(masked, axis=1)
    column_best = np.argmax(masked, axis=0)
    for row, column in zip(*np.nonzero(feasible)):
        if row_best[row] != column or column_best[column] != row:
            continue
        row_values = np.delete(masked[row], column)
        column_values = np.delete(masked[:, column], row)
        row_second = float(row_values.max(initial=-1.0))
        column_second = float(column_values.max(initial=-1.0))
        if scores[row, column] - max(row_second, column_second) >= margin:
            accepted[row, column] = True
    return accepted


def _assign(scores: np.ndarray, feasible: np.ndarray) -> list[tuple[int, int]]:
    if not feasible.size or not feasible.any():
        return []
    rows, columns = linear_sum_assignment(np.where(feasible, -scores, 1e6))
    return [(int(row), int(column)) for row, column in zip(rows, columns) if feasible[row, column]]


def _identity_candidates(
    source: FrameInventory,
    target: FrameInventory,
    bidirectional_track: np.ndarray,
    any_track: np.ndarray,
    settings: ThreeImageSettings,
) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    sam, dino, valid = _feature_matrices(source, target)
    source_area = _areas(source.objects)
    target_area = _areas(target.objects)
    ratio = target_area[None, :] / np.maximum(source_area[:, None], 1.0)
    identity = (
        valid
        & (sam >= settings.minimum_sam_cosine)
        & (dino >= settings.minimum_dino_cosine)
        & (ratio >= settings.area_ratio_low)
        & (ratio <= settings.area_ratio_high)
    )
    feature_score = np.minimum(sam, dino)
    reciprocal = _reciprocal_with_margin(feature_score, identity, settings.minimum_identity_margin)
    tracked = bidirectional_track >= settings.minimum_track_iou
    # One-direction tracks are useful for ranking, but cannot independently
    # establish identity. Untracked masks enter only through reciprocal feature
    # matching.
    feasible = identity & (tracked | reciprocal)
    score = feature_score + 0.20 * bidirectional_track + 0.05 * any_track
    return _assign(score, feasible), {
        "sam": sam,
        "dino": dino,
        "valid": valid,
        "identity": identity,
        "reciprocal": reciprocal,
        "tracked": tracked,
        "score": score,
    }


def _endpoint_map(bidirectional_track: np.ndarray, any_track: np.ndarray, minimum_iou: float) -> dict[int, int]:
    feasible = bidirectional_track >= minimum_iou
    score = bidirectional_track + 0.10 * any_track
    return {source: target for source, target in _assign(score, feasible)}


def _identity_at(
    sam: np.ndarray, dino: np.ndarray, valid: np.ndarray, row: int, column: int, settings: ThreeImageSettings
) -> bool:
    return bool(
        valid[row, column]
        and sam[row, column] >= settings.minimum_sam_cosine
        and dino[row, column] >= settings.minimum_dino_cosine
    )


def resolve_three_image_changes(
    t0: FrameInventory,
    clean: FrameInventory,
    t1: FrameInventory,
    tracks: TrackingEvidence,
    settings: ThreeImageSettings,
) -> tuple[np.ndarray, list[ObjectMask], list[dict[str, Any]], dict[str, Any]]:
    """Resolve unchanged/moved/replaced/removed/added object states."""

    shape_candidates = [item.mask.shape for frame in (t0, clean, t1) for item in frame.objects]
    if not shape_candidates:
        raise ValueError("at least one object proposal is required")
    shape = shape_candidates[0]
    if any(candidate != shape for candidate in shape_candidates):
        raise ValueError("all three object inventories must use one aligned grid")

    direct_bi, direct_any = _bidirectional_track_score(tracks.t0_to_t1, tracks.t1_to_t0, t0.objects, t1.objects)
    t0_clean_bi, t0_clean_any = _bidirectional_track_score(
        tracks.t0_to_clean, tracks.clean_to_t0, t0.objects, clean.objects
    )
    t1_clean_bi, t1_clean_any = _bidirectional_track_score(
        tracks.t1_to_clean, tracks.clean_to_t1, t1.objects, clean.objects
    )
    direct_pairs, direct_features = _identity_candidates(t0, t1, direct_bi, direct_any, settings)
    spatial = pairwise_mask_iou(t0.objects, t1.objects)
    # The clean point-cloud view can be incomplete, so requiring both track
    # directions here would discard the very bridge that is meant to recover
    # a render/photo domain gap. A one-way endpoint track is only a location
    # proposal; SAM3 and DINO still have to agree below before identity passes.
    t0_clean_map = _endpoint_map(t0_clean_any, t0_clean_any, settings.minimum_track_iou)
    t1_clean_map = _endpoint_map(t1_clean_any, t1_clean_any, settings.minimum_track_iou)
    clean_sam = cosine_similarity_matrix(clean.sam, clean.sam)
    clean_dino = cosine_similarity_matrix(clean.dino, clean.dino)
    clean_valid = clean.sam.valid[:, None] & clean.sam.valid[None, :] & clean.dino.valid[:, None] & clean.dino.valid[None, :]

    consumed_t0: set[int] = set()
    consumed_t1: set[int] = set()
    decisions: list[dict[str, Any]] = []
    output_objects: list[ObjectMask] = []

    def record_pair(source: int, target: int, decision: Label, evidence: str) -> None:
        consumed_t0.add(source)
        consumed_t1.add(target)
        clean_source = t0_clean_map.get(source)
        clean_target = t1_clean_map.get(target)
        clean_confirmed = clean_source is not None and clean_target is not None
        # A replacement is visualized on the current object only. The old
        # footprint is absent from I1 and painting its union made the overlay
        # look larger than the object that actually occupies the slot.
        mask = (
            np.asarray(t1.objects[target].mask, dtype=bool).copy()
            if decision == Label.REPLACED
            else np.logical_or(t0.objects[source].mask, t1.objects[target].mask)
        )
        decisions.append(
            {
                "decision": decision.name.lower(),
                "t0_object_id": source + 1,
                "t1_object_id": target + 1,
                "clean_t0_object_id": None if clean_source is None else clean_source + 1,
                "clean_t1_object_id": None if clean_target is None else clean_target + 1,
                "clean_location_confirmed": clean_confirmed,
                "evidence": evidence,
                "spatial_iou": float(spatial[source, target]),
                "track_iou": float(direct_bi[source, target]),
                "sam_cosine": float(direct_features["sam"][source, target]),
                "dino_cosine": float(direct_features["dino"][source, target]),
            }
        )
        if decision != Label.UNCHANGED:
            output_objects.append(
                ObjectMask(
                    mask=mask,
                    score=float(min(t0.objects[source].score, t1.objects[target].score)),
                    label=decision,
                    source=f"three_image_{evidence}",
                    metadata={"t0_object_id": source + 1, "t1_object_id": target + 1},
                )
            )

    for source, target in direct_pairs:
        decision = Label.UNCHANGED if spatial[source, target] >= settings.same_location_iou else Label.MOVED
        record_pair(source, target, decision, "direct_identity")

    # A reliable track can survive a render/photo feature-domain gap. Validate
    # such a pair by comparing its two endpoints inside the common clean-render
    # feature domain.
    bridge_edges = np.zeros_like(direct_bi, dtype=bool)
    bridge_scores = np.zeros_like(direct_bi, dtype=np.float32)
    for source, target in zip(*np.nonzero(direct_any >= settings.minimum_track_iou)):
        if source in consumed_t0 or target in consumed_t1:
            continue
        clean_source = t0_clean_map.get(int(source))
        clean_target = t1_clean_map.get(int(target))
        if clean_source is None or clean_target is None:
            continue
        if not _identity_at(clean_sam, clean_dino, clean_valid, clean_source, clean_target, settings):
            continue
        bridge_edges[source, target] = True
        bridge_scores[source, target] = (
            min(clean_sam[clean_source, clean_target], clean_dino[clean_source, clean_target]) + direct_bi[source, target]
        )
    for source, target in _assign(bridge_scores, bridge_edges):
        decision = Label.UNCHANGED if spatial[source, target] >= settings.same_location_iou else Label.MOVED
        record_pair(source, target, decision, "clean_bridge_identity")

    # Remaining old/new masks in the same slot are replacements only when the
    # two feature systems provide positive evidence that identities differ.
    replacement_edges = np.zeros_like(spatial, dtype=bool)
    replacement_scores = np.zeros_like(spatial, dtype=np.float32)
    different_sam = settings.minimum_sam_cosine - settings.different_identity_margin
    different_dino = settings.minimum_dino_cosine - settings.different_identity_margin
    for source, target in zip(*np.nonzero(spatial >= settings.replacement_location_iou)):
        if source in consumed_t0 or target in consumed_t1:
            continue
        if not direct_features["valid"][source, target]:
            continue
        confidently_different = (
            direct_features["sam"][source, target] <= different_sam
            or direct_features["dino"][source, target] <= different_dino
        )
        if confidently_different:
            replacement_edges[source, target] = True
            replacement_scores[source, target] = spatial[source, target]
    for source, target in _assign(replacement_scores, replacement_edges):
        record_pair(source, target, Label.REPLACED, "same_slot_different_identity")

    for source, item in enumerate(t0.objects):
        if source in consumed_t0:
            continue
        output_objects.append(
            ObjectMask(mask=np.asarray(item.mask, bool).copy(), score=item.score, label=Label.REMOVED,
                       source="three_image_unmatched_t0", metadata={"t0_object_id": source + 1})
        )
        decisions.append({"decision": "removed", "t0_object_id": source + 1, "t1_object_id": None})
    for target, item in enumerate(t1.objects):
        if target in consumed_t1:
            continue
        output_objects.append(
            ObjectMask(mask=np.asarray(item.mask, bool).copy(), score=item.score, label=Label.ADDED,
                       source="three_image_unmatched_t1", metadata={"t1_object_id": target + 1})
        )
        decisions.append({"decision": "added", "t0_object_id": None, "t1_object_id": target + 1})

    labels = np.zeros(shape, dtype=np.uint8)
    # Added is weakest where proposal masks overlap; same-identity motion and
    # replacement are the most specific object-level explanations.
    priority = (Label.ADDED, Label.REMOVED, Label.MOVED, Label.REPLACED)
    for label_value in priority:
        for item in output_objects:
            if item.label == label_value:
                labels[np.asarray(item.mask, bool)] = int(label_value)
    counts = {
        name: sum(row["decision"] == name for row in decisions)
        for name in ("unchanged", "moved", "replaced", "removed", "added")
    }
    diagnostics = {
        "object_counts": {"render_t0": len(t0.objects), "clean_render": len(clean.objects), "image_t1": len(t1.objects)},
        "decision_counts": counts,
        "changed_pixel_fraction": float(np.mean(labels != int(Label.UNCHANGED))),
        "association_evidence": {
            "direct_bidirectional_track_iou": direct_bi.tolist(),
            "direct_any_direction_track_iou": direct_any.tolist(),
            "direct_sam_cosine": direct_features["sam"].tolist(),
            "direct_dino_cosine": direct_features["dino"].tolist(),
            "direct_spatial_iou": spatial.tolist(),
            "t0_to_clean_bidirectional_track_iou": t0_clean_bi.tolist(),
            "t0_to_clean_any_direction_track_iou": t0_clean_any.tolist(),
            "t1_to_clean_bidirectional_track_iou": t1_clean_bi.tolist(),
            "t1_to_clean_any_direction_track_iou": t1_clean_any.tolist(),
        },
    }
    return labels, output_objects, decisions, diagnostics


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.logical_or(first, second).sum())
    if not union:
        return 0.0
    return int(np.logical_and(first, second).sum()) / union


def recover_unmatched_via_tracking(
    labels: np.ndarray,
    output_objects: list[ObjectMask],
    decisions: list[dict[str, Any]],
    tracks: TrackingEvidence,
    t0: FrameInventory,
    t1: FrameInventory,
    sam_t0_map: np.ndarray,
    dino_t0_map: np.ndarray,
    sam_t1_map: np.ndarray,
    dino_t1_map: np.ndarray,
    settings: ThreeImageSettings,
) -> tuple[np.ndarray, list[ObjectMask], list[dict[str, Any]], dict[str, Any]]:
    """Recover objects SAM3 only proposed in one frame, using the track SAM2
    already computed for every object instead of discarding it once an
    unmatched object is about to be declared removed/added.

    ``resolve_three_image_changes`` only ever pairs objects that both have an
    independent SAM3 proposal in their own frame; ``tracks.t0_to_t1[i]`` (a
    real mask propagated into image_t1 by Sam2MaskTracker) is computed for
    every t0 object regardless, but is silently dropped for any object that
    ends up unmatched. This revisits exactly those "removed"/"added" verdicts:
    if the existing track for an unmatched object is accepted, and pooling
    the target frame's own dense SAM3/DINOv2 feature maps under that tracked
    mask still agrees with the source object's descriptor at the same
    thresholds used everywhere else in this module, the proposal-generation
    miss is not evidence of a real change.
    """
    removed_t0_ids = {row["t0_object_id"] - 1 for row in decisions if row["decision"] == "removed"}
    added_t1_ids = {row["t1_object_id"] - 1 for row in decisions if row["decision"] == "added"}
    object_index_by_removed_t0 = {
        item.metadata["t0_object_id"] - 1: index for index, item in enumerate(output_objects) if item.label == Label.REMOVED
    }
    object_index_by_added_t1 = {
        item.metadata["t1_object_id"] - 1: index for index, item in enumerate(output_objects) if item.label == Label.ADDED
    }

    def area_ratio_ok(source_area: float, candidate_area: float) -> bool:
        ratio = candidate_area / max(source_area, 1.0)
        return settings.area_ratio_low <= ratio <= settings.area_ratio_high

    def try_recover(
        source_id: int,
        candidate_mask: np.ndarray | None,
        source_inventory: FrameInventory,
        candidate_map_sam: np.ndarray,
        candidate_map_dino: np.ndarray,
    ) -> tuple[float, float, float] | None:
        if candidate_mask is None:
            return None
        if not source_inventory.sam.valid[source_id] or not source_inventory.dino.valid[source_id]:
            return None
        candidate = ObjectMask(mask=candidate_mask)
        cand_sam = mask_descriptors(candidate_map_sam, [candidate], minimum_feature_cells=settings.minimum_sam_feature_cells)
        cand_dino = mask_descriptors(candidate_map_dino, [candidate], minimum_feature_cells=settings.minimum_dino_feature_cells)
        if not (cand_sam.valid[0] and cand_dino.valid[0]):
            return None
        sam_cos = float(source_inventory.sam.vectors[source_id] @ cand_sam.vectors[0])
        dino_cos = float(source_inventory.dino.vectors[source_id] @ cand_dino.vectors[0])
        if sam_cos < settings.minimum_sam_cosine or dino_cos < settings.minimum_dino_cosine:
            return None
        source_area = float(np.asarray(source_inventory.objects[source_id].mask, bool).sum())
        if not area_ratio_ok(source_area, float(np.asarray(candidate_mask, bool).sum())):
            return None
        return sam_cos, dino_cos, float(candidate_mask.sum())

    recovered_decisions: list[dict[str, Any]] = []
    dropped_object_indices: set[int] = set()
    recovered_t0_ids: set[int] = set()
    recovered_t1_ids: set[int] = set()

    for t0_id in sorted(removed_t0_ids):
        result = try_recover(t0_id, tracks.t0_to_t1[t0_id], t0, sam_t1_map, dino_t1_map)
        if result is None:
            continue
        sam_cos, dino_cos, _ = result
        tracked_mask = np.asarray(tracks.t0_to_t1[t0_id], dtype=bool)
        source_mask = np.asarray(t0.objects[t0_id].mask, dtype=bool)
        absorbed_t1 = None
        for t1_id in sorted(added_t1_ids - recovered_t1_ids):
            if _mask_iou(np.asarray(t1.objects[t1_id].mask, bool), tracked_mask) >= settings.same_location_iou:
                absorbed_t1 = t1_id
                break
        recovered_t0_ids.add(t0_id)
        dropped_object_indices.add(object_index_by_removed_t0[t0_id])
        if absorbed_t1 is not None:
            recovered_t1_ids.add(absorbed_t1)
            dropped_object_indices.add(object_index_by_added_t1[absorbed_t1])
            evidence_mask = np.logical_or(source_mask, np.asarray(t1.objects[absorbed_t1].mask, bool))
        else:
            evidence_mask = np.logical_or(source_mask, tracked_mask)
        location_iou = _mask_iou(source_mask, tracked_mask)
        new_label = Label.UNCHANGED if location_iou >= settings.same_location_iou else Label.MOVED
        recovered_decisions.append(
            {
                "decision": new_label.name.lower(),
                "t0_object_id": t0_id + 1,
                "t1_object_id": (absorbed_t1 + 1) if absorbed_t1 is not None else None,
                "evidence": "tracking_recovery_t0_to_t1",
                "track_iou": location_iou,
                "sam_cosine": sam_cos,
                "dino_cosine": dino_cos,
            }
        )
        if new_label != Label.UNCHANGED:
            output_objects.append(
                ObjectMask(
                    mask=evidence_mask,
                    score=float(t0.objects[t0_id].score),
                    label=new_label,
                    source="three_image_tracking_recovery",
                    metadata={"t0_object_id": t0_id + 1, "t1_object_id": (absorbed_t1 + 1) if absorbed_t1 is not None else None},
                )
            )

    for t1_id in sorted(added_t1_ids - recovered_t1_ids):
        result = try_recover(t1_id, tracks.t1_to_t0[t1_id], t1, sam_t0_map, dino_t0_map)
        if result is None:
            continue
        sam_cos, dino_cos, _ = result
        tracked_mask = np.asarray(tracks.t1_to_t0[t1_id], dtype=bool)
        target_mask = np.asarray(t1.objects[t1_id].mask, dtype=bool)
        absorbed_t0 = None
        for t0_id in sorted(removed_t0_ids - recovered_t0_ids):
            if _mask_iou(np.asarray(t0.objects[t0_id].mask, bool), tracked_mask) >= settings.same_location_iou:
                absorbed_t0 = t0_id
                break
        recovered_t1_ids.add(t1_id)
        dropped_object_indices.add(object_index_by_added_t1[t1_id])
        if absorbed_t0 is not None:
            recovered_t0_ids.add(absorbed_t0)
            dropped_object_indices.add(object_index_by_removed_t0[absorbed_t0])
            evidence_mask = np.logical_or(target_mask, np.asarray(t0.objects[absorbed_t0].mask, bool))
        else:
            evidence_mask = np.logical_or(target_mask, tracked_mask)
        location_iou = _mask_iou(target_mask, tracked_mask)
        new_label = Label.UNCHANGED if location_iou >= settings.same_location_iou else Label.MOVED
        recovered_decisions.append(
            {
                "decision": new_label.name.lower(),
                "t0_object_id": (absorbed_t0 + 1) if absorbed_t0 is not None else None,
                "t1_object_id": t1_id + 1,
                "evidence": "tracking_recovery_t1_to_t0",
                "track_iou": location_iou,
                "sam_cosine": sam_cos,
                "dino_cosine": dino_cos,
            }
        )
        if new_label != Label.UNCHANGED:
            output_objects.append(
                ObjectMask(
                    mask=evidence_mask,
                    score=float(t1.objects[t1_id].score),
                    label=new_label,
                    source="three_image_tracking_recovery",
                    metadata={"t0_object_id": (absorbed_t0 + 1) if absorbed_t0 is not None else None, "t1_object_id": t1_id + 1},
                )
            )

    kept_decisions = [
        row
        for row in decisions
        if not (
            (row["decision"] == "removed" and row["t0_object_id"] - 1 in recovered_t0_ids)
            or (row["decision"] == "added" and row["t1_object_id"] - 1 in recovered_t1_ids)
        )
    ]
    final_decisions = kept_decisions + recovered_decisions
    final_objects = [item for index, item in enumerate(output_objects) if index not in dropped_object_indices]

    new_labels = np.zeros(labels.shape, dtype=np.uint8)
    priority = (Label.ADDED, Label.REMOVED, Label.MOVED, Label.REPLACED)
    for label_value in priority:
        for item in final_objects:
            if item.label == label_value:
                new_labels[np.asarray(item.mask, bool)] = int(label_value)

    counts = {
        name: sum(row["decision"] == name for row in final_decisions)
        for name in ("unchanged", "moved", "replaced", "removed", "added")
    }
    recovery_diagnostics = {
        "decision_counts": counts,
        "changed_pixel_fraction": float(np.mean(new_labels != int(Label.UNCHANGED))),
        "tracking_recoveries": len(recovered_decisions),
        "tracking_recovered_t0_ids": sorted(i + 1 for i in recovered_t0_ids),
        "tracking_recovered_t1_ids": sorted(i + 1 for i in recovered_t1_ids),
    }
    return new_labels, final_objects, final_decisions, recovery_diagnostics


def _track_batches(
    tracker: Sam2MaskTracker, objects: Sequence[ObjectMask], source_image: np.ndarray, target_image: np.ndarray, batch_size: int
) -> tuple[np.ndarray | None, ...]:
    results: list[np.ndarray | None] = []
    for start in range(0, len(objects), batch_size):
        batch = objects[start : start + batch_size]
        attempts = tracker.track([np.asarray(item.mask, bool) for item in batch], source_image, target_image)
        results.extend(np.asarray(attempt.mask, bool).copy() if attempt.accepted else None for attempt in attempts)
    return tuple(results)


def _build_inventory(
    objects: Sequence[ObjectMask], sam_map: np.ndarray, dino_map: np.ndarray, settings: ThreeImageSettings
) -> FrameInventory:
    return FrameInventory(
        objects=tuple(objects),
        sam=mask_descriptors(sam_map, objects, minimum_feature_cells=settings.minimum_sam_feature_cells),
        dino=mask_descriptors(dino_map, objects, minimum_feature_cells=settings.minimum_dino_feature_cells),
    )


def _suppress_feature_matched_parts(inventory: FrameInventory, settings: ThreeImageSettings) -> FrameInventory:
    """Drop a small disjoint part inside a larger same-identity object box."""

    if len(inventory.objects) < 2:
        return inventory
    sam = cosine_similarity_matrix(inventory.sam, inventory.sam)
    dino = cosine_similarity_matrix(inventory.dino, inventory.dino)
    areas = _areas(inventory.objects)
    order = sorted(range(len(inventory.objects)), key=lambda index: -areas[index])
    retained: list[int] = []
    margin = settings.part_bbox_margin_pixels
    for index in order:
        mask = np.asarray(inventory.objects[index].mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        center_x, center_y = float(xs.mean()), float(ys.mean())
        is_part = False
        for parent in retained:
            ratio = float(areas[index] / max(areas[parent], 1.0))
            if ratio > settings.part_maximum_area_ratio:
                continue
            parent_mask = np.asarray(inventory.objects[parent].mask, dtype=bool)
            parent_y, parent_x = np.nonzero(parent_mask)
            if not len(parent_x):
                continue
            inside_box = (
                parent_x.min() - margin <= center_x <= parent_x.max() + margin
                and parent_y.min() - margin <= center_y <= parent_y.max() + margin
            )
            features_match = (
                inventory.sam.valid[index]
                and inventory.sam.valid[parent]
                and inventory.dino.valid[index]
                and inventory.dino.valid[parent]
                and sam[index, parent] >= settings.minimum_sam_cosine
                and dino[index, parent] >= settings.minimum_dino_cosine
            )
            adjacent = bool(np.logical_and(mask, binary_dilation(parent_mask, iterations=max(margin, 1))).any())
            if inside_box and (features_match or adjacent):
                is_part = True
                break
        if not is_part:
            retained.append(index)
    retained.sort()

    def subset(batch: FeatureDescriptorBatch) -> FeatureDescriptorBatch:
        indices = np.asarray(retained, dtype=int)
        return FeatureDescriptorBatch(
            vectors=batch.vectors[indices], valid=batch.valid[indices], effective_cells=batch.effective_cells[indices]
        )

    return FrameInventory(
        objects=tuple(inventory.objects[index] for index in retained), sam=subset(inventory.sam), dino=subset(inventory.dino)
    )


def _proposal_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["sam3_proposals"]["proposals"]
    return {
        "points_per_side": int(cfg["points_per_side"]),
        "points_per_batch": int(cfg["points_per_batch"]),
        "pred_iou_threshold": float(cfg["pred_iou_threshold"]),
        "stability_threshold": float(cfg["stability_threshold"]),
        "stability_offset": float(cfg["stability_offset"]),
        "crop_layers": int(cfg["crop_layers"]),
        "crop_downscale_factor": int(cfg["crop_downscale_factor"]),
        "box_nms_threshold": float(cfg["box_nms_threshold"]),
        "crop_nms_threshold": float(cfg["crop_nms_threshold"]),
        "minimum_mask_area": int(cfg["minimum_mask_area"]),
        "multimask_output": bool(cfg["multimask_output"]),
    }


def run_object_state_resolution(
    render_t0: np.ndarray,
    clean_render: np.ndarray,
    image_t1: np.ndarray,
    output_dir: str | Path,
    config: dict[str, Any],
) -> ThreeImageResult:
    """Run SAM3 + DINOv2 + SAM2-tracking inference over three aligned images
    and write a T1-aligned change-detection result.

    ``render_t0``/``clean_render``/``image_t1`` must already be produced by
    ``reconstruction.py`` (optionally refined by ``refine.py``) and share one
    pixel grid.
    """

    images = tuple(np.asarray(image, dtype=np.uint8) for image in (render_t0, clean_render, image_t1))
    if any(image.ndim != 3 or image.shape[2] != 3 for image in images):
        raise ValueError("three-image inputs must be H x W x 3 RGB arrays")
    if len({image.shape for image in images}) != 1:
        raise ValueError("three-image inputs must already share one aligned pixel grid")
    render_t0, clean_render, image_t1 = images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = ThreeImageSettings.from_config(config)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    sam_cfg = config["sam3_proposals"]
    generator = Sam3AutomaticMaskGenerator(sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"], **_proposal_kwargs(config))
    try:
        raw_t0, sam_t0 = generator.generate_with_feature_map(render_t0)
        raw_clean, sam_clean = generator.generate_with_feature_map(clean_render)
        raw_t1, sam_t1 = generator.generate_with_feature_map(image_t1)
    finally:
        generator.release()
    objects_t0 = select_object_proposals(raw_t0, settings)
    objects_clean = select_object_proposals(raw_clean, settings)
    objects_t1 = select_object_proposals(raw_t1, settings)
    timings["01_sam3_inventory_and_features"] = time.perf_counter() - started

    started = time.perf_counter()
    dino_extractor = Dinov2FeatureExtractor(config["dinov2_features"])
    try:
        dino_t0 = dino_extractor.feature_map(render_t0)
        dino_clean = dino_extractor.feature_map(clean_render)
        dino_t1 = dino_extractor.feature_map(image_t1)
    finally:
        dino_extractor.release()
    inventory_t0 = _suppress_feature_matched_parts(_build_inventory(objects_t0, sam_t0, dino_t0, settings), settings)
    inventory_clean = _suppress_feature_matched_parts(_build_inventory(objects_clean, sam_clean, dino_clean, settings), settings)
    inventory_t1 = _suppress_feature_matched_parts(_build_inventory(objects_t1, sam_t1, dino_t1, settings), settings)
    objects_t0 = list(inventory_t0.objects)
    objects_clean = list(inventory_clean.objects)
    objects_t1 = list(inventory_t1.objects)
    timings["02_dinov2_and_pooling"] = time.perf_counter() - started

    started = time.perf_counter()
    tracker = Sam2MaskTracker(config["sam2_tracking"])
    try:
        track = lambda objects, source, target: _track_batches(tracker, objects, source, target, settings.tracking_batch_size)  # noqa: E731
        tracking = TrackingEvidence(
            t0_to_t1=track(objects_t0, render_t0, image_t1),
            t1_to_t0=track(objects_t1, image_t1, render_t0),
            t0_to_clean=track(objects_t0, render_t0, clean_render),
            clean_to_t0=track(objects_clean, clean_render, render_t0),
            t1_to_clean=track(objects_t1, image_t1, clean_render),
            clean_to_t1=track(objects_clean, clean_render, image_t1),
        )
    finally:
        tracker.release()
    timings["03_bidirectional_tracking"] = time.perf_counter() - started

    started = time.perf_counter()
    labels, objects, decisions, diagnostics = resolve_three_image_changes(
        inventory_t0, inventory_clean, inventory_t1, tracking, settings
    )
    timings["04_object_state_resolution"] = time.perf_counter() - started

    if settings.recover_unmatched_via_tracking:
        started = time.perf_counter()
        labels, objects, decisions, recovery_diagnostics = recover_unmatched_via_tracking(
            labels, objects, decisions, tracking, inventory_t0, inventory_t1, sam_t0, dino_t0, sam_t1, dino_t1, settings
        )
        timings["05_tracking_recovery"] = time.perf_counter() - started
        diagnostics = {
            **diagnostics,
            "decision_counts": recovery_diagnostics["decision_counts"],
            "changed_pixel_fraction": recovery_diagnostics["changed_pixel_fraction"],
            "tracking_recovery": recovery_diagnostics,
        }

    save_image(output_dir / "render_t0.png", render_t0)
    save_image(output_dir / "clean_render.png", clean_render)
    save_image(output_dir / "target.png", image_t1)
    save_image(output_dir / "labels.png", labels)
    save_image(output_dir / "labels_color.png", colorize(labels))
    save_image(output_dir / "overlay.png", overlay(image_t1, labels))
    save_image(output_dir / "objects_t0.png", instance_overlay(render_t0, objects_t0))
    save_image(output_dir / "objects_clean.png", instance_overlay(clean_render, objects_clean))
    save_image(output_dir / "objects_t1.png", instance_overlay(image_t1, objects_t1))
    diagnostics = {**diagnostics, "settings": asdict(settings), "timings": timings, "decisions": decisions}
    save_json(output_dir / "inference.json", diagnostics)
    return ThreeImageResult(
        labels=labels, objects=tuple(objects), decisions=tuple(decisions), diagnostics=diagnostics,
        artifacts_dir=output_dir, timings=timings,
    )
