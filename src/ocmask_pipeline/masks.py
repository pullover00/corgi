from __future__ import annotations

import math

import numpy as np

from .types import Label, ObjectMask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return binary-mask intersection over union."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Euclidean distance between two masks' centroids, in
    per-axis image-fraction units. `a` and `b` must share one pixel grid."""
    ay, ax = np.nonzero(a)
    by, bx = np.nonzero(b)
    if not len(ax) or not len(bx):
        return math.inf
    height, width = a.shape
    return math.hypot(
        float(ax.mean() - bx.mean()) / max(width - 1, 1),
        float(ay.mean() - by.mean()) / max(height - 1, 1),
    )


def same_place(
    a: np.ndarray,
    b: np.ndarray,
    minimum_spatial_iou: float,
    maximum_normalized_centroid_distance: float,
) -> bool:
    """Whether two masks on the same pixel grid occupy essentially the same
    location -- the same test `same_place_pairing` applies elsewhere (the
    feature-veto gate's same-place pairing rule), reused here to decide
    MOVED vs UNCHANGED after tracking."""
    if mask_iou(a, b) < minimum_spatial_iou:
        return False
    return centroid_distance(a, b) <= maximum_normalized_centroid_distance


def visible_fraction(mask: np.ndarray, coverage: np.ndarray) -> float:
    """Measure how much of an object lies in geometrically supported pixels."""
    area = np.asarray(mask, bool).sum()
    return float(np.logical_and(mask, coverage).sum() / area) if area else 0.0


def filter_visible(
    objects: list[ObjectMask], coverage: np.ndarray, alpha: float, minimum_area: int
) -> list[ObjectMask]:
    """Keep masks with at least ``alpha`` valid rendered support.

    Appendix A.6 defines visibility as the fraction of a predicted mask lying
    inside valid R0,1 coverage. Therefore ChangeSim's ``alpha=0.8`` requires
    at least 80% of the mask to be supported by rendered geometry.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    retained = []
    for obj in objects:
        mask = np.asarray(obj.mask, bool)
        area = int(mask.sum())
        if area < minimum_area:
            continue
        support_fraction = float(np.logical_and(mask, coverage).sum() / area)
        # Retain the value so reports can distinguish paper visibility
        # filtering from SAM absence or an inferred consistency rule.
        obj.metadata["valid_render_support_fraction"] = support_fraction
        if support_fraction >= alpha:
            retained.append(obj)
    return retained


def annotate_track_support(
    tracks: list[ObjectMask | None],
    destination_coverage: np.ndarray,
) -> None:
    """Record render support for diagnostics without changing track success.

    GOLDILOCS applies its published visibility filter after classification and
    relative to R0,1.  Missing geometry in an intermediate clean render is
    therefore unknown evidence, not proof that an object changed.
    """
    coverage = np.asarray(destination_coverage, dtype=bool)
    for track in tracks:
        if track is None:
            continue
        if np.asarray(track.mask).shape != coverage.shape:
            raise ValueError("track mask and destination coverage must have equal shapes")
        track.metadata["destination_support_fraction"] = visible_fraction(
            track.mask, coverage
        )


def reject_unsupported_tracks(
    tracks: list[ObjectMask | None],
    destination_coverage: np.ndarray,
    alpha: float,
) -> list[ObjectMask | None]:
    """Turn tracks over unsupported render pixels into tracking failures.

    SAM2 is designed to propagate an object through ordinary video and can
    hallucinate a nonempty mask over black gaps in a synthetic point-cloud
    render. A propagated mask is therefore considered present in a rendered
    destination only when at least ``alpha`` of it overlaps valid geometry.
    Existing failures remain ``None``. This check is not used for real RGB
    destinations such as I1, which contain information at every pixel.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    annotate_track_support(tracks, destination_coverage)
    accepted: list[ObjectMask | None] = []
    for track in tracks:
        if track is None:
            accepted.append(None)
            continue
        support = track.metadata["destination_support_fraction"]
        if support >= alpha:
            accepted.append(track)
        else:
            track.metadata.setdefault("rejection_reasons", []).append(
                "insufficient_destination_support"
            )
            accepted.append(None)
    return accepted


def reject_inconsistent_tracks(
    source_masks: list[ObjectMask],
    tracks: list[ObjectMask | None],
    minimum_iou: float,
    area_ratio_bounds: tuple[float, float] | None = None,
) -> list[ObjectMask | None]:
    """Reject propagated masks that no longer overlap their aligned source.

    Every propagation used by the pairwise pipeline is expressed in camera-1
    pixel coordinates: R0,1, R*,1, and I1 are viewpoint aligned. A track that
    jumps to a different object or grows over an unrelated surface can remain
    nonempty and receive a positive SAM object score. Requiring spatial IoU
    prevents those hallucinated masks from being interpreted as object
    survival.

    ``area_ratio_bounds`` is an unpublished, opt-in reproduction ablation
    (default ``None`` reproduces the exact prior behavior). Auditing
    Warehouse_6_Seq_0_2 showed a single global IoU cutoff conflates two
    different populations: masks that jump to an unrelated, differently
    shaped region (extreme target/source area ratio, near-zero IoU) versus
    masks that shrink or grow moderately on the same surface because of a
    boundary disagreement or partial occlusion (area ratio near 1, IoU
    depressed mainly because the object is large). Gating on area ratio in
    addition to a much lower IoU floor targets the former without punishing
    the latter.
    """
    if len(source_masks) != len(tracks):
        raise ValueError("source_masks and tracks must have equal length")
    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be in [0, 1]")
    accepted: list[ObjectMask | None] = []
    for source, track in zip(source_masks, tracks):
        if track is None:
            accepted.append(None)
            continue
        iou = mask_iou(source.mask, track.mask)
        track.metadata["source_target_iou"] = iou
        reasons = []
        if iou < minimum_iou:
            reasons.append("insufficient_source_target_iou")
        if area_ratio_bounds is not None:
            source_area = int(np.asarray(source.mask, bool).sum())
            target_area = int(np.asarray(track.mask, bool).sum())
            ratio = target_area / source_area if source_area else float("inf")
            track.metadata["source_target_area_ratio"] = ratio
            low, high = area_ratio_bounds
            if not (low <= ratio <= high):
                reasons.append("extreme_source_target_area_ratio")
        if reasons:
            track.metadata.setdefault("rejection_reasons", []).extend(reasons)
            accepted.append(None)
        else:
            accepted.append(track)
    return accepted


def compose_labels(
    shape: tuple[int, int],
    objects: list[ObjectMask],
    priority: list[Label] | None = None,
) -> np.ndarray:
    """Rasterize object masks using the paper's deterministic overlap priority."""
    priority = priority or [
        Label.WARPED,
        Label.MOVED,
        Label.REMOVED,
        Label.ADDED,
        Label.UNCHANGED,
    ]
    output = np.full(shape, Label.UNCHANGED, dtype=np.uint8)
    # Lowest priority first, so higher-priority masks overwrite later.
    rank = {label: index for index, label in enumerate(priority)}
    for obj in sorted(objects, key=lambda item: rank.get(item.label, len(priority)), reverse=True):
        output[np.asarray(obj.mask, bool)] = int(obj.label)
    return output


def mark_replacements(labels: np.ndarray, added: list[ObjectMask], removed: list[ObjectMask], threshold: float) -> np.ndarray:
    """Map spatially overlapping additions and removals to ChangeSim replacement."""
    result = labels.copy()
    for add in added:
        for remove in removed:
            overlap = np.logical_and(add.mask, remove.mask)
            union = np.logical_or(add.mask, remove.mask)
            if union.sum() and overlap.sum() / union.sum() >= threshold:
                result[overlap] = Label.REPLACED
    return result
