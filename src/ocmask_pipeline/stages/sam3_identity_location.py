"""Shared descriptor/IoU primitives used by change_detection.py's object
association: mask-pooled feature descriptors and their cosine similarity, and
exact pairwise mask IoU.

This module previously also held a full standalone identity+replacement
classification pipeline (``classify_identity_location`` /
``compose_identity_labels`` / ``associate_identities`` and the "replaced"
object class). It was never called by change_detection.py or any script --
confirmed by a full-repo reference search -- and has been removed along with
the "replaced" concept it implemented (see types.Label and
change_detection.resolve_three_image_changes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from ..types import ObjectMask


@dataclass(frozen=True)
class FeatureDescriptorBatch:
    """Mask-pooled descriptors and their amount of spatial evidence."""

    vectors: np.ndarray
    valid: np.ndarray
    effective_cells: np.ndarray


def mask_descriptors(
    feature_map: np.ndarray,
    objects: Sequence[ObjectMask],
    *,
    minimum_feature_cells: float,
) -> FeatureDescriptorBatch:
    """Mean-pool dense features inside every soft-resized object mask."""

    features = np.asarray(feature_map, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("feature_map must have shape C x H x W")
    channels, grid_height, grid_width = features.shape
    flat = features.reshape(channels, -1)
    flat /= np.maximum(np.linalg.norm(flat, axis=0, keepdims=True), 1e-12)
    if not objects:
        return FeatureDescriptorBatch(
            vectors=np.empty((0, channels), np.float32),
            valid=np.empty(0, bool),
            effective_cells=np.empty(0, np.float32),
        )

    weights = []
    for obj in objects:
        mask = np.asarray(obj.mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError("object masks must be two-dimensional")
        resized = np.asarray(
            Image.fromarray(mask, mode="F").resize(
                (grid_width, grid_height), Image.Resampling.BOX
            ),
            dtype=np.float32,
        )
        weights.append(np.clip(resized, 0.0, 1.0).reshape(-1))
    weight_matrix = np.stack(weights)
    evidence = weight_matrix.sum(axis=1)
    vectors = weight_matrix @ flat.T
    vectors /= np.maximum(evidence[:, None], 1e-12)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    valid = (evidence >= float(minimum_feature_cells)) & np.all(
        np.isfinite(vectors), axis=1
    )
    vectors[~valid] = 0.0
    return FeatureDescriptorBatch(
        vectors=vectors.astype(np.float32),
        valid=valid.astype(bool),
        effective_cells=evidence.astype(np.float32),
    )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pairwise_mask_iou(
    source: Sequence[ObjectMask], target: Sequence[ObjectMask], validity: np.ndarray | None = None
) -> np.ndarray:
    """Compute exact mask IoUs while skipping non-overlapping bounding boxes.

    ``validity`` (optional, same grid as the masks), if given, restricts
    every mask to its True pixels before anything else is computed -- both
    sides, since a shared aligned grid means a coordinate's validity (e.g.
    ``render_t0_coverage``) describes the scene location, not just one
    image. Use this so pixels neither side has trustworthy geometry for
    don't count as an artificial mismatch when one mask is fragmented by
    reconstruction holes the other doesn't have -- see
    change_detection.ThreeImageSettings.same_location_coverage_aware.
    """

    output = np.zeros((len(source), len(target)), dtype=np.float32)
    if not source or not target:
        return output
    source_masks = [np.asarray(obj.mask, dtype=bool) for obj in source]
    target_masks = [np.asarray(obj.mask, dtype=bool) for obj in target]
    shape = source_masks[0].shape
    if any(mask.shape != shape for mask in source_masks + target_masks):
        raise ValueError("all association masks must share one aligned grid")
    if validity is not None:
        validity = np.asarray(validity, dtype=bool)
        if validity.shape != shape:
            raise ValueError("validity must share the masks' aligned grid")
        source_masks = [mask & validity for mask in source_masks]
        target_masks = [mask & validity for mask in target_masks]
    source_boxes = [_bbox(mask) for mask in source_masks]
    target_boxes = [_bbox(mask) for mask in target_masks]
    source_areas = np.asarray([mask.sum() for mask in source_masks], np.int64)
    target_areas = np.asarray([mask.sum() for mask in target_masks], np.int64)
    for source_index, (source_mask, source_box) in enumerate(
        zip(source_masks, source_boxes)
    ):
        sx0, sy0, sx1, sy1 = source_box
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        for target_index, (target_mask, target_box) in enumerate(
            zip(target_masks, target_boxes)
        ):
            tx0, ty0, tx1, ty1 = target_box
            x0, y0 = max(sx0, tx0), max(sy0, ty0)
            x1, y1 = min(sx1, tx1), min(sy1, ty1)
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = int(
                np.logical_and(
                    source_mask[y0:y1, x0:x1], target_mask[y0:y1, x0:x1]
                ).sum()
            )
            union = int(source_areas[source_index] + target_areas[target_index] - intersection)
            if union:
                output[source_index, target_index] = intersection / union
    return output


def cosine_similarity_matrix(
    source: FeatureDescriptorBatch, target: FeatureDescriptorBatch
) -> np.ndarray:
    if source.vectors.shape[1:] != target.vectors.shape[1:]:
        raise ValueError("source and target descriptor dimensions differ")
    if not len(source.vectors) or not len(target.vectors):
        return np.empty((len(source.vectors), len(target.vectors)), np.float32)
    return np.clip(source.vectors @ target.vectors.T, -1.0, 1.0).astype(np.float32)
