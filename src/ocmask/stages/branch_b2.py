"""Consolidation of nested/overlapping proposal fragments into object hypotheses.

Upstream SAM3 automatic mask generation frequently returns several nested or
near-duplicate proposals for the same physical object (a shelf and its front
face, a box and its lid). Before any identity or replacement decision can be
made, those fragments need to be merged into one hypothesis per object, with
its forward/backward SAM2 propagation status attached.

This module originally also carried a reciprocal cycle-consistency matcher
and an internal-feature rigid-pose estimator for a "Branch B2" diagnostic
that never fed a shipped prediction (see the project history in the parent
GOLDILOCS research repository). Only ``ConsolidationSettings`` and
``consolidate_hypotheses`` are load-bearing for the object-consistent
replacement pipeline; the diagnostic-only code has been dropped here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..types import Label, ObjectMask


@dataclass(frozen=True)
class ConsolidationSettings:
    minimum_iou: float = 0.45
    minimum_containment: float = 0.82
    maximum_centroid_distance_fraction: float = 0.08


@dataclass(frozen=True)
class ObjectHypothesis:
    index: int
    member_indices: tuple[int, ...]
    proposal_ids: tuple[int, ...]
    mask: np.ndarray
    propagated_mask: np.ndarray | None
    track_status: str  # present | absent | ambiguous
    area_pixels: int

    def as_object(self) -> ObjectMask:
        return ObjectMask(
            mask=self.mask.copy(),
            score=1.0,
            label=Label.UNCHANGED,
            source="branch_b2_consolidated_hypothesis",
            metadata={
                "hypothesis_index": self.index,
                "member_indices": list(self.member_indices),
                "proposal_ids": list(self.proposal_ids),
                "track_status": self.track_status,
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "member_indices": list(self.member_indices),
            "proposal_ids": list(self.proposal_ids),
            "track_status": self.track_status,
            "area_pixels": self.area_pixels,
            "propagated_area_pixels": (
                None if self.propagated_mask is None else int(self.propagated_mask.sum())
            ),
        }


def _proposal_id(obj: ObjectMask, fallback: int) -> int:
    for key in ("automatic_proposal_id", "proposal_id", "candidate_id"):
        if key in obj.metadata:
            return int(obj.metadata[key])
    return fallback


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.asarray([xs.mean(), ys.mean()], np.float64)


def _should_merge(a: np.ndarray, b: np.ndarray, settings: ConsolidationSettings) -> bool:
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return False
    area_a, area_b = int(a.sum()), int(b.sum())
    union = area_a + area_b - intersection
    iou = intersection / max(union, 1)
    containment = intersection / max(min(area_a, area_b), 1)
    diagonal = float(np.hypot(*a.shape))
    centroid_distance = float(np.linalg.norm(_centroid(a) - _centroid(b))) / max(diagonal, 1.0)
    return iou >= settings.minimum_iou or (
        containment >= settings.minimum_containment
        and centroid_distance <= settings.maximum_centroid_distance_fraction
    )


def consolidate_hypotheses(
    objects: Sequence[ObjectMask],
    propagated_tracks: Sequence[np.ndarray | None],
    track_rows: Sequence[dict[str, Any]],
    *,
    settings: ConsolidationSettings = ConsolidationSettings(),
) -> tuple[ObjectHypothesis, ...]:
    """Merge overlapping/nested proposal fragments before association.

    A hypothesis is ``absent`` only when every member was explicitly rejected
    as ``object_absent``.  Other rejected tracks remain ambiguous; this keeps a
    low-quality propagation from masquerading as replacement evidence.
    """

    if len(objects) != len(propagated_tracks) or len(objects) != len(track_rows):
        raise ValueError("objects, propagated_tracks, and track_rows must align")
    if not objects:
        return ()
    masks = [np.asarray(item.mask, bool) for item in objects]
    shape = masks[0].shape
    if any(mask.ndim != 2 or mask.shape != shape or not mask.any() for mask in masks):
        raise ValueError("all object masks must be non-empty and share one 2D shape")

    parent = list(range(len(objects)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            if _should_merge(masks[left], masks[right], settings):
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(objects)):
        groups.setdefault(find(index), []).append(index)

    hypotheses: list[ObjectHypothesis] = []
    for members in sorted(groups.values(), key=lambda value: value[0]):
        merged = np.logical_or.reduce([masks[index] for index in members])
        accepted_tracks = [
            np.asarray(propagated_tracks[index], bool)
            for index in members
            if propagated_tracks[index] is not None
        ]
        propagated = np.logical_or.reduce(accepted_tracks) if accepted_tracks else None
        if accepted_tracks:
            status = "present"
        else:
            explicitly_absent = []
            for index in members:
                reasons = {str(reason) for reason in track_rows[index].get("rejection_reasons", [])}
                explicitly_absent.append("object_absent" in reasons)
            status = "absent" if all(explicitly_absent) else "ambiguous"
        hypotheses.append(
            ObjectHypothesis(
                index=len(hypotheses),
                member_indices=tuple(members),
                proposal_ids=tuple(_proposal_id(objects[index], index) for index in members),
                mask=merged,
                propagated_mask=propagated,
                track_status=status,
                area_pixels=int(merged.sum()),
            )
        )
    return tuple(hypotheses)
