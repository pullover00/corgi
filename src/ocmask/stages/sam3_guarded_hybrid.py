"""Guarded fusion of the frozen SAM3+SAM2 and SAM3-feature experiments.

The feature-only identity experiment improved the ``replaced`` class but was
too aggressive when it was allowed to rebuild every semantic decision.  This
module implements a safer, post-classification experiment: the completed
SAM3-mask + SAM2-tracker label map remains the default and SAM3 feature
evidence may edit only explicitly authorised baseline classes.

Two interventions are intentionally disjoint:

* replacement evidence may change only baseline ADDED/REMOVED pixels to
  REPLACED; and
* moved verification may change only baseline MOVED pixels to ADDED/REMOVED.

Consequently every edit remains a changed class.  All variants preserve the
baseline binary changed/unchanged support exactly, and applying the two
interventions in either order produces the same combined result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from ..masks import mask_iou
from ..types import Label


@dataclass(frozen=True)
class HybridEvidence:
    """Native-resolution masks used by the guarded arbitration rules."""

    replacement: np.ndarray
    moved_forward_only_unverified: np.ndarray
    moved_reverse_only_unverified: np.ndarray
    moved_confirmed: np.ndarray
    moved_bidirectional: np.ndarray
    moved_unowned: np.ndarray
    replacement_pair_count: int
    moved_feature_pair_count: int
    forward_owner_count: int
    reverse_owner_count: int
    confirmed_forward_count: int
    confirmed_reverse_count: int


@dataclass(frozen=True)
class HybridDiagnostics:
    """Auditable transition and support accounting for one label map."""

    variant: str
    changed_pixel_count_before: int
    changed_pixel_count_after: int
    binary_support_equal_baseline: bool
    added_to_replaced: int
    removed_to_replaced: int
    moved_to_added: int
    moved_to_removed: int
    total_relabelled: int

    def to_dict(self) -> dict:
        return asdict(self)


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize one discrete evidence mask without interpolating its boundary."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("evidence masks must be two-dimensional")
    if tuple(binary.shape) == tuple(shape):
        return binary.copy()
    return np.asarray(
        Image.fromarray(binary.astype(np.uint8)).resize(
            shape[::-1], Image.Resampling.NEAREST
        ),
        dtype=bool,
    )


def _validate_proposal_ids(
    decisions: Sequence[Mapping],
    source_masks: Sequence[np.ndarray],
    target_masks: Sequence[np.ndarray],
) -> None:
    """Reject stale decisions instead of silently indexing the wrong mask."""

    for record in decisions:
        source_id = int(record["source_proposal_id"])
        target_id = int(record["target_proposal_id"])
        if not 1 <= source_id <= len(source_masks):
            raise ValueError(f"source proposal ID out of range: {source_id}")
        if not 1 <= target_id <= len(target_masks):
            raise ValueError(f"target proposal ID out of range: {target_id}")


def replacement_evidence_mask(
    decisions: Sequence[Mapping],
    source_masks: Sequence[np.ndarray],
    target_masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, int]:
    """Union frozen same-place/different-identity proposal intersections."""

    _validate_proposal_ids(decisions, source_masks, target_masks)
    shape = _common_shape(source_masks, target_masks)
    evidence = np.zeros(shape, dtype=bool)
    pair_count = 0
    for record in decisions:
        if str(record.get("decision")) != "replaced":
            continue
        source_id = int(record["source_proposal_id"])
        target_id = int(record["target_proposal_id"])
        evidence |= np.asarray(source_masks[source_id - 1], bool) & np.asarray(
            target_masks[target_id - 1], bool
        )
        pair_count += 1
    return evidence, pair_count


def _common_shape(
    source_masks: Sequence[np.ndarray], target_masks: Sequence[np.ndarray]
) -> tuple[int, int]:
    masks = list(source_masks) + list(target_masks)
    if not masks:
        raise ValueError("at least one proposal mask is required")
    shape = np.asarray(masks[0]).shape
    if len(shape) != 2 or any(np.asarray(mask).shape != shape for mask in masks):
        raise ValueError("all proposal masks must share one two-dimensional grid")
    return int(shape[0]), int(shape[1])


def moved_verification_evidence(
    decisions: Sequence[Mapping],
    source_masks: Sequence[np.ndarray],
    target_masks: Sequence[np.ndarray],
    forward_tracks: Mapping[int, np.ndarray],
    reverse_tracks: Mapping[int, np.ndarray],
    *,
    minimum_track_candidate_iou: float,
    native_shape: tuple[int, int],
) -> dict[str, np.ndarray | int]:
    """Build exact moved ownership and feature-confirmation regions.

    ``forward_tracks`` maps a source proposal ID to its accepted target-frame
    SAM2 raster. ``reverse_tracks`` maps a target proposal ID to its accepted
    source-frame SAM2 raster.  The frozen feature decision confirms a
    direction only when that exact track also overlaps the associated
    opposite proposal by the existing tracking-IoU floor.

    Pixels supported by both baseline directions are kept as MOVED even when
    no feature pair is available. This conservative rule treats absence of a
    feature match as uncertainty rather than proof that tracking failed.
    """

    if not 0.0 <= float(minimum_track_candidate_iou) <= 1.0:
        raise ValueError("minimum_track_candidate_iou must be in [0, 1]")
    _validate_proposal_ids(decisions, source_masks, target_masks)
    shape = _common_shape(source_masks, target_masks)

    forward_owner = np.zeros(shape, dtype=bool)
    reverse_owner = np.zeros(shape, dtype=bool)
    for source_id, track in forward_tracks.items():
        if not 1 <= int(source_id) <= len(source_masks):
            raise ValueError(f"forward source proposal ID out of range: {source_id}")
        track = np.asarray(track, dtype=bool)
        if track.shape != shape:
            raise ValueError("forward track shape differs from proposal grid")
        forward_owner |= track
    for target_id, track in reverse_tracks.items():
        if not 1 <= int(target_id) <= len(target_masks):
            raise ValueError(f"reverse target proposal ID out of range: {target_id}")
        track = np.asarray(track, dtype=bool)
        if track.shape != shape:
            raise ValueError("reverse track shape differs from proposal grid")
        # The baseline emits the original target proposal for a successful
        # target->source track, not the backward raster itself.
        reverse_owner |= np.asarray(target_masks[int(target_id) - 1], bool)

    confirmed = np.zeros(shape, dtype=bool)
    moved_pair_count = 0
    confirmed_forward = 0
    confirmed_reverse = 0
    for record in decisions:
        if str(record.get("decision")) != "moved":
            continue
        moved_pair_count += 1
        source_id = int(record["source_proposal_id"])
        target_id = int(record["target_proposal_id"])
        if source_id in forward_tracks:
            track = np.asarray(forward_tracks[source_id], dtype=bool)
            if mask_iou(track, np.asarray(target_masks[target_id - 1], bool)) >= float(
                minimum_track_candidate_iou
            ):
                confirmed |= track
                confirmed_forward += 1
        if target_id in reverse_tracks:
            track = np.asarray(reverse_tracks[target_id], dtype=bool)
            if mask_iou(track, np.asarray(source_masks[source_id - 1], bool)) >= float(
                minimum_track_candidate_iou
            ):
                confirmed |= np.asarray(target_masks[target_id - 1], bool)
                confirmed_reverse += 1

    bidirectional = forward_owner & reverse_owner
    forward_only_unverified = forward_owner & ~reverse_owner & ~confirmed
    reverse_only_unverified = reverse_owner & ~forward_owner & ~confirmed
    owned = forward_owner | reverse_owner

    return {
        "forward_only_unverified": resize_mask_nearest(
            forward_only_unverified, native_shape
        ),
        "reverse_only_unverified": resize_mask_nearest(
            reverse_only_unverified, native_shape
        ),
        "confirmed": resize_mask_nearest(confirmed, native_shape),
        "bidirectional": resize_mask_nearest(bidirectional, native_shape),
        # The caller intersects this with the parent MOVED raster.  It is kept
        # explicitly so reports can show numerical parent pixels that no
        # replayed object owns and which therefore remain untouched.
        "owned": resize_mask_nearest(owned, native_shape),
        "moved_feature_pair_count": moved_pair_count,
        "forward_owner_count": len(forward_tracks),
        "reverse_owner_count": len(reverse_tracks),
        "confirmed_forward_count": confirmed_forward,
        "confirmed_reverse_count": confirmed_reverse,
    }


def make_hybrid_evidence(
    baseline: np.ndarray,
    decisions: Sequence[Mapping],
    source_masks: Sequence[np.ndarray],
    target_masks: Sequence[np.ndarray],
    forward_tracks: Mapping[int, np.ndarray],
    reverse_tracks: Mapping[int, np.ndarray],
    *,
    minimum_track_candidate_iou: float,
) -> HybridEvidence:
    """Construct every native-resolution mask once for all ablation variants."""

    labels = np.asarray(baseline, dtype=np.uint8)
    replacement, replacement_count = replacement_evidence_mask(
        decisions, source_masks, target_masks
    )
    moved = moved_verification_evidence(
        decisions,
        source_masks,
        target_masks,
        forward_tracks,
        reverse_tracks,
        minimum_track_candidate_iou=minimum_track_candidate_iou,
        native_shape=labels.shape,
    )
    owned = np.asarray(moved["owned"], bool)
    return HybridEvidence(
        replacement=resize_mask_nearest(replacement, labels.shape),
        moved_forward_only_unverified=np.asarray(
            moved["forward_only_unverified"], bool
        ),
        moved_reverse_only_unverified=np.asarray(
            moved["reverse_only_unverified"], bool
        ),
        moved_confirmed=np.asarray(moved["confirmed"], bool),
        moved_bidirectional=np.asarray(moved["bidirectional"], bool),
        moved_unowned=(labels == int(Label.MOVED)) & ~owned,
        replacement_pair_count=replacement_count,
        moved_feature_pair_count=int(moved["moved_feature_pair_count"]),
        forward_owner_count=int(moved["forward_owner_count"]),
        reverse_owner_count=int(moved["reverse_owner_count"]),
        confirmed_forward_count=int(moved["confirmed_forward_count"]),
        confirmed_reverse_count=int(moved["confirmed_reverse_count"]),
    )


def compose_guarded_variant(
    baseline: np.ndarray,
    evidence: HybridEvidence,
    variant: str,
) -> tuple[np.ndarray, HybridDiagnostics]:
    """Apply one frozen guarded variant and enforce its transition contract."""

    allowed = {
        "replacement_only",
        "moved_verification",
        "combined_guarded_hybrid",
    }
    if variant not in allowed:
        raise ValueError(f"unknown guarded hybrid variant: {variant}")
    original = np.asarray(baseline, dtype=np.uint8)
    if original.ndim != 2:
        raise ValueError("baseline labels must be two-dimensional")
    for mask in (
        evidence.replacement,
        evidence.moved_forward_only_unverified,
        evidence.moved_reverse_only_unverified,
        evidence.moved_confirmed,
        evidence.moved_bidirectional,
        evidence.moved_unowned,
    ):
        if np.asarray(mask).shape != original.shape:
            raise ValueError("hybrid evidence and baseline labels must share a shape")
    output = original.copy()

    replacement_enabled = variant in {
        "replacement_only",
        "combined_guarded_hybrid",
    }
    moved_enabled = variant in {
        "moved_verification",
        "combined_guarded_hybrid",
    }

    # Eligibility is always evaluated against the immutable parent labels.
    # This makes the two edits disjoint and therefore order-independent.
    if moved_enabled:
        parent_moved = original == int(Label.MOVED)
        output[parent_moved & evidence.moved_forward_only_unverified] = int(
            Label.REMOVED
        )
        output[parent_moved & evidence.moved_reverse_only_unverified] = int(
            Label.ADDED
        )
    if replacement_enabled:
        eligible = (original == int(Label.ADDED)) | (
            original == int(Label.REMOVED)
        )
        output[eligible & evidence.replacement] = int(Label.REPLACED)

    binary_equal = np.array_equal(
        output != int(Label.UNCHANGED), original != int(Label.UNCHANGED)
    )
    if not binary_equal:
        raise AssertionError("guarded hybrid changed the binary support")
    changed = output != original
    # Reject any accidental transition not declared above.
    authorised = np.zeros(original.shape, dtype=bool)
    if replacement_enabled:
        authorised |= (
            ((original == int(Label.ADDED)) | (original == int(Label.REMOVED)))
            & (output == int(Label.REPLACED))
        )
    if moved_enabled:
        authorised |= (original == int(Label.MOVED)) & (
            (output == int(Label.ADDED)) | (output == int(Label.REMOVED))
        )
    if np.any(changed & ~authorised):
        raise AssertionError("hybrid produced an unauthorised class transition")

    diagnostics = HybridDiagnostics(
        variant=variant,
        changed_pixel_count_before=int(
            np.count_nonzero(original != int(Label.UNCHANGED))
        ),
        changed_pixel_count_after=int(
            np.count_nonzero(output != int(Label.UNCHANGED))
        ),
        binary_support_equal_baseline=binary_equal,
        added_to_replaced=int(
            np.count_nonzero(
                (original == int(Label.ADDED)) & (output == int(Label.REPLACED))
            )
        ),
        removed_to_replaced=int(
            np.count_nonzero(
                (original == int(Label.REMOVED))
                & (output == int(Label.REPLACED))
            )
        ),
        moved_to_added=int(
            np.count_nonzero(
                (original == int(Label.MOVED)) & (output == int(Label.ADDED))
            )
        ),
        moved_to_removed=int(
            np.count_nonzero(
                (original == int(Label.MOVED)) & (output == int(Label.REMOVED))
            )
        ),
        total_relabelled=int(np.count_nonzero(changed)),
    )
    return output, diagnostics

