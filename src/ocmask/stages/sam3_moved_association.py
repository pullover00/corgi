"""Reciprocal moved-object association for standalone SAM experiments.

GOLDILOCS classifies an object as moved when a proposal which disappeared
from one temporal view can be found in the other view.  A one-way propagation
can jump to a similar object, however.  This experiment therefore combines
two independently computed pieces of evidence for every source/target pair:

* the IoU between a source candidate propagated *forward* and a target
  candidate in the target frame; and
* the IoU between that target candidate propagated *backward* and the source
  candidate in the source frame.

The module is deliberately model- and dataset-independent.  It receives
already computed candidates/tracks, performs only NumPy/SciPy CPU work, and
never imports or changes the production pairwise pipeline.  Three variants
make the scientific ablation explicit:

``reciprocal_mutual_overlap``
    Keep only mutual row/column best pairs which pass both directional IoUs.

``hungarian_one_to_one``
    Maximize the total reciprocal score with a one-to-one Hungarian assignment.

``hungarian_motion_verified``
    Solve the same reciprocal Hungarian assignment, then suppress matched
    source/target pairs whose aligned masks still describe the same location.

Every returned moved mask is a copy of the *target-frame candidate*, never a
forward-track raster.  This preserves the pipeline's target-frame output
contract and avoids turning tracking boundary noise into the final label map.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..masks import mask_iou
from ..types import Label, ObjectMask


class AssociationVariant(str, Enum):
    """Frozen association variants used by the standalone experiment."""

    RECIPROCAL_MUTUAL_OVERLAP = "reciprocal_mutual_overlap"
    HUNGARIAN_ONE_TO_ONE = "hungarian_one_to_one"
    HUNGARIAN_MOTION_VERIFIED = "hungarian_motion_verified"


@dataclass(frozen=True)
class MovedAssociationSettings:
    """Predeclared thresholds for reciprocal association and motion evidence.

    ``minimum_forward_iou`` and ``minimum_backward_iou`` are evaluated
    independently; a high score in one direction cannot compensate for a
    failed propagation in the other direction.  The reciprocal score is the
    conservative minimum of the two IoUs and is gated separately.

    Motion verification assumes source and target masks use aligned pixel
    coordinates, as R0,1 and I1 do in this project.  It deliberately reuses
    the versioned no-3D movement cutoff: aligned IoU below
    ``maximum_static_iou`` is moved, while equality or greater is treated as
    the same-location/static explanation.  Centroid displacement remains a
    diagnostic only; adding a pixel threshold would be a separate ablation.
    """

    minimum_forward_iou: float = 0.20
    minimum_backward_iou: float = 0.20
    minimum_reciprocal_score: float = 0.20
    maximum_static_iou: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "minimum_forward_iou",
            "minimum_backward_iou",
            "minimum_reciprocal_score",
            "maximum_static_iou",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


DEFAULT_SETTINGS = MovedAssociationSettings()


@dataclass(frozen=True)
class AssociationMatch:
    """One accepted source/target association with all decision evidence."""

    source_index: int
    target_index: int
    source_proposal_id: int
    target_proposal_id: int
    forward_iou: float
    backward_iou: float
    reciprocal_score: float
    aligned_source_target_iou: float
    centroid_displacement_pixels: float
    motion_verified: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe row for experiment reports."""

        return asdict(self)


@dataclass(frozen=True)
class MovedAssociationDiagnostics:
    """Complete score matrices and assignment accounting for one run."""

    variant: str
    settings: dict[str, float]
    source_count: int
    target_count: int
    forward_track_available: tuple[bool, ...]
    backward_track_available: tuple[bool, ...]
    forward_iou: np.ndarray
    backward_iou: np.ndarray
    reciprocal_score: np.ndarray
    aligned_source_target_iou: np.ndarray
    centroid_displacement_pixels: np.ndarray
    reciprocal_eligible: np.ndarray
    motion_verified: np.ndarray
    assignment_eligible: np.ndarray
    assigned_pairs: tuple[tuple[int, int], ...]
    selected_pairs: tuple[tuple[int, int], ...]
    suppressed_static_pairs: tuple[tuple[int, int], ...]
    unmatched_source_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        """Serialize diagnostics without hiding the pairwise score matrices."""

        return {
            "variant": self.variant,
            "settings": dict(self.settings),
            "source_count": self.source_count,
            "target_count": self.target_count,
            "forward_track_available": list(self.forward_track_available),
            "backward_track_available": list(self.backward_track_available),
            "forward_iou": self.forward_iou.tolist(),
            "backward_iou": self.backward_iou.tolist(),
            "reciprocal_score": self.reciprocal_score.tolist(),
            "aligned_source_target_iou": self.aligned_source_target_iou.tolist(),
            "centroid_displacement_pixels": self.centroid_displacement_pixels.tolist(),
            "reciprocal_eligible": self.reciprocal_eligible.tolist(),
            "motion_verified": self.motion_verified.tolist(),
            "assignment_eligible": self.assignment_eligible.tolist(),
            "assigned_pairs": [list(pair) for pair in self.assigned_pairs],
            "selected_pairs": [list(pair) for pair in self.selected_pairs],
            "suppressed_static_pairs": [
                list(pair) for pair in self.suppressed_static_pairs
            ],
            "unmatched_source_indices": list(self.unmatched_source_indices),
            "unmatched_target_indices": list(self.unmatched_target_indices),
        }


@dataclass(frozen=True)
class MovedAssociationResult:
    """Target-frame moved masks plus machine-readable association evidence."""

    moved_masks: tuple[ObjectMask, ...]
    matches: tuple[AssociationMatch, ...]
    diagnostics: MovedAssociationDiagnostics

    def summary(self) -> dict[str, Any]:
        """Return report metadata while leaving mask pixels out of JSON."""

        return {
            "matches": [match.to_dict() for match in self.matches],
            "diagnostics": self.diagnostics.summary(),
            "moved_mask_count": len(self.moved_masks),
        }


def _variant(value: AssociationVariant | str) -> AssociationVariant:
    """Normalize a CLI-friendly string into the strict variant enum."""

    if isinstance(value, AssociationVariant):
        return value
    try:
        return AssociationVariant(str(value))
    except ValueError as exc:
        choices = ", ".join(item.value for item in AssociationVariant)
        raise ValueError(f"unknown association variant {value!r}; choose {choices}") from exc


def _candidate_masks(
    name: str, candidates: Sequence[ObjectMask]
) -> tuple[list[np.ndarray], tuple[int, int] | None]:
    """Validate non-empty candidate masks and their common frame shape."""

    masks: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for index, candidate in enumerate(candidates):
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"{name}[{index}] mask must be two-dimensional")
        if not np.any(mask):
            raise ValueError(f"{name}[{index}] mask must contain at least one pixel")
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            raise ValueError(f"all {name} masks must have equal shapes")
        masks.append(mask)
    return masks, shape


def _track_masks(
    name: str,
    tracks: Sequence[ObjectMask | np.ndarray | None],
    expected_count: int,
    expected_shape: tuple[int, int] | None,
) -> tuple[list[np.ndarray | None], tuple[bool, ...]]:
    """Normalize optional propagated masks without treating empty tracks as valid."""

    if len(tracks) != expected_count:
        raise ValueError(
            f"{name} must contain exactly one item per candidate "
            f"({expected_count}), got {len(tracks)}"
        )
    masks: list[np.ndarray | None] = []
    available = []
    for index, track in enumerate(tracks):
        if track is None:
            masks.append(None)
            available.append(False)
            continue
        raw = track.mask if isinstance(track, ObjectMask) else track
        mask = np.asarray(raw, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"{name}[{index}] must be two-dimensional")
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError(
                f"{name}[{index}] must have target shape {expected_shape}, got {mask.shape}"
            )
        if not np.any(mask):
            masks.append(None)
            available.append(False)
        else:
            masks.append(mask)
            available.append(True)
    return masks, tuple(available)


def _proposal_id(candidate: ObjectMask, fallback_index: int) -> int:
    """Recover stable proposal provenance while tolerating neutral test masks."""

    for key in ("automatic_proposal_id", "proposal_id", "candidate_id"):
        if key in candidate.metadata:
            return int(candidate.metadata[key])
    return int(fallback_index)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return an (x, y) centroid; candidates are guaranteed non-empty."""

    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def compute_reciprocal_score_matrices(
    source_candidates: Sequence[ObjectMask],
    source_forward_tracks: Sequence[ObjectMask | np.ndarray | None],
    target_candidates: Sequence[ObjectMask],
    target_backward_tracks: Sequence[ObjectMask | np.ndarray | None],
    *,
    settings: MovedAssociationSettings = DEFAULT_SETTINGS,
) -> dict[str, Any]:
    """Compute all directional, reciprocal, and motion score matrices.

    Rows correspond to disappeared source candidates and columns correspond to
    appeared target candidates.  Missing/empty tracks receive zero directional
    IoU and can never pass ``reciprocal_eligible``.
    """

    sources, source_shape = _candidate_masks("source_candidates", source_candidates)
    targets, target_shape = _candidate_masks("target_candidates", target_candidates)
    forward, forward_available = _track_masks(
        "source_forward_tracks",
        source_forward_tracks,
        len(sources),
        target_shape,
    )
    backward, backward_available = _track_masks(
        "target_backward_tracks",
        target_backward_tracks,
        len(targets),
        source_shape,
    )

    shape = (len(sources), len(targets))
    forward_iou = np.zeros(shape, dtype=np.float64)
    backward_iou = np.zeros(shape, dtype=np.float64)
    aligned_iou = np.full(shape, np.nan, dtype=np.float64)
    centroid_displacement = np.full(shape, np.nan, dtype=np.float64)
    aligned_frames = source_shape is not None and source_shape == target_shape

    source_centroids = [_centroid(mask) for mask in sources]
    target_centroids = [_centroid(mask) for mask in targets]
    for source_index, source_mask in enumerate(sources):
        for target_index, target_mask in enumerate(targets):
            if forward[source_index] is not None:
                forward_iou[source_index, target_index] = mask_iou(
                    forward[source_index], target_mask
                )
            if backward[target_index] is not None:
                backward_iou[source_index, target_index] = mask_iou(
                    backward[target_index], source_mask
                )
            if aligned_frames:
                aligned_iou[source_index, target_index] = mask_iou(
                    source_mask, target_mask
                )
                sx, sy = source_centroids[source_index]
                tx, ty = target_centroids[target_index]
                centroid_displacement[source_index, target_index] = float(
                    np.hypot(tx - sx, ty - sy)
                )

    # The weaker propagation direction determines the score.  This is more
    # conservative than an average or geometric mean and matches the frozen
    # experiment plan declared before evaluating ChangeSim ground truth.
    reciprocal = np.minimum(forward_iou, backward_iou)
    reciprocal_eligible = (
        (forward_iou >= settings.minimum_forward_iou)
        & (backward_iou >= settings.minimum_backward_iou)
        & (reciprocal >= settings.minimum_reciprocal_score)
    )
    if aligned_frames:
        # Strictly below 0.5 follows the existing no-3D movement rule.  The
        # equality boundary is conservative and therefore static-suppressed.
        motion_verified = aligned_iou < settings.maximum_static_iou
    else:
        # Motion verification cannot compare pixel locations across differently
        # shaped coordinate systems.  Other variants remain usable.
        motion_verified = np.zeros(shape, dtype=bool)

    return {
        "forward_iou": forward_iou,
        "backward_iou": backward_iou,
        "reciprocal_score": reciprocal,
        "aligned_source_target_iou": aligned_iou,
        "centroid_displacement_pixels": centroid_displacement,
        "reciprocal_eligible": reciprocal_eligible,
        "motion_verified": motion_verified,
        "forward_track_available": forward_available,
        "backward_track_available": backward_available,
        "source_shape": source_shape,
        "target_shape": target_shape,
    }


def solve_one_to_one_assignments(
    reciprocal_score: np.ndarray,
    eligible: np.ndarray,
    *,
    variant: AssociationVariant | str,
) -> tuple[tuple[int, int], ...]:
    """Solve a deterministic mutual-best or Hungarian one-to-one assignment.

    ``eligible`` is always enforced after solving, so rectangular matrices and
    rows/columns with no candidate cannot manufacture a zero-score match.
    Mutual-best ties use NumPy's stable first argmax.  Hungarian receives a
    tiny deterministic non-separable perturbation so equal-score alternatives
    remain repeatable without materially changing the objective.
    """

    selected_variant = _variant(variant)
    scores = np.asarray(reciprocal_score, dtype=np.float64)
    allowed = np.asarray(eligible, dtype=bool)
    if scores.ndim != 2 or allowed.shape != scores.shape:
        raise ValueError("reciprocal_score and eligible must be equal 2D matrices")
    if np.any(~np.isfinite(scores)):
        raise ValueError("reciprocal_score must contain only finite values")
    rows, columns = scores.shape
    if not rows or not columns or not np.any(allowed):
        return ()

    if selected_variant == AssociationVariant.RECIPROCAL_MUTUAL_OVERLAP:
        masked = np.where(allowed, scores, -np.inf)
        row_best = np.argmax(masked, axis=1)
        column_best = np.argmax(masked, axis=0)
        pairs = [
            (source_index, int(target_index))
            for source_index, target_index in enumerate(row_best)
            if allowed[source_index, target_index]
            and column_best[target_index] == source_index
        ]
        return tuple(pairs)

    # Minimize negative reciprocal score.  Invalid edges receive a cost much
    # larger than any valid [-1, 0] score.  The perturbation makes equal-score
    # solutions reproducible without materially changing the objective.
    invalid_cost = 1_000_000.0
    source_rank = np.arange(rows, dtype=np.float64)[:, None]
    target_rank = np.arange(columns, dtype=np.float64)[None, :]
    linear_rank = source_rank * max(columns, 1) + target_rank + 1.0
    # Squaring prevents the total perturbation from becoming a constant row
    # plus column sum for complete square assignments.
    tie_break = linear_rank**2 * (1e-12 / max(float((rows * columns) ** 2), 1.0))
    cost = np.where(allowed, -scores + tie_break, invalid_cost + tie_break)
    source_indices, target_indices = linear_sum_assignment(cost)
    pairs = [
        (int(source_index), int(target_index))
        for source_index, target_index in zip(
            source_indices, target_indices, strict=True
        )
        if allowed[source_index, target_index]
    ]
    pairs.sort()
    return tuple(pairs)


def associate_moved_objects(
    source_candidates: Sequence[ObjectMask],
    source_forward_tracks: Sequence[ObjectMask | np.ndarray | None],
    target_candidates: Sequence[ObjectMask],
    target_backward_tracks: Sequence[ObjectMask | np.ndarray | None],
    *,
    variant: AssociationVariant | str = AssociationVariant.RECIPROCAL_MUTUAL_OVERLAP,
    settings: MovedAssociationSettings = DEFAULT_SETTINGS,
) -> MovedAssociationResult:
    """Associate disappeared/appeared candidates and emit target-frame masks.

    The function does not classify unmatched candidates; callers remain free
    to label unmatched sources as removed and unmatched targets as added using
    their frozen baseline protocol.
    """

    selected_variant = _variant(variant)
    matrices = compute_reciprocal_score_matrices(
        source_candidates,
        source_forward_tracks,
        target_candidates,
        target_backward_tracks,
        settings=settings,
    )
    eligible = matrices["reciprocal_eligible"].copy()
    if selected_variant == AssociationVariant.HUNGARIAN_MOTION_VERIFIED:
        # With one empty candidate set there is nothing to verify or match;
        # return ordinary unmatched accounting instead of requiring a frame
        # shape which cannot be inferred from an empty list.
        both_sides_present = bool(source_candidates) and bool(target_candidates)
        if both_sides_present and matrices["source_shape"] != matrices["target_shape"]:
            raise ValueError(
                "motion verification requires aligned source/target mask shapes"
            )
    assigned_pairs = solve_one_to_one_assignments(
        matrices["reciprocal_score"],
        eligible,
        variant=selected_variant,
    )
    if selected_variant == AssociationVariant.HUNGARIAN_MOTION_VERIFIED:
        pairs = tuple(
            pair for pair in assigned_pairs if matrices["motion_verified"][pair]
        )
        suppressed_static_pairs = tuple(
            pair for pair in assigned_pairs if not matrices["motion_verified"][pair]
        )
    else:
        pairs = assigned_pairs
        suppressed_static_pairs = ()
    matches: list[AssociationMatch] = []
    moved_masks: list[ObjectMask] = []
    for source_index, target_index in pairs:
        match = AssociationMatch(
            source_index=source_index,
            target_index=target_index,
            source_proposal_id=_proposal_id(
                source_candidates[source_index], source_index
            ),
            target_proposal_id=_proposal_id(
                target_candidates[target_index], target_index
            ),
            forward_iou=float(matrices["forward_iou"][source_index, target_index]),
            backward_iou=float(matrices["backward_iou"][source_index, target_index]),
            reciprocal_score=float(
                matrices["reciprocal_score"][source_index, target_index]
            ),
            aligned_source_target_iou=float(
                matrices["aligned_source_target_iou"][source_index, target_index]
            ),
            centroid_displacement_pixels=float(
                matrices["centroid_displacement_pixels"][source_index, target_index]
            ),
            motion_verified=bool(
                matrices["motion_verified"][source_index, target_index]
            ),
        )
        matches.append(match)

        # Copy both pixels and metadata so association diagnostics cannot
        # mutate the cached target proposal owned by another experiment.
        target = target_candidates[target_index]
        metadata = copy.deepcopy(target.metadata)
        metadata["moved_association"] = match.to_dict()
        metadata["association_variant"] = selected_variant.value
        moved_masks.append(
            ObjectMask(
                mask=np.asarray(target.mask, dtype=bool).copy(),
                score=match.reciprocal_score,
                label=Label.MOVED,
                source="sam3_reciprocal_moved_association",
                metadata=metadata,
            )
        )

    # A static-suppressed association is explained as unchanged.  It must not
    # fall back to unmatched removed+added labels, which would leave the same
    # binary false positive under a different multiclass name.
    accounted_sources = {source_index for source_index, _ in assigned_pairs}
    accounted_targets = {target_index for _, target_index in assigned_pairs}
    diagnostics = MovedAssociationDiagnostics(
        variant=selected_variant.value,
        settings=asdict(settings),
        source_count=len(source_candidates),
        target_count=len(target_candidates),
        forward_track_available=matrices["forward_track_available"],
        backward_track_available=matrices["backward_track_available"],
        forward_iou=matrices["forward_iou"],
        backward_iou=matrices["backward_iou"],
        reciprocal_score=matrices["reciprocal_score"],
        aligned_source_target_iou=matrices["aligned_source_target_iou"],
        centroid_displacement_pixels=matrices["centroid_displacement_pixels"],
        reciprocal_eligible=matrices["reciprocal_eligible"],
        motion_verified=matrices["motion_verified"],
        assignment_eligible=eligible,
        assigned_pairs=assigned_pairs,
        selected_pairs=pairs,
        suppressed_static_pairs=suppressed_static_pairs,
        unmatched_source_indices=tuple(
            index
            for index in range(len(source_candidates))
            if index not in accounted_sources
        ),
        unmatched_target_indices=tuple(
            index
            for index in range(len(target_candidates))
            if index not in accounted_targets
        ),
    )
    return MovedAssociationResult(
        moved_masks=tuple(moved_masks),
        matches=tuple(matches),
        diagnostics=diagnostics,
    )
