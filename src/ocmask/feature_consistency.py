"""Mask-level consistency measurements for dense image feature maps.

The functions in this module deliberately do not import SAM2 or PyTorch.
Model adapters should convert an encoder tensor to a NumPy ``C x H x W``
array before calling this code. Keeping the measurement model-independent
makes the inferred feature gate easy to unit-test and replace.

SAM2's feature grid is much smaller than the input image. A binary image mask
is therefore resized with area averaging instead of nearest-neighbour
sampling. The resulting soft occupancy says how much of each feature cell is
covered by the object and prevents thin or boundary masks from disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .types import ObjectMask


@dataclass(frozen=True)
class PreparedFeatureMap:
    """A channel-first feature map normalized once for repeated mask pooling.

    ``unit_features`` contains an L2-normalized descriptor at every valid
    spatial cell. ``valid_cells`` is false where the original descriptor had
    effectively zero norm. Such cells contribute no evidence to a mask.
    """

    unit_features: np.ndarray
    valid_cells: np.ndarray

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the spatial ``(height, width)`` of the feature grid."""
        return tuple(int(value) for value in self.valid_cells.shape)


@dataclass(frozen=True)
class MaskedFeatureDescriptor:
    """One pooled object descriptor plus evidence-quality diagnostics."""

    vector: np.ndarray | None
    effective_occupancy: float
    occupied_cells: int
    valid: bool
    invalid_reason: str | None = None


@dataclass(frozen=True)
class MaskedFeatureComparison:
    """Cosine comparison of source and target object descriptors."""

    cosine_similarity: float | None
    source_effective_occupancy: float
    target_effective_occupancy: float
    source_occupied_cells: int
    target_occupied_cells: int
    source_grid_shape: tuple[int, int]
    target_grid_shape: tuple[int, int]
    valid: bool
    invalid_reason: str | None = None


@dataclass(frozen=True)
class PairFeatureCalibration:
    """Threshold inferred from static-looking controls in one image pair."""

    threshold: float | None
    positive_acceptance: float | None
    negative_acceptance: float | None
    positive_sample_count: int
    negative_sample_count: int
    valid: bool
    invalid_reason: str | None = None


@dataclass(frozen=True)
class DenseTrackFeatureComparison:
    """Symmetric dense evidence for one aligned source/target mask pair."""

    source_explained_fraction: float | None
    target_purity_fraction: float | None
    source_effective_occupancy: float
    target_effective_occupancy: float
    source_reliable_fraction: float
    feature_match_cell_count: int
    valid: bool
    invalid_reason: str | None = None


def area_resize_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Area-resize a 2-D mask to a soft feature-grid occupancy map.

    Args:
        mask: Binary or numeric image-space mask. Positive values are treated
            as foreground so the feature measurement follows the rest of the
            pipeline's binary-mask semantics.
        output_shape: Requested ``(height, width)``.

    Returns:
        A float32 array in ``[0, 1]``. A value of 0.25 means that roughly one
        quarter of the corresponding feature cell is covered by the mask.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("mask dimensions must be nonzero")

    height, width = (int(output_shape[0]), int(output_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("output_shape dimensions must be positive")

    binary = (array > 0).astype(np.float32)
    if binary.shape == (height, width):
        return binary

    # Pillow's BOX filter averages all image pixels covered by an output cell.
    # The floating-point "F" image mode preserves fractional occupancies.
    image = Image.fromarray(binary, mode="F")
    resized = image.resize((width, height), resample=Image.Resampling.BOX)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def prepare_feature_map(
    feature_map: np.ndarray,
    *,
    norm_epsilon: float = 1e-12,
) -> PreparedFeatureMap:
    """Validate and L2-normalize a NumPy ``C x H x W`` feature map.

    Normalizing each spatial descriptor prevents high-magnitude encoder cells
    from dominating the object average. The pooled object descriptor is
    normalized a second time in :func:`masked_feature_descriptor`.
    """
    try:
        features = np.asarray(feature_map)
    except Exception as error:
        raise TypeError(
            "feature_map must be NumPy-convertible; convert CUDA tensors with "
            "tensor.detach().float().cpu().numpy() first"
        ) from error

    if features.ndim != 3:
        raise ValueError("feature_map must have shape (channels, height, width)")
    channels, height, width = features.shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("feature_map dimensions must be positive")
    if not np.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError("norm_epsilon must be finite and positive")

    # float64 accumulation makes comparisons stable across float16/float32
    # encoder output and avoids threshold flips caused solely by summation.
    features = features.astype(np.float64, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError("feature_map must contain only finite values")

    norms = np.linalg.norm(features, axis=0)
    valid_cells = norms > norm_epsilon
    unit_features = np.zeros_like(features, dtype=np.float64)
    unit_features[:, valid_cells] = (
        features[:, valid_cells] / norms[valid_cells][None, :]
    )
    return PreparedFeatureMap(unit_features=unit_features, valid_cells=valid_cells)


def masked_feature_descriptor(
    feature_map: np.ndarray | PreparedFeatureMap,
    mask: np.ndarray,
    *,
    minimum_effective_occupancy: float = 0.25,
    norm_epsilon: float = 1e-12,
) -> MaskedFeatureDescriptor:
    """Pool one object mask into a unit-length feature descriptor.

    Effective occupancy is the sum of valid soft mask weights in feature-cell
    units. For example, four cells with 0.25 occupancy provide one effective
    cell of evidence. Masks below ``minimum_effective_occupancy`` are marked
    invalid instead of producing unstable or arbitrary cosine values.
    """
    if (
        not np.isfinite(minimum_effective_occupancy)
        or minimum_effective_occupancy < 0
    ):
        raise ValueError("minimum_effective_occupancy must be finite and nonnegative")
    if not np.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError("norm_epsilon must be finite and positive")

    prepared = (
        feature_map
        if isinstance(feature_map, PreparedFeatureMap)
        else prepare_feature_map(feature_map, norm_epsilon=norm_epsilon)
    )
    occupancy = area_resize_mask(mask, prepared.grid_shape).astype(
        np.float64, copy=False
    )
    occupied_cells = int(np.count_nonzero(occupancy > 0))

    # Zero-norm feature cells cannot support a meaningful comparison even if
    # the image mask covers them, so exclude them from effective evidence.
    valid_occupancy = occupancy * prepared.valid_cells
    effective_occupancy = float(valid_occupancy.sum())
    if effective_occupancy < minimum_effective_occupancy:
        return MaskedFeatureDescriptor(
            vector=None,
            effective_occupancy=effective_occupancy,
            occupied_cells=occupied_cells,
            valid=False,
            invalid_reason="insufficient_effective_occupancy",
        )

    pooled = np.sum(
        prepared.unit_features * valid_occupancy[None, :, :],
        axis=(1, 2),
    )
    pooled_norm = float(np.linalg.norm(pooled))
    if not np.isfinite(pooled_norm) or pooled_norm <= norm_epsilon:
        return MaskedFeatureDescriptor(
            vector=None,
            effective_occupancy=effective_occupancy,
            occupied_cells=occupied_cells,
            valid=False,
            invalid_reason="degenerate_pooled_descriptor",
        )

    return MaskedFeatureDescriptor(
        vector=pooled / pooled_norm,
        effective_occupancy=effective_occupancy,
        occupied_cells=occupied_cells,
        valid=True,
    )


def compare_masked_features(
    source_feature_map: np.ndarray | PreparedFeatureMap,
    target_feature_map: np.ndarray | PreparedFeatureMap,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    minimum_effective_occupancy: float = 0.25,
    norm_epsilon: float = 1e-12,
) -> MaskedFeatureComparison:
    """Compare two masked feature maps with cosine similarity.

    The two feature grids may have different spatial sizes, but their channel
    dimensions must agree because the pooled descriptors share one embedding
    space.
    """
    source_features = (
        source_feature_map
        if isinstance(source_feature_map, PreparedFeatureMap)
        else prepare_feature_map(source_feature_map, norm_epsilon=norm_epsilon)
    )
    target_features = (
        target_feature_map
        if isinstance(target_feature_map, PreparedFeatureMap)
        else prepare_feature_map(target_feature_map, norm_epsilon=norm_epsilon)
    )
    if source_features.unit_features.shape[0] != target_features.unit_features.shape[0]:
        raise ValueError("source and target feature maps must have equal channels")

    source = masked_feature_descriptor(
        source_features,
        source_mask,
        minimum_effective_occupancy=minimum_effective_occupancy,
        norm_epsilon=norm_epsilon,
    )
    target = masked_feature_descriptor(
        target_features,
        target_mask,
        minimum_effective_occupancy=minimum_effective_occupancy,
        norm_epsilon=norm_epsilon,
    )

    invalid_parts = []
    if not source.valid:
        invalid_parts.append(f"source_{source.invalid_reason}")
    if not target.valid:
        invalid_parts.append(f"target_{target.invalid_reason}")
    if invalid_parts:
        return MaskedFeatureComparison(
            cosine_similarity=None,
            source_effective_occupancy=source.effective_occupancy,
            target_effective_occupancy=target.effective_occupancy,
            source_occupied_cells=source.occupied_cells,
            target_occupied_cells=target.occupied_cells,
            source_grid_shape=source_features.grid_shape,
            target_grid_shape=target_features.grid_shape,
            valid=False,
            invalid_reason=";".join(invalid_parts),
        )

    assert source.vector is not None and target.vector is not None
    # Both descriptors are unit length. Clip only to remove tiny floating-point
    # excursions outside the mathematical cosine range.
    cosine = float(np.clip(np.dot(source.vector, target.vector), -1.0, 1.0))
    return MaskedFeatureComparison(
        cosine_similarity=cosine,
        source_effective_occupancy=source.effective_occupancy,
        target_effective_occupancy=target.effective_occupancy,
        source_occupied_cells=source.occupied_cells,
        target_occupied_cells=target.occupied_cells,
        source_grid_shape=source_features.grid_shape,
        target_grid_shape=target_features.grid_shape,
        valid=True,
    )


def _annotate_track(
    track: ObjectMask,
    comparison: MaskedFeatureComparison,
    *,
    minimum_cosine: float,
    minimum_effective_occupancy: float,
) -> None:
    """Store JSON-safe feature evidence on a propagated mask."""
    track.metadata.update(
        {
            "sam2_feature_comparison_valid": comparison.valid,
            "sam2_feature_cosine_similarity": comparison.cosine_similarity,
            "sam2_feature_source_effective_occupancy": (
                comparison.source_effective_occupancy
            ),
            "sam2_feature_target_effective_occupancy": (
                comparison.target_effective_occupancy
            ),
            "sam2_feature_source_occupied_cells": comparison.source_occupied_cells,
            "sam2_feature_target_occupied_cells": comparison.target_occupied_cells,
            "sam2_feature_source_grid_shape": list(comparison.source_grid_shape),
            "sam2_feature_target_grid_shape": list(comparison.target_grid_shape),
            "sam2_feature_invalid_reason": comparison.invalid_reason,
            "sam2_feature_minimum_cosine": float(minimum_cosine),
            "sam2_feature_minimum_effective_occupancy": float(
                minimum_effective_occupancy
            ),
            "sam2_feature_descriptor_method": (
                "area_weighted_mean_of_unit_cell_features"
            ),
        }
    )


def reject_feature_inconsistent_tracks(
    source_masks: list[ObjectMask],
    tracks: list[ObjectMask | None],
    source_feature_map: np.ndarray,
    target_feature_map: np.ndarray,
    *,
    minimum_cosine: float,
    minimum_effective_occupancy: float = 0.25,
) -> list[ObjectMask | None]:
    """Reject tracks whose pooled image features disagree with their source.

    This is an inferred reproduction ablation, not a published GOLDILOCS
    decision rule. It is intended to sit *after* spatial IoU/area gating:
    geometry asks whether the masks occupy a plausible corresponding region,
    while this function asks whether their SAM2 encoder appearance agrees.

    Invalid comparisons (usually a tiny mask or zero-norm feature evidence)
    are rejected conservatively. Every attempted non-``None`` track is
    annotated before filtering so debug reports can distinguish missing
    evidence from a low cosine score.
    """
    if len(source_masks) != len(tracks):
        raise ValueError("source_masks and tracks must have equal length")
    if not np.isfinite(minimum_cosine) or not -1 <= minimum_cosine <= 1:
        raise ValueError("minimum_cosine must be finite and in [-1, 1]")

    # Feature normalization is independent of the object masks; prepare each
    # grid once instead of repeating this work for every SAM proposal.
    source_features = prepare_feature_map(source_feature_map)
    target_features = prepare_feature_map(target_feature_map)

    accepted: list[ObjectMask | None] = []
    for source, track in zip(source_masks, tracks):
        if track is None:
            accepted.append(None)
            continue

        comparison = compare_masked_features(
            source_features,
            target_features,
            source.mask,
            track.mask,
            minimum_effective_occupancy=minimum_effective_occupancy,
        )
        _annotate_track(
            track,
            comparison,
            minimum_cosine=minimum_cosine,
            minimum_effective_occupancy=minimum_effective_occupancy,
        )

        if not comparison.valid:
            reason = "insufficient_sam2_feature_occupancy"
        elif comparison.cosine_similarity is not None and (
            comparison.cosine_similarity < minimum_cosine
        ):
            reason = "insufficient_sam2_feature_cosine_similarity"
        else:
            accepted.append(track)
            continue

        reasons = track.metadata.setdefault("rejection_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        accepted.append(None)

    return accepted


def _validate_fraction(value: float, name: str) -> None:
    """Validate a finite closed-interval fraction."""
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _aligned_cosine_map(
    source_features: PreparedFeatureMap,
    target_features: PreparedFeatureMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-cell cosine and a mask of cells with valid descriptors."""
    if source_features.unit_features.shape != target_features.unit_features.shape:
        raise ValueError(
            "aligned source and target feature maps must have identical shapes"
        )
    cosine = np.sum(
        source_features.unit_features * target_features.unit_features,
        axis=0,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    valid = source_features.valid_cells & target_features.valid_cells
    return cosine, valid


def _offset_slices(
    size: int,
    offset: int,
) -> tuple[slice, slice]:
    """Create non-wrapping source/target slices for one spatial offset."""
    if offset >= 0:
        return slice(0, max(0, size - offset)), slice(min(offset, size), size)
    return slice(min(-offset, size), size), slice(0, max(0, size + offset))


def calibrate_pair_feature_threshold(
    source_feature_map: np.ndarray | PreparedFeatureMap,
    target_feature_map: np.ndarray | PreparedFeatureMap,
    reliable_coverage: np.ndarray,
    *,
    negative_offsets: Sequence[tuple[int, int]],
    maximum_negative_acceptance: float,
    minimum_positive_acceptance: float,
    minimum_positive_samples: int,
    minimum_negative_samples: int,
    reliable_cell_threshold: float,
) -> PairFeatureCalibration:
    """Calibrate a dense cosine threshold from pair-internal controls.

    Positive controls are aligned cells inside reliable render coverage.
    Negative controls pair the same two feature maps after explicit, large
    spatial offsets. Slicing is used rather than ``np.roll`` so image content
    can never wrap from one boundary to the opposite boundary.

    The threshold is the ``1 - maximum_negative_acceptance`` quantile of the
    negative controls. Calibration is considered trustworthy only when both
    control populations have enough samples and at least
    ``minimum_positive_acceptance`` of the aligned controls meet the inferred
    threshold. Ground truth is never consulted.
    """
    _validate_fraction(maximum_negative_acceptance, "maximum_negative_acceptance")
    _validate_fraction(minimum_positive_acceptance, "minimum_positive_acceptance")
    if minimum_positive_samples < 1 or minimum_negative_samples < 1:
        raise ValueError("minimum control sample counts must be positive")
    if not 0 < reliable_cell_threshold <= 1:
        raise ValueError("reliable_cell_threshold must be in (0, 1]")

    source = (
        source_feature_map
        if isinstance(source_feature_map, PreparedFeatureMap)
        else prepare_feature_map(source_feature_map)
    )
    target = (
        target_feature_map
        if isinstance(target_feature_map, PreparedFeatureMap)
        else prepare_feature_map(target_feature_map)
    )
    aligned_cosine, aligned_valid = _aligned_cosine_map(source, target)
    height, width = source.grid_shape

    reliable_occupancy = area_resize_mask(reliable_coverage, (height, width))
    reliable_cells = reliable_occupancy >= reliable_cell_threshold
    positive_mask = reliable_cells & aligned_valid
    positive_scores = aligned_cosine[positive_mask]

    negative_parts: list[np.ndarray] = []
    for raw_offset in negative_offsets:
        if len(raw_offset) != 2:
            raise ValueError("every negative offset must contain (dy, dx)")
        dy, dx = int(raw_offset[0]), int(raw_offset[1])
        if dy == 0 and dx == 0:
            raise ValueError("negative offsets must not contain (0, 0)")

        source_y, target_y = _offset_slices(height, dy)
        source_x, target_x = _offset_slices(width, dx)
        source_region = source.unit_features[:, source_y, source_x]
        target_region = target.unit_features[:, target_y, target_x]
        if source_region.shape[1] == 0 or source_region.shape[2] == 0:
            # An offset outside the feature grid contributes no controls. It
            # is counted indirectly by the minimum-negative-samples check.
            continue

        valid = (
            source.valid_cells[source_y, source_x]
            & target.valid_cells[target_y, target_x]
            & reliable_cells[source_y, source_x]
            & reliable_cells[target_y, target_x]
        )
        scores = np.sum(source_region * target_region, axis=0)
        if np.any(valid):
            negative_parts.append(np.clip(scores[valid], -1.0, 1.0))

    negative_scores = (
        np.concatenate(negative_parts)
        if negative_parts
        else np.empty(0, dtype=np.float64)
    )
    positive_count = int(positive_scores.size)
    negative_count = int(negative_scores.size)

    sample_reasons = []
    if positive_count < minimum_positive_samples:
        sample_reasons.append("insufficient_positive_control_samples")
    if negative_count < minimum_negative_samples:
        sample_reasons.append("insufficient_negative_control_samples")
    if sample_reasons:
        return PairFeatureCalibration(
            threshold=None,
            positive_acceptance=None,
            negative_acceptance=None,
            positive_sample_count=positive_count,
            negative_sample_count=negative_count,
            valid=False,
            invalid_reason=";".join(sample_reasons),
        )

    # ``higher`` chooses an observed control value and is deterministic across
    # NumPy versions. We record the realized negative acceptance as ties at the
    # quantile can make it larger than the requested nominal tail probability.
    threshold = float(
        np.quantile(
            negative_scores,
            1.0 - maximum_negative_acceptance,
            method="higher",
        )
    )
    positive_acceptance = float(np.mean(positive_scores >= threshold))
    negative_acceptance = float(np.mean(negative_scores >= threshold))
    if positive_acceptance < minimum_positive_acceptance:
        return PairFeatureCalibration(
            threshold=threshold,
            positive_acceptance=positive_acceptance,
            negative_acceptance=negative_acceptance,
            positive_sample_count=positive_count,
            negative_sample_count=negative_count,
            valid=False,
            invalid_reason="positive_control_acceptance_below_minimum",
        )

    return PairFeatureCalibration(
        threshold=threshold,
        positive_acceptance=positive_acceptance,
        negative_acceptance=negative_acceptance,
        positive_sample_count=positive_count,
        negative_sample_count=negative_count,
        valid=True,
    )


def _square_morphology(mask: np.ndarray, radius: int, *, operation: str) -> np.ndarray:
    """Dilate or erode a feature-grid mask with a square neighborhood."""
    if radius < 0:
        raise ValueError("morphology radii must be nonnegative")
    binary = np.asarray(mask, dtype=bool)
    if radius == 0:
        return binary.copy()

    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    if operation == "dilate":
        result = np.zeros_like(binary)
        combine = np.logical_or
    elif operation == "erode":
        result = np.ones_like(binary)
        combine = np.logical_and
    else:
        raise ValueError("operation must be 'dilate' or 'erode'")

    height, width = binary.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            window = padded[dy : dy + height, dx : dx + width]
            result = combine(result, window)
    return result


def compare_aligned_dense_features(
    source_feature_map: np.ndarray | PreparedFeatureMap,
    target_feature_map: np.ndarray | PreparedFeatureMap,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    reliable_coverage: np.ndarray,
    *,
    cosine_threshold: float,
    minimum_effective_occupancy: float,
    minimum_reliable_source_fraction: float,
    reliable_cell_threshold: float,
    mask_cell_threshold: float,
    dilation_radius: int,
    erosion_radius: int,
) -> DenseTrackFeatureComparison:
    """Measure whole-object completeness and purity on an aligned feature grid.

    ``source_explained_fraction`` asks what fraction of reliable source-object
    evidence is both spatially covered by the target track and feature-similar.
    ``target_purity_fraction`` asks the symmetric question from the target
    track. A residual fragment can have a strong pooled feature cosine but a
    low source-explained fraction, which is why both dense fractions matter.

    A small dilation tolerates feature-grid quantization at mask boundaries.
    Optional erosion removes uncertain boundary cells from each denominator.
    """
    if not np.isfinite(cosine_threshold) or not -1 <= cosine_threshold <= 1:
        raise ValueError("cosine_threshold must be finite and in [-1, 1]")
    if (
        not np.isfinite(minimum_effective_occupancy)
        or minimum_effective_occupancy < 0
    ):
        raise ValueError("minimum_effective_occupancy must be finite and nonnegative")
    _validate_fraction(
        minimum_reliable_source_fraction,
        "minimum_reliable_source_fraction",
    )
    if not 0 < reliable_cell_threshold <= 1:
        raise ValueError("reliable_cell_threshold must be in (0, 1]")
    if not 0 < mask_cell_threshold <= 1:
        raise ValueError("mask_cell_threshold must be in (0, 1]")
    if dilation_radius < 0 or erosion_radius < 0:
        raise ValueError("morphology radii must be nonnegative")

    source = (
        source_feature_map
        if isinstance(source_feature_map, PreparedFeatureMap)
        else prepare_feature_map(source_feature_map)
    )
    target = (
        target_feature_map
        if isinstance(target_feature_map, PreparedFeatureMap)
        else prepare_feature_map(target_feature_map)
    )
    cosine, feature_valid = _aligned_cosine_map(source, target)
    grid_shape = source.grid_shape

    source_occupancy = area_resize_mask(source_mask, grid_shape).astype(
        np.float64, copy=False
    )
    target_occupancy = area_resize_mask(target_mask, grid_shape).astype(
        np.float64, copy=False
    )
    reliable_occupancy = area_resize_mask(reliable_coverage, grid_shape).astype(
        np.float64, copy=False
    )

    source_support = source_occupancy >= mask_cell_threshold
    target_support = target_occupancy >= mask_cell_threshold
    source_core = _square_morphology(
        source_support, erosion_radius, operation="erode"
    )
    target_core = _square_morphology(
        target_support, erosion_radius, operation="erode"
    )
    source_neighborhood = _square_morphology(
        source_support, dilation_radius, operation="dilate"
    )
    target_neighborhood = _square_morphology(
        target_support, dilation_radius, operation="dilate"
    )

    reliable_weights = np.where(
        reliable_occupancy >= reliable_cell_threshold,
        reliable_occupancy,
        0.0,
    )
    reliable_cells = reliable_weights > 0
    valid_weights = feature_valid.astype(np.float64)
    common_weights = reliable_weights * valid_weights
    source_total_weights = source_occupancy * source_core * valid_weights
    source_weights = source_total_weights * reliable_weights
    target_weights = target_occupancy * target_core * common_weights
    source_effective = float(source_weights.sum())
    target_effective = float(target_weights.sum())
    source_total = float(source_total_weights.sum())
    source_reliable_fraction = (
        source_effective / source_total if source_total > 0 else 0.0
    )

    invalid_parts = []
    if source_effective < minimum_effective_occupancy:
        invalid_parts.append("source_insufficient_effective_occupancy")
    if target_effective < minimum_effective_occupancy:
        invalid_parts.append("target_insufficient_effective_occupancy")
    if source_reliable_fraction < minimum_reliable_source_fraction:
        invalid_parts.append("insufficient_reliable_source_fraction")
    if invalid_parts:
        return DenseTrackFeatureComparison(
            source_explained_fraction=None,
            target_purity_fraction=None,
            source_effective_occupancy=source_effective,
            target_effective_occupancy=target_effective,
            source_reliable_fraction=source_reliable_fraction,
            feature_match_cell_count=0,
            valid=False,
            invalid_reason=";".join(invalid_parts),
        )

    feature_matches = feature_valid & (cosine >= cosine_threshold)
    source_explained = float(
        np.sum(source_weights * target_neighborhood * feature_matches)
        / source_effective
    )
    target_purity = float(
        np.sum(target_weights * source_neighborhood * feature_matches)
        / target_effective
    )
    return DenseTrackFeatureComparison(
        source_explained_fraction=source_explained,
        target_purity_fraction=target_purity,
        source_effective_occupancy=source_effective,
        target_effective_occupancy=target_effective,
        source_reliable_fraction=source_reliable_fraction,
        feature_match_cell_count=int(
            np.count_nonzero(
                feature_matches
                & reliable_cells
                & (source_support | target_support)
            )
        ),
        valid=True,
    )


def _default_large_offsets(grid_shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Choose deterministic far offsets when config specifies pair controls.

    These offsets compare different image regions without wraparound. They are
    expressed relative to the feature-grid dimensions so the same rule works
    for different SAM2 FPN resolutions.
    """
    height, width = grid_shape
    candidates = (
        (0, width // 2),
        (height // 2, 0),
        (height // 3, width // 3),
        (height // 3, -(width // 3)),
    )
    # Very small synthetic grids can collapse integer fractions to zero.
    return tuple(dict.fromkeys(offset for offset in candidates if offset != (0, 0)))


def apply_aligned_sam2_feature_gate(
    source_masks: list[ObjectMask],
    tracks: list[ObjectMask | None],
    source_feature_map: np.ndarray,
    target_feature_map: np.ndarray,
    reliable_coverage: np.ndarray,
    *,
    maximum_negative_acceptance: float,
    minimum_positive_acceptance: float,
    minimum_calibration_cells: int,
    minimum_reliable_cells: int,
    boundary_ignore_cells: int,
    correspondence_dilation_cells: int,
    minimum_reliable_source_fraction: float,
    minimum_explained_source_fraction: float,
    minimum_target_purity_fraction: float,
    low_evidence_policy: str,
    negative_offsets: Sequence[tuple[int, int]] | None = None,
    reliable_cell_threshold: float | None = None,
    mask_cell_threshold: float | None = None,
) -> list[ObjectMask | None]:
    """Apply a tri-state dense SAM2 feature gate to aligned clean-render tracks.

    This gate is an **unpublished reproduction experiment**, not a rule stated
    in GOLDILOCS. It should only be used for aligned source-to-clean and
    target-to-clean tracks; moved-object search intentionally changes spatial
    position and is incompatible with the aligned comparison.

    Decisions:

    - ``accept``: calibrated evidence is adequate and both completeness tests
      pass;
    - ``reject``: calibrated evidence is adequate but source completeness or
      target purity is too low;
    - ``ambiguous``: calibration or per-track evidence is inadequate.

    Ambiguous tracks are retained. Treating missing evidence as a tracking
    failure would manufacture change predictions over render holes or tiny
    feature-grid masks.
    """
    if len(source_masks) != len(tracks):
        raise ValueError("source_masks and tracks must have equal length")
    _validate_fraction(
        minimum_explained_source_fraction,
        "minimum_explained_source_fraction",
    )
    _validate_fraction(minimum_target_purity_fraction, "minimum_target_purity_fraction")
    _validate_fraction(
        minimum_reliable_source_fraction,
        "minimum_reliable_source_fraction",
    )
    if minimum_calibration_cells < 1 or minimum_reliable_cells < 1:
        raise ValueError("minimum cell counts must be positive")
    if boundary_ignore_cells < 0 or correspondence_dilation_cells < 0:
        raise ValueError("feature-grid morphology sizes must be nonnegative")
    if low_evidence_policy != "keep_ambiguous":
        raise ValueError(
            "only low_evidence_policy='keep_ambiguous' is supported because "
            "missing feature evidence must not manufacture a change"
        )
    if reliable_cell_threshold is None:
        reliable_cell_threshold = float(np.finfo(np.float32).eps)
    if mask_cell_threshold is None:
        mask_cell_threshold = float(np.finfo(np.float32).eps)
    if not 0 < reliable_cell_threshold <= 1:
        raise ValueError("reliable_cell_threshold must be in (0, 1]")
    if not 0 < mask_cell_threshold <= 1:
        raise ValueError("mask_cell_threshold must be in (0, 1]")

    source_features = prepare_feature_map(source_feature_map)
    target_features = prepare_feature_map(target_feature_map)
    if negative_offsets is None:
        negative_offsets = _default_large_offsets(source_features.grid_shape)
    else:
        negative_offsets = tuple(
            (int(offset[0]), int(offset[1])) for offset in negative_offsets
        )
    calibration = calibrate_pair_feature_threshold(
        source_features,
        target_features,
        reliable_coverage,
        negative_offsets=negative_offsets,
        maximum_negative_acceptance=maximum_negative_acceptance,
        minimum_positive_acceptance=minimum_positive_acceptance,
        minimum_positive_samples=minimum_calibration_cells,
        minimum_negative_samples=minimum_calibration_cells,
        reliable_cell_threshold=reliable_cell_threshold,
    )

    calibration_metadata = {
        "sam2_dense_feature_threshold": calibration.threshold,
        "sam2_dense_feature_calibration_valid": calibration.valid,
        "sam2_dense_feature_calibration_invalid_reason": calibration.invalid_reason,
        "sam2_dense_feature_positive_acceptance": calibration.positive_acceptance,
        "sam2_dense_feature_negative_acceptance": calibration.negative_acceptance,
        "sam2_dense_feature_positive_sample_count": (
            calibration.positive_sample_count
        ),
        "sam2_dense_feature_negative_sample_count": (
            calibration.negative_sample_count
        ),
        "sam2_dense_feature_minimum_source_explained_fraction": float(
            minimum_explained_source_fraction
        ),
        "sam2_dense_feature_minimum_target_purity_fraction": float(
            minimum_target_purity_fraction
        ),
        "sam2_dense_feature_gate_provenance": (
            "unpublished_reproduction_experiment"
        ),
        "sam2_dense_feature_negative_offsets": [
            [int(dy), int(dx)] for dy, dx in negative_offsets
        ],
        "sam2_dense_feature_reliable_cell_threshold": float(
            reliable_cell_threshold
        ),
        "sam2_dense_feature_mask_cell_threshold": float(mask_cell_threshold),
    }

    output: list[ObjectMask | None] = []
    for source_mask, track in zip(source_masks, tracks):
        if track is None:
            output.append(None)
            continue
        track.metadata.update(calibration_metadata)

        if not calibration.valid or calibration.threshold is None:
            track.metadata["sam2_dense_feature_gate_decision"] = "ambiguous"
            ambiguity = track.metadata.setdefault("ambiguity_reasons", [])
            reason = f"sam2_feature_calibration:{calibration.invalid_reason}"
            if reason not in ambiguity:
                ambiguity.append(reason)
            output.append(track)
            continue

        dense = compare_aligned_dense_features(
            source_features,
            target_features,
            source_mask.mask,
            track.mask,
            reliable_coverage,
            cosine_threshold=calibration.threshold,
            minimum_effective_occupancy=float(minimum_reliable_cells),
            minimum_reliable_source_fraction=minimum_reliable_source_fraction,
            reliable_cell_threshold=reliable_cell_threshold,
            mask_cell_threshold=mask_cell_threshold,
            dilation_radius=correspondence_dilation_cells,
            erosion_radius=boundary_ignore_cells,
        )
        prototype = compare_masked_features(
            source_features,
            target_features,
            source_mask.mask,
            track.mask,
            minimum_effective_occupancy=float(minimum_reliable_cells),
        )
        track.metadata.update(
            {
                "sam2_dense_feature_source_explained_fraction": (
                    dense.source_explained_fraction
                ),
                "sam2_dense_feature_target_purity_fraction": (
                    dense.target_purity_fraction
                ),
                "sam2_dense_feature_source_effective_occupancy": (
                    dense.source_effective_occupancy
                ),
                "sam2_dense_feature_target_effective_occupancy": (
                    dense.target_effective_occupancy
                ),
                "sam2_dense_feature_source_reliable_fraction": (
                    dense.source_reliable_fraction
                ),
                "sam2_dense_feature_match_cell_count": (
                    dense.feature_match_cell_count
                ),
                "sam2_dense_feature_comparison_valid": dense.valid,
                "sam2_dense_feature_comparison_invalid_reason": (
                    dense.invalid_reason
                ),
                "sam2_feature_prototype_cosine_similarity": (
                    prototype.cosine_similarity
                ),
            }
        )

        if not dense.valid:
            track.metadata["sam2_dense_feature_gate_decision"] = "ambiguous"
            ambiguity = track.metadata.setdefault("ambiguity_reasons", [])
            reason = f"sam2_dense_feature_evidence:{dense.invalid_reason}"
            if reason not in ambiguity:
                ambiguity.append(reason)
            output.append(track)
            continue

        rejection_reasons = []
        assert dense.source_explained_fraction is not None
        assert dense.target_purity_fraction is not None
        if dense.source_explained_fraction < minimum_explained_source_fraction:
            rejection_reasons.append(
                "insufficient_sam2_source_explained_fraction"
            )
        if dense.target_purity_fraction < minimum_target_purity_fraction:
            rejection_reasons.append("insufficient_sam2_target_purity_fraction")

        if rejection_reasons:
            track.metadata["sam2_dense_feature_gate_decision"] = "reject"
            existing_reasons = track.metadata.setdefault("rejection_reasons", [])
            for reason in rejection_reasons:
                if reason not in existing_reasons:
                    existing_reasons.append(reason)
            output.append(None)
        else:
            track.metadata["sam2_dense_feature_gate_decision"] = "accept"
            output.append(track)

    return output


def apply_conditional_sam2_feature_gate(
    source_masks: list[ObjectMask],
    tracks: list[ObjectMask | None],
    source_feature_map: np.ndarray,
    target_feature_map: np.ndarray,
    reliable_coverage: np.ndarray,
    *,
    safe_iou: float,
    safe_area_ratio_bounds: tuple[float, float],
    maximum_negative_acceptance: float,
    minimum_positive_acceptance: float,
    minimum_calibration_cells: int,
    minimum_reliable_cells: int,
    minimum_reliable_source_fraction: float,
    minimum_reliable_target_fraction: float,
    low_evidence_policy: str,
    negative_offsets: Sequence[tuple[int, int]],
    reliable_cell_threshold: float,
) -> list[ObjectMask | None]:
    """Use SAM2 appearance only for area/IoU-ambiguous aligned tracks.

    This is a separate **unpublished experiment** built on top of the area
    gate. Tracks with strong spatial agreement bypass feature comparison.
    Borderline tracks are rejected only when:

    1. pair-level feature calibration is trustworthy;
    2. both masks have sufficient reliable clean-render support; and
    3. their pooled SAM2 prototype cosine is below the calibrated threshold.

    High similarity merely preserves the area-gate result. It is not treated
    as proof that the whole object survived, because a rasterized residual can
    retain the same semantic appearance. Ambiguous evidence is also retained
    to avoid manufacturing changes over holes or small masks.
    """
    if len(source_masks) != len(tracks):
        raise ValueError("source_masks and tracks must have equal length")
    _validate_fraction(safe_iou, "safe_iou")
    _validate_fraction(
        minimum_reliable_source_fraction,
        "minimum_reliable_source_fraction",
    )
    _validate_fraction(
        minimum_reliable_target_fraction,
        "minimum_reliable_target_fraction",
    )
    if low_evidence_policy != "keep_ambiguous":
        raise ValueError(
            "conditional feature gating supports only "
            "low_evidence_policy='keep_ambiguous'"
        )
    if minimum_calibration_cells < 1 or minimum_reliable_cells < 1:
        raise ValueError("minimum cell counts must be positive")
    if not 0 < reliable_cell_threshold <= 1:
        raise ValueError("reliable_cell_threshold must be in (0, 1]")
    low_area, high_area = (
        float(safe_area_ratio_bounds[0]),
        float(safe_area_ratio_bounds[1]),
    )
    if not 0 < low_area <= high_area:
        raise ValueError("safe_area_ratio_bounds must be positive and ordered")

    source_features = prepare_feature_map(source_feature_map)
    target_features = prepare_feature_map(target_feature_map)
    if source_features.grid_shape != target_features.grid_shape:
        raise ValueError("aligned feature maps must have equal spatial shapes")

    # Compute routing before calibration. If every accepted area-gate track is
    # already spatially safe, no feature-derived decision is needed.
    routed: list[tuple[float, float, bool] | None] = []
    any_uncertain = False
    for source_mask, track in zip(source_masks, tracks):
        if track is None:
            routed.append(None)
            continue
        source_binary = np.asarray(source_mask.mask, dtype=bool)
        target_binary = np.asarray(track.mask, dtype=bool)
        intersection = int(np.logical_and(source_binary, target_binary).sum())
        union = int(np.logical_or(source_binary, target_binary).sum())
        source_area = int(source_binary.sum())
        target_area = int(target_binary.sum())
        iou = float(intersection / union) if union else 0.0
        area_ratio = (
            float(target_area / source_area) if source_area else float("inf")
        )
        is_safe = (
            iou >= safe_iou and low_area <= area_ratio <= high_area
        )
        routed.append((iou, area_ratio, is_safe))
        any_uncertain |= not is_safe

    if not any_uncertain:
        for route, track in zip(routed, tracks):
            if track is not None and route is not None:
                track.metadata.update(
                    {
                        "sam2_conditional_feature_gate_decision": (
                            "not_checked_spatially_safe"
                        ),
                        "sam2_conditional_feature_iou": route[0],
                        "sam2_conditional_feature_area_ratio": route[1],
                    }
                )
        return tracks

    calibration = calibrate_pair_feature_threshold(
        source_features,
        target_features,
        reliable_coverage,
        negative_offsets=negative_offsets,
        maximum_negative_acceptance=maximum_negative_acceptance,
        minimum_positive_acceptance=minimum_positive_acceptance,
        minimum_positive_samples=minimum_calibration_cells,
        minimum_negative_samples=minimum_calibration_cells,
        reliable_cell_threshold=reliable_cell_threshold,
    )
    calibration_metadata = {
        "sam2_conditional_feature_threshold": calibration.threshold,
        "sam2_conditional_feature_calibration_valid": calibration.valid,
        "sam2_conditional_feature_calibration_invalid_reason": (
            calibration.invalid_reason
        ),
        "sam2_conditional_feature_positive_acceptance": (
            calibration.positive_acceptance
        ),
        "sam2_conditional_feature_negative_acceptance": (
            calibration.negative_acceptance
        ),
        "sam2_conditional_feature_positive_sample_count": (
            calibration.positive_sample_count
        ),
        "sam2_conditional_feature_negative_sample_count": (
            calibration.negative_sample_count
        ),
        "sam2_conditional_feature_gate_provenance": (
            "unpublished_conditional_prototype_experiment"
        ),
    }

    reliable = np.asarray(reliable_coverage, dtype=bool)
    grid_shape = source_features.grid_shape
    reliable_grid = area_resize_mask(reliable, grid_shape)
    reliable_grid = np.where(
        reliable_grid >= reliable_cell_threshold,
        reliable_grid,
        0.0,
    )

    output: list[ObjectMask | None] = []
    for source_mask, track, route in zip(source_masks, tracks, routed):
        if track is None or route is None:
            output.append(None)
            continue
        iou, area_ratio, is_safe = route
        track.metadata.update(
            {
                "sam2_conditional_feature_iou": iou,
                "sam2_conditional_feature_area_ratio": area_ratio,
            }
        )
        if is_safe:
            track.metadata["sam2_conditional_feature_gate_decision"] = (
                "not_checked_spatially_safe"
            )
            output.append(track)
            continue

        track.metadata.update(calibration_metadata)
        if not calibration.valid or calibration.threshold is None:
            track.metadata["sam2_conditional_feature_gate_decision"] = "ambiguous"
            track.metadata.setdefault("ambiguity_reasons", []).append(
                f"sam2_conditional_calibration:{calibration.invalid_reason}"
            )
            output.append(track)
            continue

        source_occupancy = area_resize_mask(source_mask.mask, grid_shape)
        target_occupancy = area_resize_mask(track.mask, grid_shape)
        source_total = float(source_occupancy.sum())
        target_total = float(target_occupancy.sum())
        source_reliable = (
            float((source_occupancy * reliable_grid).sum() / source_total)
            if source_total
            else 0.0
        )
        target_reliable = (
            float((target_occupancy * reliable_grid).sum() / target_total)
            if target_total
            else 0.0
        )
        comparison = compare_masked_features(
            source_features,
            target_features,
            np.logical_and(source_mask.mask, reliable),
            np.logical_and(track.mask, reliable),
            minimum_effective_occupancy=float(minimum_reliable_cells),
        )
        track.metadata.update(
            {
                "sam2_conditional_feature_prototype_cosine": (
                    comparison.cosine_similarity
                ),
                "sam2_conditional_feature_comparison_valid": comparison.valid,
                "sam2_conditional_feature_comparison_invalid_reason": (
                    comparison.invalid_reason
                ),
                "sam2_conditional_feature_source_reliable_fraction": (
                    source_reliable
                ),
                "sam2_conditional_feature_target_reliable_fraction": (
                    target_reliable
                ),
            }
        )
        evidence_reasons = []
        if not comparison.valid:
            evidence_reasons.append(
                f"feature_comparison:{comparison.invalid_reason}"
            )
        if source_reliable < minimum_reliable_source_fraction:
            evidence_reasons.append("insufficient_reliable_source_fraction")
        if target_reliable < minimum_reliable_target_fraction:
            evidence_reasons.append("insufficient_reliable_target_fraction")
        if evidence_reasons:
            track.metadata["sam2_conditional_feature_gate_decision"] = "ambiguous"
            track.metadata.setdefault("ambiguity_reasons", []).extend(
                f"sam2_conditional_evidence:{reason}"
                for reason in evidence_reasons
            )
            output.append(track)
            continue

        assert comparison.cosine_similarity is not None
        if comparison.cosine_similarity < calibration.threshold:
            track.metadata["sam2_conditional_feature_gate_decision"] = (
                "reject_low_similarity"
            )
            track.metadata.setdefault("rejection_reasons", []).append(
                "insufficient_sam2_conditional_prototype_cosine"
            )
            output.append(None)
        else:
            track.metadata["sam2_conditional_feature_gate_decision"] = (
                "accept_feature_similar"
            )
            output.append(track)

    return output


def apply_pooled_cosine_margin_gate(
    source_masks: list[ObjectMask],
    tracks: list[ObjectMask | None],
    source_feature_map: np.ndarray,
    target_feature_map: np.ndarray,
    reliable_coverage: np.ndarray,
    *,
    margin: float,
    minimum_effective_occupancy: float,
    negative_offsets: Sequence[tuple[int, int]],
    maximum_negative_acceptance: float,
    minimum_positive_acceptance: float,
    minimum_calibration_cells: int,
    reliable_cell_threshold: float,
) -> list[dict]:
    """Three-band pooled-cosine veto proposal for an accepted aligned track.

    This is a research ablation, not a published GOLDILOCS rule. Unlike
    :func:`apply_aligned_sam2_feature_gate` and
    :func:`apply_conditional_sam2_feature_gate`, it never deletes a track
    itself -- every track keeps its geometric decision, annotated with one of
    three states, so the caller can route ``confident_different`` tracks
    through independent verification instead of an unconditional hard reject.
    Reuses the single pooled mask-level descriptor (:func:`compare_masked_features`)
    rather than the dense per-cell completeness/purity measurement, since the
    pooled cosine was found to disagree far less with itself on borderline
    cases than the dense fractions do.

    Bands, relative to the pair-internal calibrated threshold ``T``:
    ``cosine >= T`` -> ``confident_same``; ``cosine <= T - margin`` ->
    ``confident_different``; otherwise -> ``uncertain``. Calibration failure
    or insufficient mask evidence also yields ``uncertain`` -- absence of
    evidence is never treated as evidence of a mismatch.
    """
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if len(source_masks) != len(tracks):
        raise ValueError("source_masks and tracks must have equal length")

    source_features = prepare_feature_map(source_feature_map)
    target_features = prepare_feature_map(target_feature_map)
    calibration = calibrate_pair_feature_threshold(
        source_features,
        target_features,
        reliable_coverage,
        negative_offsets=negative_offsets,
        maximum_negative_acceptance=maximum_negative_acceptance,
        minimum_positive_acceptance=minimum_positive_acceptance,
        minimum_positive_samples=minimum_calibration_cells,
        minimum_negative_samples=minimum_calibration_cells,
        reliable_cell_threshold=reliable_cell_threshold,
    )

    decisions: list[dict] = []
    for source, track in zip(source_masks, tracks):
        if track is None:
            decisions.append({"state": "not_applicable", "track": None, "cosine": None})
            continue
        if not calibration.valid:
            decisions.append(
                {
                    "state": "uncertain",
                    "track": track,
                    "cosine": None,
                    "threshold": None,
                    "reason": f"invalid_calibration:{calibration.invalid_reason}",
                }
            )
            continue

        comparison = compare_masked_features(
            source_features,
            target_features,
            source.mask,
            track.mask,
            minimum_effective_occupancy=minimum_effective_occupancy,
        )
        if not comparison.valid:
            decisions.append(
                {
                    "state": "uncertain",
                    "track": track,
                    "cosine": None,
                    "threshold": calibration.threshold,
                    "reason": comparison.invalid_reason,
                }
            )
            continue

        cosine = comparison.cosine_similarity
        assert cosine is not None
        threshold = calibration.threshold
        assert threshold is not None
        if cosine >= threshold:
            state = "confident_same"
        elif cosine <= threshold - margin:
            state = "confident_different"
        else:
            state = "uncertain"
        decisions.append(
            {
                "state": state,
                "track": track,
                "cosine": cosine,
                "threshold": threshold,
                "source_effective_occupancy": comparison.source_effective_occupancy,
                "target_effective_occupancy": comparison.target_effective_occupancy,
            }
        )

    return decisions
