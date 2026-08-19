"""Feature-veto experiment for objects accepted by the clean geometry gate.

The production gate remains untouched.  This standalone module pairs visible
R0,1 and I1 proposals by aligned location, then asks whether their frozen SAM3
descriptors agree.  Feature evidence is veto-only: it may promote an object
which geometry called unchanged, but it can never turn an existing geometry
rejection back into unchanged.

Three feature bands keep the inference rule explicit:

* cosine >= ``T_same``: similar, preserve the gate;
* cosine <= ``T_same - margin``: confidently different, promote;
* the interval between them: uncertain, abstain.

An optional fourth component compares object color, independent of the SAM3
embedding, and can additionally promote a pair straight to "different":
embedding similarity is invariant to properties a same-shaped,
differently-colored replacement changes, so a pair whose color clearly
disagrees is change evidence the cosine test alone cannot see. The
comparison is done in CIE L*a*b* chrominance (a*/b*, lightness dropped) --
closer to how a person judges "is this the same color" than raw RGB, and
much less sensitive to the exposure/shading gap between a rendered source
frame and a real target photo, which a raw-RGB histogram was found to
confuse with an actual color change. Color evidence is one-directional,
same as the embedding veto itself: it can only promote toward "different",
never rescind an embedding-driven "different" call. It is opt-in -- omit
``source_image``/``target_image`` to get the original three-band behavior
unchanged.

Invalid descriptors, missing counterparts, and non-reciprocal spatial pairs
also abstain.  No function in this module reads ChangeSim ground truth.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..masks import compose_labels, filter_visible, mark_replacements
from ..types import Label, ObjectMask
from .sam3_identity_location import (
    FeatureDescriptorBatch,
    cosine_similarity_matrix,
    pairwise_mask_iou,
)


@dataclass(frozen=True)
class FeatureVetoPair:
    """One deterministic reciprocal same-place proposal comparison."""

    source_index: int
    target_index: int
    source_proposal_id: int
    target_proposal_id: int
    spatial_iou: float
    centroid_distance: float
    area_ratio: float
    cosine: float
    same_threshold: float
    different_threshold: float
    feature_band: str
    source_gate_accepted: bool
    target_gate_accepted: bool
    color_distance: float | None = None
    color_different_threshold: float | None = None
    band_source: str = "embedding"
    color_components: tuple[float, float, float] | None = None
    color_evidence_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        # Preserve the exact historical cache fingerprint when the new
        # illumination evidence is not enabled.
        if self.color_components is None:
            record.pop("color_components")
        if self.color_evidence_status is None:
            record.pop("color_evidence_status")
        return record


@dataclass(frozen=True)
class FeatureVetoResult:
    """Paired comparisons, promotion sets, and pre-GT funnel accounting."""

    pairs: tuple[FeatureVetoPair, ...]
    hard_source_ids: tuple[int, ...]
    hard_target_ids: tuple[int, ...]
    guarded_source_ids: tuple[int, ...]
    guarded_target_ids: tuple[int, ...]
    funnel: dict[str, Any]


@dataclass(frozen=True)
class IlluminationColorEvidence:
    """GT-free color disagreement after removing the image-wide illuminant.

    ``components`` are target-minus-source residuals for log intensity,
    log(R/G), and log(B/G).  A valid distance is their standardized Euclidean
    norm.  Invalid evidence deliberately has no distance so callers abstain.
    """

    distance: float | None
    components: tuple[float, float, float] | None
    status: str
    source_mask_pixels: int
    target_mask_pixels: int
    source_context_pixels: int
    target_context_pixels: int
    source_clipped_fraction: float | None
    target_clipped_fraction: float | None


def _proposal_id(obj: ObjectMask, fallback: int) -> int:
    return int(obj.metadata.get("automatic_proposal_id", fallback))


def _centroids(objects: Sequence[ObjectMask]) -> np.ndarray:
    values = []
    for obj in objects:
        mask = np.asarray(obj.mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        height, width = mask.shape
        if not len(xs):
            values.append((math.nan, math.nan))
        else:
            values.append(
                (
                    float(xs.mean() / max(width - 1, 1)),
                    float(ys.mean() / max(height - 1, 1)),
                )
            )
    return np.asarray(values, np.float32).reshape(-1, 2)


def color_histogram(image: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray | None:
    """L1-normalized RGB color histogram over one object's masked pixels.

    Returns ``None`` when the mask covers no pixels, so callers can treat a
    degenerate footprint as "no color evidence" rather than a false signal.
    """
    pixels = np.asarray(image)[np.asarray(mask, dtype=bool)]
    if pixels.shape[0] == 0:
        return None
    pixels = pixels[:, :3].astype(np.float64) / 255.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    histogram, _ = np.histogramdd(pixels, bins=(edges, edges, edges))
    total = histogram.sum()
    if total <= 0:
        return None
    return (histogram / total).reshape(-1)


def histogram_intersection(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Standard color-distribution similarity in [0, 1]; 1 = identical, 0 = disjoint."""
    if a is None or b is None:
        return None
    return float(np.minimum(a, b).sum())


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    normalized = channel / 255.0
    return np.where(
        normalized <= 0.04045, normalized / 12.92, ((normalized + 0.055) / 1.055) ** 2.4
    )


def _log_color_features(image: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Return log geometric intensity and two log-chromaticities."""

    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Expected HWC RGB image")
    linear = _srgb_to_linear(pixels[..., :3].astype(np.float64))
    red, green, blue = linear[..., 0], linear[..., 1], linear[..., 2]
    log_red = np.log(red + epsilon)
    log_green = np.log(green + epsilon)
    log_blue = np.log(blue + epsilon)
    return np.stack(
        [
            (log_red + log_green + log_blue) / 3.0,
            log_red - log_green,
            log_blue - log_green,
        ],
        axis=-1,
    )


def illumination_relative_color_distance(
    source_image: np.ndarray,
    source_mask: np.ndarray,
    target_image: np.ndarray,
    target_mask: np.ndarray,
    *,
    coverage: np.ndarray | None = None,
    mode: str = "global_illumination",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scales: tuple[float, float, float] = (0.15, 0.20, 0.20),
    minimum_mask_pixels: int = 64,
    minimum_context_pixels: int = 4096,
    maximum_clipped_fraction: float = 0.25,
) -> IlluminationColorEvidence:
    """Compare two masks after accounting for whole-image illumination.

    The representation is ``[mean(log RGB), log(R/G), log(B/G)]``. Subtracting the
    corresponding full-image median cancels a global exposure multiplier and
    per-channel white-balance gains under the usual diagonal illuminant model.
    ``object_only`` skips that subtraction, ``global_exposure`` subtracts only
    log-intensity term, and ``global_illumination`` subtracts all components.

    This intentionally uses robust summaries rather than histograms.  It is
    also conservative: insufficient support or heavy black/white clipping
    produces invalid evidence rather than a change decision.
    """

    if mode not in {"object_only", "global_exposure", "global_illumination"}:
        raise ValueError(
            "mode must be one of 'object_only', 'global_exposure', "
            "or 'global_illumination'"
        )
    source = np.asarray(source_image)
    target = np.asarray(target_image)
    if source.shape != target.shape or source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("source and target RGB images must have equal HWC shapes")
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if source_mask.shape != source.shape[:2] or target_mask.shape != source.shape[:2]:
        raise ValueError("color masks must match the RGB image shape")
    if coverage is None:
        coverage_mask = np.ones(source.shape[:2], dtype=bool)
    else:
        coverage_mask = np.asarray(coverage, dtype=bool)
        if coverage_mask.shape != source.shape[:2]:
            raise ValueError("coverage must match the RGB image shape")
    scale_array = np.asarray(scales, dtype=np.float64)
    center_array = np.asarray(center, dtype=np.float64)
    if scale_array.shape != (3,) or not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError("color scales must contain three positive finite values")
    if center_array.shape != (3,) or not np.all(np.isfinite(center_array)):
        raise ValueError("color center must contain three finite values")
    if minimum_mask_pixels < 1 or minimum_context_pixels < 1:
        raise ValueError("minimum color support must be positive")
    if not 0.0 <= maximum_clipped_fraction <= 1.0:
        raise ValueError("maximum_clipped_fraction must be in [0, 1]")

    source_count = int(source_mask.sum())
    target_count = int(target_mask.sum())
    context_count = int(coverage_mask.sum())
    if source_count < minimum_mask_pixels or target_count < minimum_mask_pixels:
        return IlluminationColorEvidence(
            None,
            None,
            "insufficient_mask_pixels",
            source_count,
            target_count,
            context_count,
            context_count,
            None,
            None,
        )
    if context_count < minimum_context_pixels:
        return IlluminationColorEvidence(
            None,
            None,
            "insufficient_context_pixels",
            source_count,
            target_count,
            context_count,
            context_count,
            None,
            None,
        )

    source_linear = _srgb_to_linear(source[..., :3].astype(np.float64))
    target_linear = _srgb_to_linear(target[..., :3].astype(np.float64))
    source_y = np.sum(
        source_linear * np.asarray([0.2126729, 0.7151522, 0.0721750]), axis=-1
    )
    target_y = np.sum(
        target_linear * np.asarray([0.2126729, 0.7151522, 0.0721750]), axis=-1
    )
    source_clipped = (source_y <= 1e-4) | (source_y >= 0.995)
    target_clipped = (target_y <= 1e-4) | (target_y >= 0.995)
    source_clipped_fraction = float(source_clipped[source_mask].mean())
    target_clipped_fraction = float(target_clipped[target_mask].mean())
    if max(source_clipped_fraction, target_clipped_fraction) > maximum_clipped_fraction:
        return IlluminationColorEvidence(
            None,
            None,
            "excessive_mask_clipping",
            source_count,
            target_count,
            context_count,
            context_count,
            source_clipped_fraction,
            target_clipped_fraction,
        )

    source_features = _log_color_features(source)
    target_features = _log_color_features(target)
    source_object = np.median(source_features[source_mask], axis=0)
    target_object = np.median(target_features[target_mask], axis=0)
    source_context = np.median(source_features[coverage_mask], axis=0)
    target_context = np.median(target_features[coverage_mask], axis=0)
    source_signature = source_object.copy()
    target_signature = target_object.copy()
    if mode == "global_exposure":
        source_signature[0] -= source_context[0]
        target_signature[0] -= target_context[0]
    elif mode == "global_illumination":
        source_signature -= source_context
        target_signature -= target_context
    components = target_signature - source_signature
    distance = float(np.linalg.norm((components - center_array) / scale_array))
    return IlluminationColorEvidence(
        distance,
        tuple(map(float, components)),
        "valid",
        source_count,
        target_count,
        context_count,
        context_count,
        source_clipped_fraction,
        target_clipped_fraction,
    )


def rgb_to_lab(pixels: np.ndarray) -> np.ndarray:
    """Convert (..., 3) sRGB uint8/float pixels to CIE L*a*b* (D65 white point)."""
    linear = _srgb_to_linear(np.asarray(pixels, dtype=np.float64)[..., :3])
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = x / xn, y / yn, z / zn
    delta = 6.0 / 29.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    lightness = 116.0 * fy - 16.0
    a_axis = 500.0 * (fx - fy)
    b_axis = 200.0 * (fy - fz)
    return np.stack([lightness, a_axis, b_axis], axis=-1)


def mean_lab_chroma(image: np.ndarray, mask: np.ndarray, min_pixels: int = 4) -> tuple[float, float] | None:
    """Median (a*, b*) chrominance of one object's masked pixels.

    Deliberately drops L* (lightness): a synthesized render and a real photo
    of the same unchanged object can differ substantially in exposure and
    shading without the object's actual color having changed, and L* is
    where that difference shows up. a*/b* is far more stable under a global
    lighting shift while still capturing what a person would call "the
    object changed color." Median rather than mean for the same robustness
    reason color_histogram doesn't need: a handful of shadow/highlight
    pixels shouldn't swing the whole region's color.
    """
    pixels = np.asarray(image)[np.asarray(mask, dtype=bool)]
    if pixels.shape[0] < min_pixels:
        return None
    lab = rgb_to_lab(pixels)
    return float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))


def image_median_luminance(image: np.ndarray, sample_rate: int = 10) -> float:
    """Robust per-image illumination estimate: median L* from a subsample.

    Sampling reduces cost for large images while remaining stable.
    """
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Expected HWC RGB image")
    # subsample by taking every `sample_rate`-th pixel in flattened order
    flat = pixels.reshape(-1, 3)
    if sample_rate > 1:
        flat = flat[::sample_rate]
    lab = rgb_to_lab(flat.astype(np.uint8))
    return float(np.median(lab[:, 0]))


def masked_ab_hist(
    image: np.ndarray,
    mask: np.ndarray,
    bins: tuple[int, int] = (16, 16),
    normalize_luminance: bool = False,
    l_ref: float = 50.0,
) -> np.ndarray | None:
    """Compute an L1-normalized 2D histogram over Lab a*/b* for masked pixels.

    If `normalize_luminance` is True, scale the image's linear RGB so the
    median L* becomes `l_ref` before converting to Lab. Returns None when
    the mask covers no pixels.
    """
    pixels = np.asarray(image)
    mask_bool = np.asarray(mask, dtype=bool)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Expected HWC RGB image")
    if mask_bool.sum() == 0:
        return None
    # Optionally scale linear RGB to normalize illumination
    if normalize_luminance:
        L_med = image_median_luminance(pixels)
        scale = float(l_ref) / (L_med + 1e-6)
        # apply scale in sRGB-linear space: scale uint8 values then clip
        scaled = np.clip(pixels.astype(np.float64) * scale, 0.0, 255.0).astype(np.uint8)
        lab = rgb_to_lab(scaled[mask_bool])
    else:
        lab = rgb_to_lab(pixels[mask_bool])
    a = lab[:, 1]
    b = lab[:, 2]
    a_edges = np.linspace(-128.0, 127.0, bins[0] + 1)
    b_edges = np.linspace(-128.0, 127.0, bins[1] + 1)
    hist, _ = np.histogramdd(np.stack([a, b], axis=1), bins=(a_edges, b_edges))
    total = hist.sum()
    if total <= 0:
        return None
    return (hist / total).reshape(-1)


def chroma_distance(
    a: tuple[float, float] | None, b: tuple[float, float] | None
) -> float | None:
    """Euclidean distance between two (a*, b*) chrominance points.

    0 = identical hue/saturation; roughly 2-3 is a human just-noticeable
    difference in full Lab, larger in the a*b* plane alone since it omits
    L*'s contribution -- treat this as "clearly a different color" starting
    somewhere in the tens, not as a calibrated Delta-E.
    """
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _validate_inputs(
    source: Sequence[ObjectMask],
    target: Sequence[ObjectMask],
    source_features: FeatureDescriptorBatch,
    target_features: FeatureDescriptorBatch,
    source_gate_accepted: np.ndarray,
    target_gate_accepted: np.ndarray,
) -> None:
    if source_features.vectors.shape[0] != len(source):
        raise ValueError("source descriptor count differs from proposal count")
    if target_features.vectors.shape[0] != len(target):
        raise ValueError("target descriptor count differs from proposal count")
    if np.asarray(source_gate_accepted).shape != (len(source),):
        raise ValueError("source gate flags differ from proposal count")
    if np.asarray(target_gate_accepted).shape != (len(target),):
        raise ValueError("target gate flags differ from proposal count")
    ids = [_proposal_id(obj, index + 1) for index, obj in enumerate(source)]
    if len(ids) != len(set(ids)):
        raise ValueError("source proposal IDs must be unique")
    ids = [_proposal_id(obj, index + 1) for index, obj in enumerate(target)]
    if len(ids) != len(set(ids)):
        raise ValueError("target proposal IDs must be unique")


def pair_and_classify_gate_features(
    source: Sequence[ObjectMask],
    target: Sequence[ObjectMask],
    source_features: FeatureDescriptorBatch,
    target_features: FeatureDescriptorBatch,
    source_gate_accepted: np.ndarray,
    target_gate_accepted: np.ndarray,
    *,
    same_threshold: float,
    different_margin: float,
    minimum_spatial_iou: float,
    maximum_centroid_distance: float,
    area_ratio_bounds: tuple[float, float],
    source_image: np.ndarray | None = None,
    target_image: np.ndarray | None = None,
    color_different_threshold: float | None = None,
    color_method: str = "chroma",
    color_coverage: np.ndarray | None = None,
    color_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    color_scales: tuple[float, float, float] = (0.15, 0.20, 0.20),
    color_minimum_mask_pixels: int = 64,
    color_minimum_context_pixels: int = 4096,
    color_maximum_clipped_fraction: float = 0.25,
    color_apply_to_embedding_bands: Sequence[str] = ("similar", "uncertain"),
) -> FeatureVetoResult:
    """Pair by location first, then apply hard and guarded feature vetoes.

    ``source_image``/``target_image`` (the same rendered/real RGB frames the
    proposal masks were drawn on) opt into the fourth, color-based signal:
    when both are given and a pair's cosine reads "similar" or "uncertain",
    a CIE Lab a*/b* chrominance distance (see ``chroma_distance``) that
    reaches or exceeds ``color_different_threshold`` promotes the pair to
    "different" anyway. Omit either image (or the threshold) to keep the
    original cosine-only behavior exactly as it was.
    """

    source = list(source)
    target = list(target)
    source_gate_accepted = np.asarray(source_gate_accepted, dtype=bool)
    target_gate_accepted = np.asarray(target_gate_accepted, dtype=bool)
    color_enabled = (
        source_image is not None
        and target_image is not None
        and color_different_threshold is not None
    )
    if color_method not in {
        "chroma",
        "illum_norm_hist",
        "chroma_hist",
        "object_only",
        "global_exposure",
        "global_illumination",
    }:
        raise ValueError("unsupported color_method")
    if color_enabled and float(color_different_threshold) < 0:
        raise ValueError("color_different_threshold must be non-negative")
    allowed_color_bands = set(map(str, color_apply_to_embedding_bands))
    if not allowed_color_bands.issubset({"similar", "uncertain"}):
        raise ValueError("color evidence may apply only to similar/uncertain bands")
    _validate_inputs(
        source,
        target,
        source_features,
        target_features,
        source_gate_accepted,
        target_gate_accepted,
    )
    if not 0 <= minimum_spatial_iou <= 1:
        raise ValueError("minimum_spatial_iou must be in [0, 1]")
    if not 0 <= maximum_centroid_distance <= math.sqrt(2):
        raise ValueError("maximum_centroid_distance is invalid")
    if not 0 <= different_margin <= 2:
        raise ValueError("different_margin is invalid")
    low_area, high_area = map(float, area_ratio_bounds)
    if low_area <= 0 or high_area < low_area:
        raise ValueError("invalid area-ratio bounds")

    spatial_iou = pairwise_mask_iou(source, target)
    similarity = cosine_similarity_matrix(source_features, target_features)
    source_areas = np.asarray([np.asarray(obj.mask, bool).sum() for obj in source])
    target_areas = np.asarray([np.asarray(obj.mask, bool).sum() for obj in target])
    ratios = target_areas[None, :] / np.maximum(source_areas[:, None], 1)
    source_centroids = _centroids(source)
    target_centroids = _centroids(target)
    distances = np.linalg.norm(
        source_centroids[:, None, :] - target_centroids[None, :, :], axis=2
    )
    valid = (
        source_features.valid[:, None]
        & target_features.valid[None, :]
        & (spatial_iou >= float(minimum_spatial_iou))
        & (distances <= float(maximum_centroid_distance))
        & (ratios >= low_area)
        & (ratios <= high_area)
    )
    # Pairing deliberately depends only on aligned geometry. Looking at cosine
    # while selecting the counterpart would bias the later similarity test.
    score = np.where(valid, spatial_iou, -1.0)
    row_best = np.argmax(score, axis=1) if len(target) else np.empty(len(source), int)
    column_best = np.argmax(score, axis=0) if len(source) else np.empty(len(target), int)
    different_threshold = float(same_threshold) - float(different_margin)
    pairs: list[FeatureVetoPair] = []
    paired_source: set[int] = set()
    paired_target: set[int] = set()
    for source_index, target_index in enumerate(row_best):
        target_index = int(target_index)
        if not len(target) or not valid[source_index, target_index]:
            continue
        if int(column_best[target_index]) != source_index:
            continue
        cosine = float(similarity[source_index, target_index])
        if cosine >= float(same_threshold):
            band = "similar"
        elif cosine <= different_threshold:
            band = "different"
        else:
            band = "uncertain"
        band_source = "embedding"
        color_dist = None
        color_components = None
        color_evidence_status = None
        if color_enabled and band in allowed_color_bands:
            # Only asked to promote, never to rescind: an embedding-driven
            # "different" call already has its evidence and color adds
            # nothing further in that direction.
            if color_method == "chroma":
                color_dist = chroma_distance(
                    mean_lab_chroma(source_image, source[source_index].mask),
                    mean_lab_chroma(target_image, target[target_index].mask),
                )
                color_evidence_status = "valid" if color_dist is not None else "invalid"
            elif color_method in {"illum_norm_hist", "chroma_hist"}:
                # Compute masked a*/b* histograms. ``illum_norm_hist`` scales the
                # source images' linear RGB so their median L* matches l_ref
                # before histogramming; ``chroma_hist`` uses raw a*/b* histograms.
                normalize = color_method == "illum_norm_hist"
                hist1 = masked_ab_hist(
                    source_image, source[source_index].mask, bins=(16, 16), normalize_luminance=normalize
                )
                hist2 = masked_ab_hist(
                    target_image, target[target_index].mask, bins=(16, 16), normalize_luminance=normalize
                )
                if hist1 is None or hist2 is None:
                    color_dist = None
                else:
                    # histogram_intersection returns similarity in [0,1]. Convert
                    # to a distance where larger means more different.
                    sim = histogram_intersection(hist1, hist2)
                    color_dist = None if sim is None else float(1.0 - sim)
                color_evidence_status = "valid" if color_dist is not None else "invalid"
            else:
                evidence = illumination_relative_color_distance(
                    source_image,
                    source[source_index].mask,
                    target_image,
                    target[target_index].mask,
                    coverage=color_coverage,
                    mode=color_method,
                    center=color_center,
                    scales=color_scales,
                    minimum_mask_pixels=color_minimum_mask_pixels,
                    minimum_context_pixels=color_minimum_context_pixels,
                    maximum_clipped_fraction=color_maximum_clipped_fraction,
                )
                color_dist = evidence.distance
                color_components = evidence.components
                color_evidence_status = evidence.status
            if color_dist is not None and color_dist >= float(color_different_threshold):
                band = "different"
                band_source = "color"
        pair = FeatureVetoPair(
            source_index=source_index,
            target_index=target_index,
            source_proposal_id=_proposal_id(source[source_index], source_index + 1),
            target_proposal_id=_proposal_id(target[target_index], target_index + 1),
            spatial_iou=float(spatial_iou[source_index, target_index]),
            centroid_distance=float(distances[source_index, target_index]),
            area_ratio=float(ratios[source_index, target_index]),
            cosine=cosine,
            same_threshold=float(same_threshold),
            different_threshold=different_threshold,
            feature_band=band,
            source_gate_accepted=bool(source_gate_accepted[source_index]),
            target_gate_accepted=bool(target_gate_accepted[target_index]),
            color_distance=color_dist,
            color_different_threshold=(
                float(color_different_threshold) if color_enabled else None
            ),
            band_source=band_source,
            color_components=color_components,
            color_evidence_status=color_evidence_status,
        )
        pairs.append(pair)
        paired_source.add(source_index)
        paired_target.add(target_index)

    hard_source: set[int] = set()
    hard_target: set[int] = set()
    guarded_source: set[int] = set()
    guarded_target: set[int] = set()
    for pair in pairs:
        # Geometry-rejected objects remain changed but are not counted as new
        # feature promotions. Their paired accepted endpoint is promoted.
        if pair.feature_band in {"different", "uncertain"}:
            if pair.source_gate_accepted:
                hard_source.add(pair.source_proposal_id)
            if pair.target_gate_accepted:
                hard_target.add(pair.target_proposal_id)
        if pair.feature_band == "different":
            if pair.source_gate_accepted:
                guarded_source.add(pair.source_proposal_id)
            if pair.target_gate_accepted:
                guarded_target.add(pair.target_proposal_id)

    pair_bands = {
        name: sum(pair.feature_band == name for pair in pairs)
        for name in ("similar", "uncertain", "different")
    }

    def endpoint_funnel(
        objects: Sequence[ObjectMask],
        features: FeatureDescriptorBatch,
        gate_accepted: np.ndarray,
        paired: set[int],
        side: str,
    ) -> dict[str, int]:
        accepted_indices = set(np.flatnonzero(gate_accepted).tolist())
        invalid = {index for index in accepted_indices if not features.valid[index]}
        no_counterpart = accepted_indices - invalid - paired
        counts = {
            "visible": len(objects),
            "geometry_accepted": len(accepted_indices),
            "geometry_rejected": int((~gate_accepted).sum()),
            "descriptor_invalid_abstain": len(invalid),
            "no_reciprocal_counterpart_abstain": len(no_counterpart),
            "similar_preserved": 0,
            "uncertain_abstain": 0,
            "different_guarded_promoted": 0,
            "hard_promoted": 0,
        }
        for pair in pairs:
            index = pair.source_index if side == "source" else pair.target_index
            accepted = pair.source_gate_accepted if side == "source" else pair.target_gate_accepted
            if not accepted:
                continue
            if pair.feature_band == "similar":
                counts["similar_preserved"] += 1
            elif pair.feature_band == "uncertain":
                counts["uncertain_abstain"] += 1
                counts["hard_promoted"] += 1
            else:
                counts["different_guarded_promoted"] += 1
                counts["hard_promoted"] += 1
        terminal = (
            counts["geometry_rejected"]
            + counts["descriptor_invalid_abstain"]
            + counts["no_reciprocal_counterpart_abstain"]
            + counts["similar_preserved"]
            + counts["uncertain_abstain"]
            + counts["different_guarded_promoted"]
        )
        if terminal != counts["visible"]:
            raise AssertionError(f"{side} feature funnel does not conserve proposals")
        return counts

    funnel = {
        "pairs": {
            "reciprocal_same_place": len(pairs),
            **pair_bands,
            "hard_veto_pairs": pair_bands["different"] + pair_bands["uncertain"],
            "guarded_veto_pairs": pair_bands["different"],
            "both_gate_accepted": sum(
                pair.source_gate_accepted and pair.target_gate_accepted for pair in pairs
            ),
            "source_only_gate_accepted": sum(
                pair.source_gate_accepted and not pair.target_gate_accepted for pair in pairs
            ),
            "target_only_gate_accepted": sum(
                not pair.source_gate_accepted and pair.target_gate_accepted for pair in pairs
            ),
            "both_gate_rejected": sum(
                not pair.source_gate_accepted and not pair.target_gate_accepted for pair in pairs
            ),
        },
        "source": endpoint_funnel(
            source,
            source_features,
            source_gate_accepted,
            paired_source,
            "source",
        ),
        "target": endpoint_funnel(
            target,
            target_features,
            target_gate_accepted,
            paired_target,
            "target",
        ),
    }
    return FeatureVetoResult(
        pairs=tuple(pairs),
        hard_source_ids=tuple(sorted(hard_source)),
        hard_target_ids=tuple(sorted(hard_target)),
        guarded_source_ids=tuple(sorted(guarded_source)),
        guarded_target_ids=tuple(sorted(guarded_target)),
        funnel=funnel,
    )


def _copy_object(obj: ObjectMask, label: Label, source: str) -> ObjectMask:
    return ObjectMask(
        mask=np.asarray(obj.mask, dtype=bool).copy(),
        score=float(obj.score),
        label=label,
        source=source,
        metadata=copy.deepcopy(obj.metadata),
    )


def ordinary_promoted_objects(
    source_by_id: Mapping[int, ObjectMask],
    target_by_id: Mapping[int, ObjectMask],
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    forward_tracks: Mapping[int, np.ndarray | None],
    reverse_tracks: Mapping[int, np.ndarray | None],
) -> tuple[list[ObjectMask], dict[str, int]]:
    """Apply the unchanged source/target downstream tracking semantics."""

    objects: list[ObjectMask] = []
    counts = {"added": 0, "removed": 0, "moved": 0}
    for proposal_id in source_ids:
        if proposal_id not in source_by_id or proposal_id not in forward_tracks:
            raise ValueError(f"missing promoted source/track {proposal_id}")
        source = source_by_id[proposal_id]
        track = forward_tracks[proposal_id]
        if track is None:
            objects.append(_copy_object(source, Label.REMOVED, "feature_veto_source_failed"))
            counts["removed"] += 1
        else:
            objects.append(
                ObjectMask(
                    mask=np.asarray(track, bool).copy(),
                    score=float(source.score),
                    label=Label.MOVED,
                    source="feature_veto_source_to_target",
                    metadata={"source_proposal_id": proposal_id},
                )
            )
            counts["moved"] += 1
    for proposal_id in target_ids:
        if proposal_id not in target_by_id or proposal_id not in reverse_tracks:
            raise ValueError(f"missing promoted target/track {proposal_id}")
        target = target_by_id[proposal_id]
        track = reverse_tracks[proposal_id]
        if track is None:
            objects.append(_copy_object(target, Label.ADDED, "feature_veto_target_failed"))
            counts["added"] += 1
        else:
            objects.append(_copy_object(target, Label.MOVED, "feature_veto_target_to_source"))
            counts["moved"] += 1
    return objects, counts


def merge_objects_with_parent(
    parent_labels: np.ndarray,
    objects: list[ObjectMask],
    coverage: np.ndarray,
    *,
    visibility_alpha: float,
    minimum_mask_area: int,
    replacement_overlap_iou: float = 0.10,
) -> tuple[np.ndarray, int]:
    """Merge promoted objects using the parent's complete ending protocol.

    In addition to visibility and label priority, the parent converts overlap
    between added and removed masks into ChangeSim's ``replaced`` class.  It
    is important to repeat that ordinary conversion here: the later direct
    SAM3 replacement branch is a separate experimental source of evidence.
    """

    parent = np.asarray(parent_labels, np.uint8)
    retained = filter_visible(
        objects, np.asarray(coverage, bool), visibility_alpha, minimum_mask_area
    )
    new_low = compose_labels(np.asarray(coverage).shape, retained)
    added = [item for item in retained if item.label == Label.ADDED]
    removed = [item for item in retained if item.label == Label.REMOVED]
    new_low = mark_replacements(
        new_low, added, removed, float(replacement_overlap_iou)
    )
    new = np.asarray(
        Image.fromarray(new_low).resize(parent.shape[::-1], Image.Resampling.NEAREST),
        np.uint8,
    )
    output = parent.copy()
    # Existing direct replacements and warped pixels are protected. Otherwise
    # reuse the published priority MOVED > REMOVED > ADDED > UNCHANGED.
    protected = (parent == int(Label.REPLACED)) | (parent == int(Label.WARPED))
    moved = (new == int(Label.MOVED)) & ~protected
    removed = (
        (new == int(Label.REMOVED))
        & ~protected
        & (parent != int(Label.MOVED))
    )
    added = (new == int(Label.ADDED)) & (parent == int(Label.UNCHANGED))
    output[added] = int(Label.ADDED)
    output[removed] = int(Label.REMOVED)
    output[moved] = int(Label.MOVED)
    # The baseline applies replacement after normal priority composition, so
    # it wins over added/removed/moved.  Only the still-higher warped label is
    # protected when these incremental predictions are merged into the
    # parent (``parent_labels``) raster.
    replaced = (new == int(Label.REPLACED)) & (parent != int(Label.WARPED))
    output[replaced] = int(Label.REPLACED)
    return output, len(retained)


def direct_replacement_mask(
    pairs: Sequence[FeatureVetoPair],
    source_by_id: Mapping[int, ObjectMask],
    target_by_id: Mapping[int, ObjectMask],
    native_shape: tuple[int, int],
    *,
    allowed_band_sources: Sequence[str] = ("embedding", "color"),
) -> np.ndarray:
    """Return the conservative intersection of confidently different pairs."""

    if not source_by_id or not target_by_id:
        return np.zeros(native_shape, bool)
    shape = np.asarray(next(iter(source_by_id.values())).mask).shape
    mask = np.zeros(shape, bool)
    allowed = set(map(str, allowed_band_sources))
    for pair in pairs:
        if pair.feature_band != "different" or pair.band_source not in allowed:
            continue
        mask |= np.asarray(source_by_id[pair.source_proposal_id].mask, bool) & np.asarray(
            target_by_id[pair.target_proposal_id].mask, bool
        )
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            native_shape[::-1], Image.Resampling.NEAREST
        ),
        bool,
    )


def apply_direct_semantics(
    changed_labels: np.ndarray,
    replacement_mask: np.ndarray,
    moved_masks: Sequence[np.ndarray],
) -> np.ndarray:
    """Relabel existing changed support; replacement wins final arbitration."""

    original = np.asarray(changed_labels, np.uint8)
    output = original.copy()
    changed_support = original != int(Label.UNCHANGED)
    for mask in moved_masks:
        native = np.asarray(mask, bool)
        if native.shape != original.shape:
            raise ValueError("direct moved mask shape differs from label map")
        eligible = native & changed_support & (output != int(Label.REPLACED))
        output[eligible] = int(Label.MOVED)
    replacement = np.asarray(replacement_mask, bool)
    if replacement.shape != original.shape:
        raise ValueError("direct replacement mask shape differs from label map")
    output[replacement & changed_support] = int(Label.REPLACED)
    if not np.array_equal(
        output != int(Label.UNCHANGED), original != int(Label.UNCHANGED)
    ):
        raise AssertionError("direct semantic reasoning changed binary support")
    return output


class _TrackerProtocol:
    """Structural type for the tracker this stage needs: see adapters/sam2.py."""

    def track(
        self, masks: Sequence[np.ndarray], source_image: np.ndarray, target_image: np.ndarray
    ) -> list[Any]:  # pragma: no cover - protocol only
        raise NotImplementedError


def apply_feature_veto_direct_replacement(
    source: Sequence[ObjectMask],
    target: Sequence[ObjectMask],
    source_features: FeatureDescriptorBatch,
    target_features: FeatureDescriptorBatch,
    source_gate_accepted: np.ndarray,
    target_gate_accepted: np.ndarray,
    baseline_labels: np.ndarray,
    coverage: np.ndarray,
    tracker: _TrackerProtocol,
    source_image: np.ndarray,
    target_image: np.ndarray,
    *,
    same_threshold: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, FeatureVetoResult]:
    """Gate same-place proposal pairs on appearance, then paint direct replacement.

    Faithful extraction of the "guarded direct replacement" composition from
    ``scripts/run_sam3_feature_veto_gate_experiment.py`` (the one variant, of
    five the original ablation computed, that is load-bearing here): pair
    visible source/target proposals by aligned location, veto a same-place
    pair whose SAM3 descriptors read confidently different, re-track exactly
    those newly guarded proposals with a fresh forward/backward SAM2 pass,
    merge the results onto ``baseline_labels`` (stage 7's motion-and-
    replacement-refined raster) with the standard visibility/priority
    protocol, and finally paint the conservative intersection of every
    confidently-different pair directly as REPLACED. The other four
    ablation rungs (a plain baseline replay, a hard/uncertain veto without
    direct replacement, and a variant that also folds in moved-object
    reasoning) are not computed -- see this module's docstring for why only
    this one is load-bearing for the method.

    ``source``/``target`` must be the same visible proposal lists, in the
    same order, as ``source_features``/``target_features`` were pooled from
    (i.e. exactly the appearance stage's inputs/outputs);
    ``source_gate_accepted``/``target_gate_accepted`` are the complement of
    stage 3's changed-proposal flags (True where the clean-render gate
    accepted, i.e. called the proposal statically unchanged).
    """

    same_place = config["same_place_pairing"]
    decision = pair_and_classify_gate_features(
        source,
        target,
        source_features,
        target_features,
        source_gate_accepted,
        target_gate_accepted,
        same_threshold=float(same_threshold),
        different_margin=float(config["feature_thresholds"]["different_identity_margin"]),
        minimum_spatial_iou=float(same_place["minimum_spatial_iou"]),
        maximum_centroid_distance=float(same_place["maximum_normalized_centroid_distance"]),
        area_ratio_bounds=tuple(same_place["area_ratio_bounds"]),
    )
    source_by_id = {_proposal_id(obj, index + 1): obj for index, obj in enumerate(source)}
    target_by_id = {_proposal_id(obj, index + 1): obj for index, obj in enumerate(target)}

    forward_attempts = tracker.track(
        [np.asarray(source_by_id[pid].mask, bool) for pid in decision.guarded_source_ids],
        source_image,
        target_image,
    )
    reverse_attempts = tracker.track(
        [np.asarray(target_by_id[pid].mask, bool) for pid in decision.guarded_target_ids],
        target_image,
        source_image,
    )
    forward_tracks = {
        pid: (np.asarray(attempt.mask, bool) if attempt.accepted else None)
        for pid, attempt in zip(decision.guarded_source_ids, forward_attempts, strict=True)
    }
    reverse_tracks = {
        pid: (np.asarray(attempt.mask, bool) if attempt.accepted else None)
        for pid, attempt in zip(decision.guarded_target_ids, reverse_attempts, strict=True)
    }
    guarded_objects, _ = ordinary_promoted_objects(
        source_by_id,
        target_by_id,
        decision.guarded_source_ids,
        decision.guarded_target_ids,
        forward_tracks,
        reverse_tracks,
    )
    changed_object_pipeline = config["changed_object_pipeline"]
    guarded_labels, _ = merge_objects_with_parent(
        baseline_labels,
        guarded_objects,
        coverage,
        visibility_alpha=float(changed_object_pipeline["visibility_alpha"]),
        minimum_mask_area=int(changed_object_pipeline["minimum_mask_area"]),
        replacement_overlap_iou=float(changed_object_pipeline["replacement_overlap_iou"]),
    )
    replacement = direct_replacement_mask(
        decision.pairs, source_by_id, target_by_id, np.asarray(baseline_labels).shape
    )
    direct_labels = apply_direct_semantics(guarded_labels, replacement, [])
    return direct_labels, decision
