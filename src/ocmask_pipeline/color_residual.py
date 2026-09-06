"""Dense, pixel-level color-residual signal for same-footprint object
replacement (e.g. a red block swapped for a blue one), found directly from
render_t0/image_t1 rather than from identity matching between two SAM3
proposals -- the replaced object is frequently never segmented as its own
proposal at all (it gets absorbed into a larger surrounding surface's mask,
e.g. a whole tabletop), so appearance-based identity matching never gets a
chance to compare it against anything, regardless of whether that matching
uses color.

Neither GOLDILOCS nor SceneDiff have an equivalent: GOLDILOCS's only dense
appearance-residual mechanism is SSIM on masks that already survived its own
automatic segmentation -- exactly the same segmentation-granularity blind
spot this module exists to catch, not something it protects against.
SceneDiff has no dense-residual mechanism at all.

Validated end-to-end this session on Meeting_room_Instance_1's
Inst_1_test_IMG_1862 (a red T-block swapped for a blue one, on an otherwise
unchanged white table SAM3 segments as a single proposal): chromaticity
distance -- dropping lightness, keeping hue+saturation -- isolates the
swapped object's exact silhouette, where a naive RGB distance is swamped by
specular-highlight/rendering-splat noise across the whole frame. Known
limitation: only catches sufficiently saturated color changes, gated by
``minimum_colorfulness`` -- a low-saturation appearance change (e.g. a faint
chalk marking) falls below that gate and is not caught by this signal.
"""

from __future__ import annotations

import numpy as np


def chromaticity_distance(
    first: np.ndarray, second: np.ndarray, epsilon: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel chromaticity-space distance between two same-shape (H, W, 3)
    RGB images, plus the minimum "colorfulness" (max-min channel spread) of
    the two. Chromaticity (R/(R+G+B), G/(R+G+B)) captures hue+saturation
    while dropping lightness, which is what makes it robust to the
    specular-highlight and splat-noise differences a raw RGB distance picks
    up everywhere a real reconstruction and a real photo will never match
    exactly. Colorfulness gates out near-black/white/gray pixels, where
    chromaticity is numerically unstable (dividing two tiny or two large-but-
    similar numbers) and uninformative regardless of any real appearance
    change -- callers should reject pixels below some minimum on this before
    trusting the distance.
    """
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    first_chroma = first[..., :2] / (first.sum(axis=-1, keepdims=True) + epsilon)
    second_chroma = second[..., :2] / (second.sum(axis=-1, keepdims=True) + epsilon)
    distance = np.linalg.norm(first_chroma - second_chroma, axis=-1)
    colorfulness = np.minimum(
        first.max(axis=-1) - first.min(axis=-1),
        second.max(axis=-1) - second.min(axis=-1),
    )
    return distance, colorfulness


def color_replacement_candidate_mask(
    render_t0: np.ndarray,
    image_t1: np.ndarray,
    coverage: np.ndarray,
    minimum_colorfulness: float,
    residual_percentile: float,
) -> np.ndarray:
    """Boolean (H, W) mask of pixels whose color changed enough, on both
    sides, to plausibly be part of a same-footprint object replacement.

    Not yet split into connected components or filtered by minimum area --
    see change_detection.find_color_replacement_regions, which does both and
    also excludes pixels an earlier stage already explained.

    ``residual_percentile`` is taken over colorfulness-gated, covered pixels
    only, so it adapts to how much real appearance variation this specific
    frame happens to contain rather than using one fixed absolute distance
    cutoff across scenes of very different character.
    """
    distance, colorfulness = chromaticity_distance(render_t0, image_t1)
    gated = np.asarray(coverage, dtype=bool) & (colorfulness >= minimum_colorfulness)
    if not gated.any():
        return np.zeros(coverage.shape, dtype=bool)
    threshold = np.percentile(distance[gated], residual_percentile)
    return gated & (distance >= threshold)
