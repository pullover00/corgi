"""Ground-contact correction for MASt3R depth regressions fused into background.

An added object whose MASt3R depth happens to regress onto a background
plane (self-consistent with both cameras' own observations, see
``experiments/collision_reid_promotion.py``) never registers a conflict in
``reverse_depth_filter``, so it is silently kept as "static" -- reference case:
a forklift added to a warehouse scene regressed to 5.11m / -2.27m (below the
floor, physically impossible), fused into a wall, and ``keep1`` was ``True``
for essentially the whole object.

This module fixes that class of bug with no new depth model and no search:
find the room's floor from points already trusted by the *unmodified*
``reverse_depth_filter`` (stage 2), find where the candidate object's own
mask visibly contacts that floor in-frame (stage 3), and solve, once, for the
rigid depth correction the floor-contact assumption implies (stage 4). The
whole pipeline is driven from one signal computed before any of this expensive
geometry runs: what fraction of the candidate's own points the *existing*
``reverse_depth_filter`` already silently kept as static (``keep_fraction``
below). A high fraction is this exact bug; a low fraction means
``reverse_depth_filter`` already did its job and this method does not apply.

Two callers consume the fitted floor + correction very differently, on
purpose (see their docstrings): ``exclude_from_keep`` for change *detection*
(never substitutes a floor-consistent point -- that would manufacture false
agreement and bury the very evidence a detector needs), and
``render_fill_for_mask`` for *rendering* clean plates (deletes the object and
paints in real floor texture, because a clean plate must never show content
that isn't really there). ``apply_rigid_median_correction`` is a third,
separate function: it relocates the object's own points and exists to
mechanistically verify the bug/fix (feed its output back through
``reverse_depth_filter`` and watch the conflict rate jump), not as a
production detection/rendering code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Stage 1: up-vector from the camera's own rotation.
# ---------------------------------------------------------------------------


def up_vector_from_world_to_camera(world_to_camera: np.ndarray) -> np.ndarray:
    """Derive the world-space "up" direction from a camera's own pose.

    ``world_to_camera`` transforms world points into camera space
    (``p_cam = R @ p_world + t``) under this codebase's pinhole convention,
    where image row/``v`` grows downward, so the camera's own +Y axis is
    "down" in the picture (see ``geometry.project_points``). The world-space
    direction that maps onto that +Y camera axis is ``R^T @ [0, 1, 0]``,
    which -- because ``R`` is orthonormal -- is simply the second row of
    ``R`` read as a vector; negating it gives "world up" as this camera sees
    it.

    This is deliberately derived from calibrated camera geometry alone, not
    image content: an earlier attempt used "points below the object in the
    frame" as an up-vector proxy and silently locked onto a wall whenever the
    camera was tilted, because image-down is not scene-down in that case.
    """
    rotation = np.asarray(world_to_camera, dtype=np.float64)[:3, :3]
    down = rotation[1, :]
    norm = np.linalg.norm(down)
    if norm < 1e-9:
        raise ValueError("degenerate camera rotation matrix (zero-length row)")
    return -down / norm


def _camera_center_world(world_to_camera: np.ndarray) -> np.ndarray:
    """Camera center in world coordinates: solve ``0 = R @ C + t``."""
    wtc = np.asarray(world_to_camera, dtype=np.float64)
    rotation, translation = wtc[:3, :3], wtc[:3, 3]
    return -rotation.T @ translation


def _pixel_rays_world(
    rows: np.ndarray, cols: np.ndarray, intrinsics: np.ndarray, world_to_camera: np.ndarray
) -> np.ndarray:
    """World-frame ray directions parametrized so that a point placed at this
    codebase's own definition of "depth" (the camera-frame z coordinate, see
    ``geometry.project_points``/``reverse_depth_filter``) along the ray sits
    at ``camera_center + depth * ray``. Rays are intentionally *not*
    normalized to unit length -- the pinhole back-projection already has
    z=1, which is exactly what makes the scale factor along each ray equal
    to camera-frame depth, matching every other depth value in this
    pipeline.
    """
    wtc = np.asarray(world_to_camera, dtype=np.float64)
    rotation = wtc[:3, :3]
    k_inv = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64))
    pixels_homogeneous = np.stack(
        [np.asarray(cols, dtype=np.float64), np.asarray(rows, dtype=np.float64), np.ones(len(rows))],
        axis=1,
    )
    camera_rays = pixels_homogeneous @ k_inv.T
    return camera_rays @ rotation  # == (rotation.T @ camera_rays.T).T


def ray_plane_implied_depth(
    rows: np.ndarray,
    cols: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    floor: "FloorPlane",
    *,
    minimum_ray_plane_alignment: float = 1e-6,
) -> np.ndarray:
    """Solve ``d = (P - C)*n / (r*n)`` per pixel: the depth the floor-contact
    assumption implies for that pixel's own camera ray. Rays nearly parallel
    to the floor plane (the object's silhouette pointing at the horizon)
    have no well-conditioned solution and are returned as NaN.
    """
    center = _camera_center_world(world_to_camera)
    rays = _pixel_rays_world(rows, cols, intrinsics, world_to_camera)
    denominator = rays @ floor.normal
    depth = np.full(len(rows), np.nan, dtype=np.float64)
    valid = np.abs(denominator) > minimum_ray_plane_alignment
    depth[valid] = ((floor.point - center) @ floor.normal) / denominator[valid]
    return depth


# ---------------------------------------------------------------------------
# Stage 2: floor detection -- histogram + RANSAC + least-squares refit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorPlane:
    """A fitted floor plane plus enough provenance to audit the fit."""

    point: np.ndarray
    normal: np.ndarray
    inlier_points: np.ndarray
    inlier_colors: np.ndarray | None
    inlier_count: int
    residual_rms: float
    candidate_band_point_count: int
    height_band: tuple[float, float]
    up_agreement_cosine: float


# A genuine floor should sit almost exactly perpendicular to the
# camera-derived up-axis; the spec's own reference case (forklift pair) fit
# at cosine 0.97-0.995 agreement between the refit normal and up. A fit that
# clears the RANSAC inlier-count bar but disagrees this much with up
# (observed on a real pair: 0.668, ~48 degrees off) is strong evidence the
# histogram's lowest-occupied band caught a *different* mostly-flat,
# mostly-low surface -- a ramp, the base of a shelving unit, a wall seen at
# a shallow angle -- rather than the actual floor, and RANSAC/SVD then fit
# that surface well without it being the surface this method needs. 0.9 is
# set comfortably below the reference cases (>=0.99 on synthetic data and
# the confirmed-good forklift pair) and comfortably above the observed bad
# fit (0.668), so it separates the two without being tuned tighter than
# that one real counterexample justifies. A rejected fit propagates as "not
# applicable" (``None``), not a low-quality plane a caller might use
# anyway -- confirmed by mechanistic verification: on the real pair this
# rejects, the before/after reverse_depth_filter conflict rate did not move
# at all (stayed 0%) after "correcting" with the bad plane, unlike the
# genuine floor case where it jumped from 16% to 100%.
FLOOR_UP_AGREEMENT_MINIMUM_COSINE = 0.9


def detect_floor_plane(
    static_points: np.ndarray,
    static_colors: np.ndarray | None,
    up: np.ndarray,
    *,
    height_bins: int = 200,
    low_fraction: float = 0.4,
    band_padding_fraction: float = 0.01,
    ransac_iterations: int = 300,
    ransac_inlier_threshold_m: float = 0.03,
    minimum_band_points: int = 200,
    minimum_inliers: int = 100,
    minimum_up_agreement_cosine: float = FLOOR_UP_AGREEMENT_MINIMUM_COSINE,
    rng: np.random.Generator | None = None,
) -> "FloorPlane | None":
    """Find the room's floor among already-trusted static points.

    No semantic labeling: heights are histogrammed along the camera-derived
    ``up`` axis, and the *dominant* bin among the lowest ``low_fraction`` of
    occupied bins is taken as the floor candidate band -- the floor doesn't
    need to be recognized, only to be the largest, flattest, lowest cluster
    in the room, which is what a real floor is. RANSAC on that band gives a
    plane robust to whatever furniture/clutter also falls in the height band;
    a least-squares SVD refit on the RANSAC inlier set sharpens it.

    The refit plane's normal is then checked against the *independent*
    camera-derived ``up`` vector (``minimum_up_agreement_cosine``): a real
    floor must sit almost exactly perpendicular to up, so a fit that
    disagrees sharply is evidence the lowest-occupied-band heuristic caught
    the wrong surface, not a floor with a large residual (see the module-
    level constant's docstring for the real counterexample this catches).

    Returns ``None`` when there isn't enough trusted geometry to fit a floor
    (too few static points, or no height variation at all), or when the fit
    fails the up-agreement check above -- callers must treat both as "not
    applicable", not as an error.
    """
    points = np.asarray(static_points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = None
    if static_colors is not None:
        colors = np.asarray(static_colors).reshape(-1, 3)[finite]
    if len(points) < minimum_band_points:
        return None

    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    heights = points @ up
    low, high = float(heights.min()), float(heights.max())
    if high <= low:
        return None

    bin_edges = np.linspace(low, high, height_bins + 1)
    bin_index = np.clip(np.digitize(heights, bin_edges) - 1, 0, height_bins - 1)
    counts = np.bincount(bin_index, minlength=height_bins)
    cutoff_bin = max(1, int(height_bins * low_fraction))
    low_counts = counts[:cutoff_bin]
    if not low_counts.any():
        return None
    peak_bin = int(np.argmax(low_counts))
    band_low, band_high = bin_edges[peak_bin], bin_edges[peak_bin + 1]
    pad = band_padding_fraction * (high - low)
    band_mask = (heights >= band_low - pad) & (heights <= band_high + pad)
    band_points = points[band_mask]
    band_colors = colors[band_mask] if colors is not None else None
    if len(band_points) < minimum_band_points:
        return None

    rng = rng if rng is not None else np.random.default_rng(0)
    sample_count = len(band_points)
    if sample_count < 3:
        return None
    best_inliers, best_count = None, -1
    for _ in range(ransac_iterations):
        p0, p1, p2 = band_points[rng.choice(sample_count, size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        distance = (band_points - p0) @ normal
        inliers = np.abs(distance) <= ransac_inlier_threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best_inliers, best_count = inliers, count
    if best_inliers is None or best_count < minimum_inliers:
        return None

    inlier_points = band_points[best_inliers]
    inlier_colors = band_colors[best_inliers] if band_colors is not None else None
    centroid = inlier_points.mean(axis=0)
    centered = inlier_points - centroid
    # Least-squares refit: the plane normal is the least-variance direction,
    # i.e. the right singular vector with the smallest singular value.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    if normal @ up < 0:
        normal = -normal
    residual_rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    up_agreement_cosine = float(normal @ up)
    if up_agreement_cosine < minimum_up_agreement_cosine:
        # The lowest-occupied height band fit a real plane, but not one
        # perpendicular to the camera's own up-axis -- almost certainly not
        # the floor (see FLOOR_UP_AGREEMENT_MINIMUM_COSINE's docstring).
        # Treat exactly like "no floor found": the correction is not
        # applicable, not "applied against a low-confidence plane."
        return None

    return FloorPlane(
        point=centroid,
        normal=normal,
        inlier_points=inlier_points,
        inlier_colors=inlier_colors,
        inlier_count=int(best_count),
        residual_rms=residual_rms,
        candidate_band_point_count=int(len(band_points)),
        height_band=(float(band_low), float(band_high)),
        up_agreement_cosine=up_agreement_cosine,
    )


# ---------------------------------------------------------------------------
# Stage 3: contact-band extraction from a SAM mask.
# ---------------------------------------------------------------------------


def extract_contact_band(mask: np.ndarray) -> np.ndarray:
    """Return the bottom edge of ``mask``: one (row, col) pair per occupied
    column, the largest (lowest-in-image) row in that column.

    Because ``mask`` is a real per-pixel proposal in an already-rendered
    view, every ``True`` pixel is by construction the nearest visible
    surface along its own ray -- there is nothing nearer occluding it to
    filter out separately. The per-column bottom-most pixel is therefore
    exactly "the part of the object visibly resting on the floor."
    """
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.lexsort((ys, xs))  # sort by column, then row ascending within column
    xs_sorted, ys_sorted = xs[order], ys[order]
    is_last_in_column = np.r_[xs_sorted[1:] != xs_sorted[:-1], True]
    return np.stack([ys_sorted[is_last_in_column], xs_sorted[is_last_in_column]], axis=1)


# ---------------------------------------------------------------------------
# Stage 4: ray-plane intersection + median rigid correction, plus the
# whole-mask sanity check that decides base-only vs. whole-object rendering.
# ---------------------------------------------------------------------------

# Heuristic judgment calls (documented here so they are auditable without
# re-deriving them): an object whose own mask spans more than 60% of the
# image's vertical extent cannot plausibly be the compact, floor-resting
# object this method targets in-frame (the reference forklift case spans a
# small fraction of the frame); it is more likely a wall/structural element
# whose *upper* portion would be wrongly projected onto the floor if treated
# as one rigid floor-contact object. Similarly, the floor-implied depth
# across the *whole* mask should stay within a modest multiple of the
# contact band's own corrected depth -- the reference case's whole-mask
# spread (0.54-1.09m) is close to 1x its own corrected median depth
# (0.55m), so 1.5x leaves headroom without accepting a implausibly large
# spread. Neither constant is more precisely calibrated than that; both are
# simple, auditable heuristics per the spec, not tuned search parameters.
WHOLE_MASK_ROW_SPAN_FRACTION_LIMIT = 0.6
WHOLE_MASK_DEPTH_SPREAD_RELATIVE_LIMIT = 1.5


@dataclass(frozen=True)
class GroundContactCorrection:
    """Diagnostics + the one scalar (``correction_ratio``) stage 4 solves for."""

    succeeded: bool
    reason: str
    contact_pixels: np.ndarray
    contact_pixel_count: int
    original_median_contact_depth: float | None
    corrected_median_contact_depth: float | None
    correction_ratio: float | None
    contact_implied_depth_std: float | None
    whole_mask_row_span_fraction: float | None
    whole_mask_implied_depth_spread: float | None
    whole_mask_applicable: bool
    base_only: bool


def _empty_correction(reason: str, contact_pixels: np.ndarray | None = None) -> GroundContactCorrection:
    return GroundContactCorrection(
        succeeded=False,
        reason=reason,
        contact_pixels=contact_pixels if contact_pixels is not None else np.empty((0, 2), dtype=np.int64),
        contact_pixel_count=0 if contact_pixels is None else int(len(contact_pixels)),
        original_median_contact_depth=None,
        corrected_median_contact_depth=None,
        correction_ratio=None,
        contact_implied_depth_std=None,
        whole_mask_row_span_fraction=None,
        whole_mask_implied_depth_spread=None,
        whole_mask_applicable=False,
        base_only=False,
    )


def fit_ground_contact_correction(
    mask: np.ndarray,
    own_depth_map: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    floor: FloorPlane,
    *,
    row_span_fraction_limit: float = WHOLE_MASK_ROW_SPAN_FRACTION_LIMIT,
    depth_spread_relative_limit: float = WHOLE_MASK_DEPTH_SPREAD_RELATIVE_LIMIT,
) -> GroundContactCorrection:
    """Stage 4: the single rigid depth correction the floor-contact band implies,
    plus the whole-mask sanity check from the spec's "per-object geometric
    precondition" (a tall/floor-to-ceiling object needs a base-only fallback
    rather than whole-mask floor projection).
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return _empty_correction("empty_mask")

    contact = extract_contact_band(mask)
    if len(contact) == 0:
        return _empty_correction("no_contact_band")

    rows, cols = contact[:, 0], contact[:, 1]
    implied_depth = ray_plane_implied_depth(rows, cols, intrinsics, world_to_camera, floor)
    original_depth = np.asarray(own_depth_map, dtype=np.float64)[rows, cols]
    valid = (
        np.isfinite(implied_depth)
        & (implied_depth > 0)
        & np.isfinite(original_depth)
        & (original_depth > 0)
    )
    if not valid.any():
        return _empty_correction("no_valid_floor_intersection_at_contact_band", contact)

    original_median = float(np.median(original_depth[valid]))
    corrected_median = float(np.median(implied_depth[valid]))
    if original_median <= 0 or corrected_median <= 0:
        return _empty_correction("degenerate_median_depth", contact)
    # A single scalar depth-ratio shift, applied consistently to every
    # contact pixel's own ray (not an independent per-pixel refit) -- see
    # module docstring for why ratio (not an additive offset) was chosen:
    # it is scale-invariant along each ray's own parametrization, so it
    # generalizes smoothly across the whole mask even when the mask's
    # original (wrong) depths vary somewhat pixel to pixel.
    correction_ratio = corrected_median / original_median
    contact_std = float(np.std(implied_depth[valid]))

    ys, xs = np.nonzero(mask)
    row_span_fraction = float((ys.max() - ys.min() + 1) / mask.shape[0])
    whole_mask_implied = ray_plane_implied_depth(ys, xs, intrinsics, world_to_camera, floor)
    whole_finite = np.isfinite(whole_mask_implied) & (whole_mask_implied > 0)
    if whole_finite.any():
        spread = float(
            np.percentile(whole_mask_implied[whole_finite], 95)
            - np.percentile(whole_mask_implied[whole_finite], 5)
        )
    else:
        spread = None
    whole_mask_ok = (
        row_span_fraction <= row_span_fraction_limit
        and spread is not None
        and whole_finite.mean() > 0.5
        and spread <= depth_spread_relative_limit * max(corrected_median, 1e-6)
    )

    return GroundContactCorrection(
        succeeded=True,
        reason="ok" if whole_mask_ok else "whole_mask_sanity_failed_base_only_fallback",
        contact_pixels=contact,
        contact_pixel_count=int(len(contact)),
        original_median_contact_depth=original_median,
        corrected_median_contact_depth=corrected_median,
        correction_ratio=correction_ratio,
        contact_implied_depth_std=contact_std,
        whole_mask_row_span_fraction=row_span_fraction,
        whole_mask_implied_depth_spread=spread,
        whole_mask_applicable=whole_mask_ok,
        base_only=not whole_mask_ok,
    )


def apply_rigid_median_correction(
    points: np.ndarray,
    mask: np.ndarray,
    correction: GroundContactCorrection,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    own_depth_map: np.ndarray,
) -> np.ndarray:
    """Relocate every point under ``mask`` by the one rigid depth-ratio shift
    ``correction.correction_ratio``, each along its own camera ray.

    This exists to *mechanistically verify* the bug and the fix -- feed the
    output back through ``geometry.reverse_depth_filter`` and the previously
    invisible object should now register conflicts (reference case: 0% ->
    ~52% of the object's points). It is intentionally not used as either
    production downstream path: see ``exclude_from_keep`` (detection) and
    ``render_fill_for_mask`` (rendering) for those, and the module docstring
    for why relocation is the wrong operation for both.
    """
    if not correction.succeeded or correction.correction_ratio is None:
        raise ValueError("correction did not succeed; nothing to apply")
    points_out = np.asarray(points, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    original_depth = np.asarray(own_depth_map, dtype=np.float64)[ys, xs]
    valid = np.isfinite(original_depth) & (original_depth > 0)
    ys, xs, original_depth = ys[valid], xs[valid], original_depth[valid]
    if len(ys) == 0:
        return points_out
    new_depth = original_depth * correction.correction_ratio
    center = _camera_center_world(world_to_camera)
    rays = _pixel_rays_world(ys, xs, intrinsics, world_to_camera)
    points_out[ys, xs] = center[None, :] + new_depth[:, None] * rays
    return points_out


# ---------------------------------------------------------------------------
# Precondition gate: keep-fraction, computed against the *existing,
# unmodified* reverse_depth_filter output, before any of the (expensive)
# floor-fit work above runs.
# ---------------------------------------------------------------------------

# Calibrated against the 4 known cases from the prior investigation: forklift
# (84%) and cabinet (51%) keep-fraction were the actual target bug --
# self-consistent-wrong-depth, silently kept static by reverse_depth_filter
# -- and should run the correction. A shelf item (18%) and a different
# forklift instance (0%) were cases reverse_depth_filter already correctly
# rejected as conflicting (not this bug) and must be skipped. 0.45 sits
# roughly at the midpoint of the 51%/18% gap between the lowest known
# positive and the highest known negative, which is as far as 4 data points
# justify tuning it; not agonized over further per the spec.
APPLICABILITY_KEEP_FRACTION_THRESHOLD = 0.45


def promoted_candidate_keep_fraction(candidate_mask: np.ndarray, keep: np.ndarray) -> float:
    """Fraction of a promoted candidate's own points already silently kept
    static by the existing, unmodified ``reverse_depth_filter`` output --
    the applicability signal for this whole method (see module docstring).
    """
    region = np.asarray(candidate_mask, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    if region.shape != keep.shape:
        raise ValueError("candidate_mask and keep must share one pixel grid")
    if not region.any():
        return 0.0
    return float(keep[region].mean())


def ground_contact_gate_applicable(
    keep_fraction: float, threshold: float = APPLICABILITY_KEEP_FRACTION_THRESHOLD
) -> bool:
    """Whether the keep-fraction precondition clears the applicability gate."""
    return keep_fraction >= threshold


# ---------------------------------------------------------------------------
# The two production downstream operations. Both share the floor fit and the
# ray-plane primitives above; they must not be merged into one function --
# see module docstring for why detection and rendering need opposite
# treatment of the object's points.
# ---------------------------------------------------------------------------


def exclude_from_keep(keep: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Detection path: drop a ground-contact-flagged object's points from the
    trusted-static ``keep0``/``keep1`` set entirely.

    Never substitutes a floor-consistent point here -- that would manufacture
    false agreement between the two views and bury the very evidence a
    downstream detector needs to see. This is applied to the object's *whole*
    mask regardless of whether the whole-mask sanity check passed: even when
    the top of a tall object can't be confidently floor-projected for
    rendering, none of it should still count as confirmed-static once the
    gate has flagged it.
    """
    keep = np.asarray(keep, dtype=bool).copy()
    mask = np.asarray(mask, dtype=bool)
    if keep.shape != mask.shape:
        raise ValueError("keep and mask must share one pixel grid")
    keep[mask] = False
    return keep


def render_fill_for_mask(
    mask: np.ndarray,
    floor: FloorPlane,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    correction: GroundContactCorrection,
) -> tuple[np.ndarray, np.ndarray]:
    """Rendering path: synthetic floor-surface points to plug the hole left
    by deleting an object's points from the clean/static cloud, colored by
    the nearest real floor pixel (never a synthetic/uniform fill color) --
    a clean plate feeding downstream Difix3D novel-view synthesis must never
    show content that isn't really there.

    Fills the whole mask footprint when the per-object whole-mask sanity
    check passed (``correction.whole_mask_applicable``); otherwise falls
    back to filling only the contact band (``correction.base_only``), since
    projecting a tall object's upper body onto the floor plane would be
    physically wrong (it likely occludes a wall, not the floor).
    """
    if correction.base_only:
        pixels = correction.contact_pixels
        if len(pixels) == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        rows, cols = pixels[:, 0], pixels[:, 1]
    else:
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        rows, cols = ys, xs
    if len(rows) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    depth = ray_plane_implied_depth(rows, cols, intrinsics, world_to_camera, floor)
    valid = np.isfinite(depth) & (depth > 0)
    rows, cols, depth = rows[valid], cols[valid], depth[valid]
    if len(rows) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    center = _camera_center_world(world_to_camera)
    rays = _pixel_rays_world(rows, cols, intrinsics, world_to_camera)
    fill_points = center[None, :] + depth[:, None] * rays

    if floor.inlier_colors is not None and len(floor.inlier_points):
        tree = cKDTree(floor.inlier_points)
        _, nearest = tree.query(fill_points, k=1)
        fill_colors = np.clip(np.asarray(floor.inlier_colors)[nearest], 0, 255).astype(np.uint8)
    else:
        # No real floor color evidence available. Leave pixels uncolored
        # (black) rather than fabricate a synthetic uniform tint -- a
        # missing-color placeholder is auditable; a plausible-looking made
        # up color is not.
        fill_colors = np.zeros((len(fill_points), 3), dtype=np.uint8)
    return fill_points, fill_colors
