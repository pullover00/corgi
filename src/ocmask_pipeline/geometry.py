from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass
class Projection:
    uv: np.ndarray
    depth: np.ndarray
    valid: np.ndarray


def transform_points(points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    """Transform arbitrary-shaped world-coordinate points into camera space."""
    shape = points.shape
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate([flat, np.ones((len(flat), 1), dtype=flat.dtype)], axis=1)
    transformed = homogeneous @ np.asarray(world_to_camera).T
    return transformed[..., :3].reshape(shape)


def project_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    image_shape: tuple[int, int],
    minimum_depth: float = 1e-6,
) -> Projection:
    """Project world points to integer pixels and mark valid in-frame samples."""
    camera = transform_points(points, world_to_camera).reshape(-1, 3)
    projected = camera @ np.asarray(intrinsics).T
    z = camera[:, 2]
    # Homogeneous division is expected to produce invalid values for points on
    # the camera plane; the validity mask below removes them.
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_float = projected[:, :2] / projected[:, 2:3]
    uv = np.rint(uv_float).astype(np.int64)
    height, width = image_shape
    valid = (
        np.isfinite(uv_float).all(axis=1)
        & np.isfinite(z)
        & (z > minimum_depth)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return Projection(uv=uv, depth=z, valid=valid)


def reverse_depth_filter(
    points: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsics: np.ndarray,
    target_world_to_camera: np.ndarray,
    depth_epsilon: float = 1e-3,
    minimum_depth: float = 1e-6,
) -> np.ndarray:
    """Keep points that do not sit in front of the target view's observed surface.

    Points outside the target frustum are retained; visibility filtering handles them
    later. Invalid source points are rejected.
    """
    flat = points.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    projection = project_points(
        flat, target_intrinsics, target_world_to_camera, target_depth.shape, minimum_depth
    )
    keep = finite.copy()
    indices = np.flatnonzero(projection.valid)
    if len(indices):
        uv = projection.uv[indices]
        observed = target_depth[uv[:, 1], uv[:, 0]]
        observed_valid = np.isfinite(observed) & (observed > minimum_depth)
        conflict = observed_valid & (projection.depth[indices] < observed - depth_epsilon)
        keep[indices[conflict]] = False
    return keep.reshape(points.shape[:2])


# Per-point outcome categories returned by confidence_gated_depth_filter's
# DepthFilterResult.status, kept as plain ints so the array stays a compact
# uint8 rather than an object array of enums.
STATUS_REJECTED_NONFINITE = 0
STATUS_CONFLICT = 1
STATUS_CONFIRMED_STATIC = 2
STATUS_KEPT_BY_OWN_CONFIDENCE = 3
STATUS_REJECTED_LOW_OWN_CONFIDENCE = 4


@dataclass
class DepthFilterResult:
    keep: np.ndarray
    status: np.ndarray


def confidence_gated_depth_filter(
    points: np.ndarray,
    point_confidence: np.ndarray,
    target_depth: np.ndarray,
    target_confidence: np.ndarray,
    target_intrinsics: np.ndarray,
    target_world_to_camera: np.ndarray,
    depth_epsilon: float = 1e-3,
    minimum_depth: float = 1e-6,
    opposing_confidence_threshold: float | None = None,
    own_confidence_threshold: float | None = None,
) -> DepthFilterResult:
    """Like ``reverse_depth_filter``, but distrust an absent opposing observation.

    ``reverse_depth_filter`` keeps a point by default whenever the opposing
    view offers no reliable evidence either way (out of frustum, or its
    reprojected pixel has no valid depth). That silent "keep" is exactly what
    lets changed-region geometry survive into the canonical reconstruction
    when the opposing camera simply never got a clear look at the region.

    This function keeps the same conflict test -- reliable opposing evidence
    (finite depth, and confidence at or above ``opposing_confidence_threshold``
    when that threshold is set) still decides the point outright, exactly as
    today: nearer than observed -> reject, otherwise -> confirm. The only
    change is what happens when there is *no* reliable opposing evidence: the
    point falls back to a decision based on its *own* confidence rather than
    being kept unconditionally.

    With both thresholds left at their default of ``None`` (meaning "off"),
    every point that has no reliable opposing evidence is kept, and ``.keep``
    is bit-for-bit identical to ``reverse_depth_filter`` on the same inputs.
    """
    flat = points.reshape(-1, 3)
    flat_confidence = np.asarray(point_confidence, dtype=np.float32).reshape(-1)
    finite = np.isfinite(flat).all(axis=1)
    projection = project_points(
        flat, target_intrinsics, target_world_to_camera, target_depth.shape, minimum_depth
    )

    keep = finite.copy()
    status = np.where(finite, STATUS_KEPT_BY_OWN_CONFIDENCE, STATUS_REJECTED_NONFINITE).astype(
        np.uint8
    )
    if own_confidence_threshold is not None:
        low_own = finite & (flat_confidence < own_confidence_threshold)
        keep[low_own] = False
        status[low_own] = STATUS_REJECTED_LOW_OWN_CONFIDENCE

    indices = np.flatnonzero(projection.valid)
    if len(indices):
        uv = projection.uv[indices]
        observed_depth = target_depth[uv[:, 1], uv[:, 0]]
        depth_ok = np.isfinite(observed_depth) & (observed_depth > minimum_depth)
        if opposing_confidence_threshold is not None:
            observed_confidence = np.asarray(target_confidence, dtype=np.float32)[
                uv[:, 1], uv[:, 0]
            ]
            confidence_ok = np.isfinite(observed_confidence) & (
                observed_confidence >= opposing_confidence_threshold
            )
        else:
            confidence_ok = np.ones(len(indices), dtype=bool)
        reliable = depth_ok & confidence_ok
        conflict = reliable & (projection.depth[indices] < observed_depth - depth_epsilon)
        confirmed = reliable & ~conflict
        keep[indices[confirmed]] = True
        status[indices[confirmed]] = STATUS_CONFIRMED_STATIC
        keep[indices[conflict]] = False
        status[indices[conflict]] = STATUS_CONFLICT

    return DepthFilterResult(
        keep=keep.reshape(points.shape[:2]), status=status.reshape(points.shape[:2])
    )


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    image_shape: tuple[int, int],
    background: int = 0,
    z_epsilon: float = 1e-4,
    splat_radius: int = 0,
    fill_holes: bool = False,
    hole_fill_min_neighbors: int = 5,
    hole_fill_max_relative_depth: float = 0.02,
    point_confidence: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render colored points with a deterministic nearest-depth z-buffer.

    ``splat_radius`` expands every projected sample to a square footprint. Each
    destination pixel still uses the nearest sample, so foreground points
    correctly occlude background splats. Optional hole filling is deliberately
    conservative: it fills only one-pixel holes surrounded by enough rendered
    neighbors whose depths describe the same local surface.

    Returns ``(image, depth, coverage, world_positions, confidence)``:
    ``world_positions`` is the (H, W, 3) world-space coordinate of whichever
    input point won the z-buffer test at each pixel (NaN where uncovered),
    for callers that need to know *where in 3D* a rendered pixel actually
    came from -- e.g. masking it by a 2D object proposal to get that
    object's own 3D point cloud. ``confidence`` is that winning point's own
    ``point_confidence`` value (NaN where uncovered, or wherever
    ``point_confidence`` was not supplied) -- e.g. the upstream multi-view
    reconstruction model's own per-point depth confidence, for a continuous
    alternative to ``coverage``'s binary rendered/not-rendered signal (see
    change_detection's visibility filter).
    """
    if splat_radius < 0:
        raise ValueError("splat_radius must be non-negative")
    if not 1 <= hole_fill_min_neighbors <= 8:
        raise ValueError("hole_fill_min_neighbors must be between 1 and 8")
    if hole_fill_max_relative_depth < 0:
        raise ValueError("hole_fill_max_relative_depth must be non-negative")

    height, width = image_shape
    flat_points = points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    flat_confidence = point_confidence.reshape(-1) if point_confidence is not None else None
    projection = project_points(flat_points, intrinsics, world_to_camera, image_shape)
    valid_idx = np.flatnonzero(projection.valid & np.isfinite(flat_colors).all(axis=1))
    image = np.full((height, width, 3), background, dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=bool)
    world_positions = np.full((height, width, 3), np.nan, dtype=np.float32)
    confidence = np.full((height, width), np.nan, dtype=np.float32)
    if not len(valid_idx):
        return image, depth, coverage, world_positions, confidence

    uv = projection.uv[valid_idx]
    z = projection.depth[valid_idx]

    # Projected dense pointmaps develop a regular lattice when camera motion
    # spreads adjacent samples farther than one target pixel. A small splat
    # footprint covers those sub-pixel gaps before z-buffer collision handling.
    if splat_radius:
        base_idx = valid_idx
        base_uv = uv
        offsets = np.array(
            [
                (dx, dy)
                for dy in range(-splat_radius, splat_radius + 1)
                for dx in range(-splat_radius, splat_radius + 1)
            ],
            dtype=np.int64,
        )
        uv = (base_uv[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        valid_idx = np.repeat(base_idx, len(offsets))
        z = np.repeat(z, len(offsets))
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        uv, valid_idx, z = uv[inside], valid_idx[inside], z[inside]

    pixel = uv[:, 1] * width + uv[:, 0]
    # Stable sorting makes equal-depth collisions deterministic.
    order = np.lexsort((valid_idx, z, pixel))
    pixel_sorted = pixel[order]
    first = np.r_[True, pixel_sorted[1:] != pixel_sorted[:-1]]
    winners = valid_idx[order[first]]
    # Use the expanded destination coordinates rather than the original
    # projection coordinates when splatting is enabled.
    win_uv = uv[order[first]]
    win_z = z[order[first]]
    image[win_uv[:, 1], win_uv[:, 0]] = np.clip(flat_colors[winners], 0, 255).astype(np.uint8)
    depth[win_uv[:, 1], win_uv[:, 0]] = win_z
    coverage[win_uv[:, 1], win_uv[:, 0]] = True
    world_positions[win_uv[:, 1], win_uv[:, 0]] = flat_points[winners]
    if flat_confidence is not None:
        confidence[win_uv[:, 1], win_uv[:, 0]] = flat_confidence[winners]

    if fill_holes:
        image, depth, coverage = fill_small_render_holes(
            image,
            depth,
            coverage,
            minimum_neighbors=hole_fill_min_neighbors,
            maximum_relative_depth_range=hole_fill_max_relative_depth,
            depth_epsilon=z_epsilon,
        )
        # Hole-filled pixels get no world position or confidence: they were
        # never actually observed, only interpolated for a visually-complete
        # render, and a fabricated value would be actively misleading for
        # geometric identity matching (change_detection._geometric_matrix)
        # or the visibility filter (change_detection._visible_fraction).
    return image, depth, coverage, world_positions, confidence


def fill_small_render_holes(
    image: np.ndarray,
    depth: np.ndarray,
    coverage: np.ndarray,
    minimum_neighbors: int = 5,
    maximum_relative_depth_range: float = 0.02,
    depth_epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill isolated render holes without crossing depth discontinuities.

    Only the original 8-neighborhood is inspected and filling runs once. This
    prevents a valid region from growing into large unseen areas. A candidate
    must have ``minimum_neighbors`` covered neighbors, and their depth range
    must be small relative to their median depth.
    """
    image_out = np.asarray(image).copy()
    depth_out = np.asarray(depth, dtype=np.float32).copy()
    coverage_out = np.asarray(coverage, dtype=bool).copy()
    height, width = coverage_out.shape

    neighbor_depths = []
    neighbor_colors = []
    neighbor_valid = []
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ):
        shifted_depth = np.full((height, width), np.nan, dtype=np.float32)
        shifted_color = np.zeros((height, width, 3), dtype=image_out.dtype)
        shifted_valid = np.zeros((height, width), dtype=bool)

        source_y = slice(max(0, -dy), min(height, height - dy))
        source_x = slice(max(0, -dx), min(width, width - dx))
        target_y = slice(max(0, dy), min(height, height + dy))
        target_x = slice(max(0, dx), min(width, width + dx))
        shifted_depth[target_y, target_x] = depth_out[source_y, source_x]
        shifted_color[target_y, target_x] = image_out[source_y, source_x]
        shifted_valid[target_y, target_x] = coverage_out[source_y, source_x]
        neighbor_depths.append(shifted_depth)
        neighbor_colors.append(shifted_color)
        neighbor_valid.append(shifted_valid)

    valid_stack = np.stack(neighbor_valid)
    depth_stack = np.stack(neighbor_depths)
    color_stack = np.stack(neighbor_colors)
    valid_count = valid_stack.sum(axis=0)
    masked_depth = np.where(valid_stack, depth_stack, np.nan)
    # NumPy warns for pixels with no covered neighbors. Those pixels are
    # explicitly rejected by ``valid_count`` below, so the NaNs are expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_depth = np.nanmedian(masked_depth, axis=0)
        depth_range = np.nanmax(masked_depth, axis=0) - np.nanmin(masked_depth, axis=0)
    relative_limit = maximum_relative_depth_range * np.maximum(
        np.abs(median_depth), depth_epsilon
    )
    fill = (
        ~coverage_out
        & (valid_count >= minimum_neighbors)
        & np.isfinite(median_depth)
        & (depth_range <= relative_limit + depth_epsilon)
    )
    if not np.any(fill):
        return image_out, depth_out, coverage_out

    # Median RGB is robust to a single differently colored neighbor while the
    # depth-consistency test above protects actual geometry boundaries.
    masked_colors = np.where(valid_stack[..., None], color_stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_color = np.nanmedian(masked_colors, axis=0)
    image_out[fill] = np.clip(median_color[fill], 0, 255).astype(image_out.dtype)
    depth_out[fill] = median_depth[fill]
    coverage_out[fill] = True
    return image_out, depth_out, coverage_out


def canonical_cloud(
    points0: np.ndarray,
    image0: np.ndarray,
    keep0: np.ndarray,
    points1: np.ndarray,
    image1: np.ndarray,
    keep1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine the two conflict-filtered pointmaps and their RGB observations."""
    p0, c0 = points0[keep0], image0[keep0]
    p1, c1 = points1[keep1], image1[keep1]
    return np.concatenate([p0, p1]), np.concatenate([c0, c1])


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Create a robust false-color preview of a pointmap's camera-space depth."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return output
    low, high = np.percentile(depth[valid], (2, 98))
    scale = max(float(high - low), np.finfo(np.float32).eps)
    value = np.clip((depth - low) / scale, 0, 1)
    # Compact blue→cyan→yellow→red map without a plotting dependency.
    output[..., 0] = np.clip(1.5 - np.abs(4 * value - 3), 0, 1) * 255
    output[..., 1] = np.clip(1.5 - np.abs(4 * value - 2), 0, 1) * 255
    output[..., 2] = np.clip(1.5 - np.abs(4 * value - 1), 0, 1) * 255
    output[~valid] = 0
    return output


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a dependency-free ASCII point cloud for visual inspection."""
    valid = np.isfinite(points).all(axis=1)
    pts = points[valid]
    rgb = np.clip(colors[valid], 0, 255).astype(np.uint8)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "w", encoding="ascii") as handle:
        handle.write(header)
        np.savetxt(handle, np.column_stack([pts, rgb]), fmt="%.7g %.7g %.7g %d %d %d")
