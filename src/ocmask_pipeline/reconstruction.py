"""Multi-view reconstruction of the T0 (before) and T1 (after) scene states
with VGGT-Omega, and the mutual depth-conflict filter that turns them into
render_t0 / clean_render / image_t1 for change_detection.py.

Unlike a two-photo (MASt3R-style) reconstruction, T0 and T1 are each *many*
frames of their own walkthrough, jointly posed by VGGT-Omega in one shared
world frame. That gives far denser coverage per side than a single photo can
(see docs/METHODS.md), which is what makes the depth-conflict filter below
effective: it can only prune a transient object if the opposing time step's
own reconstruction actually saw the true background behind it.

Call ``reconstruct_and_render`` with two lists of already-extracted frame
image paths (T0 frames, T1 frames) plus which index in each list is the
"reference" frame -- the T1 reference frame is the camera everything is
rendered into and becomes ``image_t1``; the T0 reference frame supplies the
opposing-view depth for T1's half of the conflict filter. Dataset-specific
frame extraction (video decoding, orientation correction, frame selection)
is intentionally not here -- see scripts/run_scenediff_pair.py for that.

Must run in a conda env with vggt-omega on the path (its own torch/CUDA pin
differs from the SAM3/DINOv2/SAM2 env change_detection.py runs in -- see
README.md).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _scene_scale(points: np.ndarray) -> float:
    """Characteristic spatial extent of a point cloud, in whatever arbitrary
    units this reconstruction call happened to land on (VGGT-Omega/MASt3R do
    not guarantee metric scale, and it can differ scene to scene). Used to
    make the geometric-identity test's distance thresholds scale-invariant
    (a fixed absolute threshold would be meaningless across scenes) -- see
    change_detection._geometric_matrix. The 90th-percentile radius around the
    centroid is a robust-to-outliers stand-in for "how big is this scene."
    """
    if points.size == 0:
        return 1.0
    centroid = points.mean(axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)
    scale = float(np.percentile(radii, 90))
    return scale if scale > 1e-9 else 1.0


@dataclass
class ReconstructionResult:
    render_t0: np.ndarray
    clean_render: np.ndarray
    image_t1: np.ndarray
    input_points: np.ndarray  # uncleaned T0 (aggregated, confidence-filtered), for inspection/viz
    input_colors: np.ndarray
    target_points: np.ndarray  # uncleaned T1
    target_colors: np.ndarray
    cleaned_points: np.ndarray  # canonical, depth-conflict-filtered union
    cleaned_colors: np.ndarray
    render_t0_coverage_fraction: float
    clean_render_coverage_fraction: float
    # Per-pixel world-space position for each of the three aligned frames
    # (NaN where unobserved/unrendered), for masking by a 2D object proposal
    # to get that object's own 3D point cloud -- see
    # change_detection._geometric_matrix. image_t1_positions is the query's
    # own directly-observed depth unprojection, not a render, so it has no
    # "hole" concept the way render_t0/clean_render's splatted renders do.
    render_t0_positions: np.ndarray | None = None
    clean_render_positions: np.ndarray | None = None
    image_t1_positions: np.ndarray | None = None
    scene_scale: float | None = None  # characteristic spatial extent of input_points, for scale-invariant thresholds
    alignment_residual: float | None = None  # see localize_and_render_query
    # Per-pixel boolean coverage for render_t0 (True where the z-buffer
    # splatted a real point, not a hole-fill), same grid as render_t0 itself
    # -- the binary visibility signal GOLDILOCS's (and SceneDiff's) own
    # visibility/occlusion filters use, for change_detection's visibility
    # filter to drop removed/added/moved decisions mostly supported by
    # unrendered pixels rather than real geometry.
    # render_t0_coverage_fraction above is this array's scalar mean; kept
    # separately since the filter needs the full per-pixel map, not just
    # the frame-level summary.
    render_t0_coverage: np.ndarray | None = None
    # Per-pixel depth confidence for render_t0 (NaN where uncovered), the
    # winning point's own VGGT-Omega depth_conf value splatted through the
    # same z-buffer as render_t0_positions. A continuous alternative to
    # render_t0_coverage's binary signal: GOLDILOCS and SceneDiff each
    # reconstruct from a single best-matching pair and only ever have a
    # binary rendered/not-rendered signal to filter on, whereas jointly
    # reconstructing many reference views gives a genuine per-pixel
    # confidence estimate from multi-view agreement -- discarded after
    # aggregation up to this point, otherwise. See change_detection's
    # confidence-weighted visibility filter.
    render_t0_confidence: np.ndarray | None = None
    # Per-pixel count (0..N) of how many of the reference scene's N
    # individual images -- rendered separately into the query view, never
    # aggregated with each other -- land within a small radius of the
    # *aggregate* render_t0's winning point at that pixel (see
    # localize_and_render_query). Unlike render_t0_confidence (a single
    # winning point's own internal VGGT-Omega uncertainty), this measures
    # cross-frame agreement: geometry only one or two near-degenerate
    # reference views ever independently reconstructed gets a low count even
    # if that one point's own confidence was high. Only populated by
    # localize_and_render_query (needs individual reference frames' own
    # world_points/depth_conf, which reconstruct_and_render's joint T0+T1
    # call does not keep separated). PASLCD-specific analogue of GOLDILOCS's
    # cross-query-view majority vote (docs/goldilocs_analysis.html roadmap
    # item 6), which needs multiple *query* photos PASLCD does not provide;
    # this substitutes the reference side's redundancy instead. See
    # change_detection's reference-corroboration filter.
    render_t0_corroboration: np.ndarray | None = None


def _unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx, fy = intrinsic[:, 0, 0][:, None, None], intrinsic[:, 1, 1][:, None, None]
    cx, cy = intrinsic[:, 0, 2][:, None, None], intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)), camera_points - translation[:, None, None, :])


def run_vggt_omega(
    image_paths: list[str | Path], config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Run VGGT-Omega jointly on every listed frame; returns dense per-frame
    world_points, colors (0-255 uint8), depth_conf, extrinsic, intrinsic --
    all indexed in the same order as ``image_paths``, in one shared world
    frame."""

    recon_cfg = config["reconstruction"]
    vggt_root = Path(recon_cfg["vggt_omega_root"])
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))

    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(recon_cfg["vggt_omega_checkpoint"], map_location="cpu"))

    images = load_and_preprocess_images(
        [str(p) for p in image_paths], image_resolution=int(recon_cfg["image_resolution"])
    ).to("cuda")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    out: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            out[key] = value

    out["world_points_from_depth"] = _unproject_depth_map_to_point_map(out["depth"], out["extrinsic"], out["intrinsic"])
    torch.cuda.empty_cache()

    rgb_images = np.transpose(out["images"], (0, 2, 3, 1))
    out["rgb"] = np.clip(rgb_images * 255.0, 0, 255).astype(np.uint8)
    out["depth_2d"] = out["depth"][..., 0]
    return out


def reconstruct_and_render(
    t0_image_paths: list[str | Path],
    t1_image_paths: list[str | Path],
    t0_reference_index: int,
    t1_reference_index: int,
    config: dict[str, Any],
) -> ReconstructionResult:
    """Reconstruct T0/T1 jointly, apply the mutual depth-conflict filter, and
    splat both the uncleaned T0 cloud and the cleaned canonical cloud into
    the T1 reference camera. See module docstring for the frame-list
    contract."""

    from .geometry import render_points, reverse_depth_filter

    recon_cfg = config["reconstruction"]
    conf_pct = float(recon_cfg["depth_confidence_percentile"])
    cleaned_conf_pct = float(recon_cfg["cleaned_depth_confidence_percentile"])
    depth_epsilon = float(recon_cfg["depth_epsilon"])
    min_valid_depth = float(recon_cfg["minimum_valid_depth"])

    predictions = run_vggt_omega(list(t0_image_paths) + list(t1_image_paths), config)
    num_t0 = len(t0_image_paths)
    world_points, colors, depth_conf, depth_2d = (
        predictions["world_points_from_depth"], predictions["rgb"], predictions["depth_conf"], predictions["depth_2d"]
    )
    height, width = colors.shape[1:3]
    image_shape = (height, width)
    conf_threshold = np.percentile(depth_conf, conf_pct)
    # Normalization anchors for the continuous confidence splatted into
    # render_t0_confidence (see render_points / ReconstructionResult):
    # conf_threshold is the bar a point must already clear to be aggregated
    # at all (0 on this scale), cleaned_threshold is this same pipeline's
    # own, already-tuned notion of genuinely high confidence -- the bar
    # cleaned_points itself requires (1 on this scale). Reusing these two
    # existing, already-meaningful percentiles avoids introducing a new
    # unmotivated absolute threshold on depth_conf's raw scale (which is
    # unbounded and model-specific: VGGT-Omega's confidence head produces
    # 1 + exp(logits), not a normalized [0, 1] value).
    cleaned_threshold = np.percentile(depth_conf, cleaned_conf_pct)
    confidence_scale = max(float(cleaned_threshold - conf_threshold), 1e-6)

    def aggregate_uncleaned(frame_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points, cols, conf = world_points[frame_slice], colors[frame_slice], depth_conf[frame_slice]
        mask = np.isfinite(points).all(axis=-1) & (conf >= conf_threshold)
        normalized_conf = np.clip((conf[mask] - conf_threshold) / confidence_scale, 0.0, 1.0)
        return points[mask], cols[mask], normalized_conf

    input_points, input_colors, input_confidence = aggregate_uncleaned(slice(0, num_t0))
    target_points, target_colors, _ = aggregate_uncleaned(slice(num_t0, len(world_points)))

    t0_ref, t1_ref = t0_reference_index, num_t0 + t1_reference_index
    t0_ref_depth, t0_ref_intrinsic, t0_ref_extrinsic = depth_2d[t0_ref], predictions["intrinsic"][t0_ref], predictions["extrinsic"][t0_ref]
    t1_ref_depth, t1_ref_intrinsic, t1_ref_extrinsic = depth_2d[t1_ref], predictions["intrinsic"][t1_ref], predictions["extrinsic"][t1_ref]

    def kept_cloud(frame_slice: slice, opposing_depth, opposing_intrinsic, opposing_extrinsic) -> tuple[np.ndarray, np.ndarray]:
        cleaned_threshold = np.percentile(depth_conf, cleaned_conf_pct)
        kept_points, kept_colors = [], []
        for i in range(frame_slice.start, frame_slice.stop):
            conf_ok = depth_conf[i] >= cleaned_threshold
            finite = np.isfinite(world_points[i]).all(axis=-1)
            geometric_keep = reverse_depth_filter(
                world_points[i], opposing_depth, opposing_intrinsic, opposing_extrinsic, depth_epsilon, min_valid_depth
            )
            keep = conf_ok & finite & geometric_keep
            kept_points.append(world_points[i][keep])
            kept_colors.append(colors[i][keep])
        return np.concatenate(kept_points), np.concatenate(kept_colors)

    kept_t0_points, kept_t0_colors = kept_cloud(slice(0, num_t0), t1_ref_depth, t1_ref_intrinsic, t1_ref_extrinsic)
    kept_t1_points, kept_t1_colors = kept_cloud(slice(num_t0, len(world_points)), t0_ref_depth, t0_ref_intrinsic, t0_ref_extrinsic)
    cleaned_points = np.concatenate([kept_t0_points, kept_t1_points])
    cleaned_colors = np.concatenate([kept_t0_colors, kept_t1_colors])

    render_kwargs = dict(
        z_epsilon=float(recon_cfg["z_buffer_epsilon"]),
        splat_radius=int(recon_cfg["splat_radius"]),
        fill_holes=bool(recon_cfg["hole_fill_enabled"]),
        hole_fill_min_neighbors=int(recon_cfg["hole_fill_min_neighbors"]),
        hole_fill_max_relative_depth=float(recon_cfg["hole_fill_max_relative_depth"]),
    )
    render_t0, _, coverage_t0, render_t0_positions, render_t0_confidence = render_points(
        input_points, input_colors, t1_ref_intrinsic, t1_ref_extrinsic, image_shape, point_confidence=input_confidence, **render_kwargs
    )
    clean_render, _, coverage_clean, clean_render_positions, _ = render_points(cleaned_points, cleaned_colors, t1_ref_intrinsic, t1_ref_extrinsic, image_shape, **render_kwargs)
    image_t1 = colors[t1_ref]

    return ReconstructionResult(
        render_t0=render_t0,
        clean_render=clean_render,
        image_t1=image_t1,
        input_points=input_points,
        input_colors=input_colors,
        target_points=target_points,
        target_colors=target_colors,
        cleaned_points=cleaned_points,
        cleaned_colors=cleaned_colors,
        render_t0_coverage_fraction=float(coverage_t0.mean()),
        clean_render_coverage_fraction=float(coverage_clean.mean()),
        render_t0_positions=render_t0_positions,
        clean_render_positions=clean_render_positions,
        image_t1_positions=world_points[t1_ref],
        scene_scale=_scene_scale(input_points),
        render_t0_coverage=coverage_t0,
        render_t0_confidence=render_t0_confidence,
    )


@dataclass
class ReferenceScene:
    """A reference ("before") scene reconstructed from its own image set in
    isolation -- no query/after frame in the batch. Used by
    ``localize_and_render_query`` to render the reference scene into a later
    query camera without that query ever having influenced the reference
    reconstruction itself (unlike ``reconstruct_and_render``, which puts T0
    and T1 in one joint VGGT-Omega call, so T0's own estimated geometry can
    be perturbed by T1's presence in the batch through the model's
    cross-attention -- see ``reconstruct_reference_scene``'s docstring)."""

    image_paths: list[Path]
    world_points: np.ndarray  # (N, H, W, 3), this call's own isolated world frame
    colors: np.ndarray  # (N, H, W, 3) uint8
    depth_conf: np.ndarray  # (N, H, W)
    extrinsic: np.ndarray  # (N, 3-or-4, 4)
    intrinsic: np.ndarray  # (N, 3, 3)

    def save(self, path: str | Path) -> None:
        """Persist to a single .npz, so an expensive reference reconstruction
        can be computed once (e.g. an overnight batch run) and reused across
        separate later processes/sessions instead of only living in memory
        for one run_paslcd_scene.py invocation."""
        np.savez_compressed(
            path,
            image_paths=np.array([str(p) for p in self.image_paths]),
            world_points=self.world_points,
            colors=self.colors,
            depth_conf=self.depth_conf,
            extrinsic=self.extrinsic,
            intrinsic=self.intrinsic,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceScene":
        with np.load(path) as data:
            return cls(
                image_paths=[Path(p) for p in data["image_paths"]],
                world_points=data["world_points"],
                colors=data["colors"],
                depth_conf=data["depth_conf"],
                extrinsic=data["extrinsic"],
                intrinsic=data["intrinsic"],
            )


def reconstruct_reference_scene(t0_image_paths: list[str | Path], config: dict[str, Any]) -> ReferenceScene:
    """Reconstruct the reference ("before") scene from its own image set
    alone. Computed once per scene and reused for every query via
    ``localize_and_render_query``, instead of re-running the full reference
    reconstruction jointly with each individual query as
    ``reconstruct_and_render`` does."""

    predictions = run_vggt_omega(list(t0_image_paths), config)
    return ReferenceScene(
        image_paths=[Path(p) for p in t0_image_paths],
        world_points=predictions["world_points_from_depth"],
        colors=predictions["rgb"],
        depth_conf=predictions["depth_conf"],
        extrinsic=predictions["extrinsic"],
        intrinsic=predictions["intrinsic"],
    )


def _camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    """World-space camera centers from a batch of world-to-camera extrinsics."""
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation)


def _fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama (1991) closed-form least-squares similarity transform: finds
    (rotation, translation, scale) minimizing ||target - (scale * rotation @
    source + translation)||^2. Used to align two independent VGGT-Omega
    calls' otherwise-arbitrary world frames using the camera poses of the
    reference images they share (see localize_and_render_query)."""
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError(f"expected two (N, 3) arrays of the same shape, got {source.shape} vs {target.shape}")
    n = source.shape[0]
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = (target_centered.T @ source_centered) / n
    u, singular_values, vt = np.linalg.svd(covariance)
    sign_correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign_correction[2, 2] = -1.0
    rotation = u @ sign_correction @ vt
    source_variance = (source_centered**2).sum(axis=1).mean()
    scale = float(np.trace(np.diag(singular_values) @ sign_correction) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    return rotation, translation, scale


def localize_and_render_query(
    t0_image_paths: list[str | Path],
    query_image_path: str | Path,
    reference_scene: ReferenceScene,
    t0_reference_index: int,
    config: dict[str, Any],
) -> ReconstructionResult:
    """Render ``reference_scene``'s isolated reconstruction into a query
    camera, without the query ever contributing to the reference geometry.

    ``t0_image_paths`` must be the exact same images, in the same order, used
    to build ``reference_scene``: this function jointly reconstructs
    ``t0_image_paths + [query_image_path]`` (needed to localize the
    otherwise-unposed query at all), then fits a similarity transform between
    that call's poses for the t0 images and reference_scene's own poses for
    the same images -- two independent VGGT-Omega calls do not share a
    coordinate frame or scale, only (up to noise) the same relative camera
    geometry -- and uses that transform to bring reference_scene's point
    cloud into the query's own frame. The query's own geometry, and the one
    T0 anchor frame used to clean the query's own points, come directly from
    this joint call and need no transform.

    The returned ``alignment_residual`` (mean camera-center error after
    alignment, in reference_scene's own world units) is a sanity check: a
    small residual means the two calls agree on the reference geometry's
    shape (the premise this function depends on); a large one means they
    don't and the result should not be trusted.
    """

    from .geometry import render_points, reverse_depth_filter

    recon_cfg = config["reconstruction"]
    conf_pct = float(recon_cfg["depth_confidence_percentile"])
    cleaned_conf_pct = float(recon_cfg["cleaned_depth_confidence_percentile"])
    depth_epsilon = float(recon_cfg["depth_epsilon"])
    min_valid_depth = float(recon_cfg["minimum_valid_depth"])

    num_t0 = len(t0_image_paths)
    if [Path(p) for p in t0_image_paths] != reference_scene.image_paths:
        raise ValueError("t0_image_paths must exactly match reference_scene.image_paths (same images, same order)")

    predictions = run_vggt_omega(list(t0_image_paths) + [query_image_path], config)
    world_points, colors, depth_conf, depth_2d = (
        predictions["world_points_from_depth"], predictions["rgb"], predictions["depth_conf"], predictions["depth_2d"]
    )
    height, width = colors.shape[1:3]
    image_shape = (height, width)

    # --- align reference_scene's isolated frame into this call's frame ---
    source_centers = _camera_centers(reference_scene.extrinsic)
    target_centers = _camera_centers(predictions["extrinsic"][:num_t0])
    rotation, translation, scale = _fit_similarity_transform(source_centers, target_centers)
    aligned_centers = scale * source_centers @ rotation.T + translation
    alignment_residual = float(np.linalg.norm(target_centers - aligned_centers, axis=1).mean())

    def align(points: np.ndarray) -> np.ndarray:
        return scale * points @ rotation.T + translation

    ref_conf_threshold = np.percentile(reference_scene.depth_conf, conf_pct)
    # See reconstruct_and_render's identical normalization for why: anchors
    # the continuous render_t0_confidence to this pipeline's own existing
    # confidence percentiles (0 = the aggregation bar, 1 = the stricter bar
    # cleaned_points already requires) instead of depth_conf's raw,
    # unbounded, model-specific scale.
    ref_cleaned_threshold = np.percentile(reference_scene.depth_conf, cleaned_conf_pct)
    ref_confidence_scale = max(float(ref_cleaned_threshold - ref_conf_threshold), 1e-6)
    ref_mask = np.isfinite(reference_scene.world_points).all(axis=-1) & (reference_scene.depth_conf >= ref_conf_threshold)
    input_points = align(reference_scene.world_points[ref_mask])
    input_colors = reference_scene.colors[ref_mask]
    input_confidence = np.clip((reference_scene.depth_conf[ref_mask] - ref_conf_threshold) / ref_confidence_scale, 0.0, 1.0)

    query_index = num_t0
    query_depth, query_intrinsic, query_extrinsic = depth_2d[query_index], predictions["intrinsic"][query_index], predictions["extrinsic"][query_index]

    target_conf_threshold = np.percentile(depth_conf[query_index], conf_pct)
    target_mask = np.isfinite(world_points[query_index]).all(axis=-1) & (depth_conf[query_index] >= target_conf_threshold)
    target_points, target_colors = world_points[query_index][target_mask], colors[query_index][target_mask]

    # T0 anchor frame for cleaning the query's own points -- already in this
    # call's (the query's) own frame, unlike reference_scene's stored poses.
    t0_ref_depth = depth_2d[t0_reference_index]
    t0_ref_intrinsic, t0_ref_extrinsic = predictions["intrinsic"][t0_reference_index], predictions["extrinsic"][t0_reference_index]

    # reverse_depth_filter operates on one (H, W, 3) frame at a time (same as
    # reconstruct_and_render's kept_cloud) -- reference_scene.world_points is
    # a batch of frames, so this must loop rather than pass the batch in directly.
    # (ref_cleaned_threshold computed above, alongside ref_conf_threshold.)
    kept_t0_points_list, kept_t0_colors_list = [], []
    for frame_index in range(reference_scene.world_points.shape[0]):
        frame_points = align(reference_scene.world_points[frame_index])
        conf_ok = reference_scene.depth_conf[frame_index] >= ref_cleaned_threshold
        finite = np.isfinite(frame_points).all(axis=-1)
        geometric_keep = reverse_depth_filter(
            frame_points, query_depth, query_intrinsic, query_extrinsic, depth_epsilon, min_valid_depth
        )
        keep = conf_ok & finite & geometric_keep
        kept_t0_points_list.append(frame_points[keep])
        kept_t0_colors_list.append(reference_scene.colors[frame_index][keep])
    kept_t0_points = np.concatenate(kept_t0_points_list)
    kept_t0_colors = np.concatenate(kept_t0_colors_list)

    query_cleaned_threshold = np.percentile(depth_conf[query_index], cleaned_conf_pct)
    query_conf_ok = depth_conf[query_index] >= query_cleaned_threshold
    query_finite = np.isfinite(world_points[query_index]).all(axis=-1)
    query_geometric_keep = reverse_depth_filter(
        world_points[query_index], t0_ref_depth, t0_ref_intrinsic, t0_ref_extrinsic, depth_epsilon, min_valid_depth
    )
    query_keep = query_conf_ok & query_finite & query_geometric_keep
    kept_query_points, kept_query_colors = world_points[query_index][query_keep], colors[query_index][query_keep]

    cleaned_points = np.concatenate([kept_t0_points, kept_query_points])
    cleaned_colors = np.concatenate([kept_t0_colors, kept_query_colors])

    render_kwargs = dict(
        z_epsilon=float(recon_cfg["z_buffer_epsilon"]),
        splat_radius=int(recon_cfg["splat_radius"]),
        fill_holes=bool(recon_cfg["hole_fill_enabled"]),
        hole_fill_min_neighbors=int(recon_cfg["hole_fill_min_neighbors"]),
        hole_fill_max_relative_depth=float(recon_cfg["hole_fill_max_relative_depth"]),
    )
    render_t0, _, coverage_t0, render_t0_positions, render_t0_confidence = render_points(
        input_points, input_colors, query_intrinsic, query_extrinsic, image_shape, point_confidence=input_confidence, **render_kwargs
    )

    # Cross-reference-view corroboration (docs/goldilocs_analysis.html
    # roadmap item 6). Render each reference image's own point cloud into
    # the query view in isolation -- never merged with any other reference
    # frame -- and count, per pixel, how many of those N independent renders
    # land within corroboration_radius of the aggregate render_t0's point
    # above. fill_holes is deliberately forced off for these per-frame
    # renders: corroboration must reflect what a frame actually observed,
    # not neighbor-interpolated fill, or a single sparse frame could
    # spuriously "corroborate" a large area it never really saw.
    scene_scale_value = _scene_scale(input_points)
    corroboration_radius = float(recon_cfg["corroboration_radius_fraction"]) * scene_scale_value
    corroboration_count = np.zeros(image_shape, dtype=np.uint8)
    per_frame_kwargs = dict(render_kwargs, fill_holes=False)
    for frame_index in range(reference_scene.world_points.shape[0]):
        frame_mask = (
            np.isfinite(reference_scene.world_points[frame_index]).all(axis=-1)
            & (reference_scene.depth_conf[frame_index] >= ref_conf_threshold)
        )
        if not frame_mask.any():
            continue
        frame_points = align(reference_scene.world_points[frame_index][frame_mask])
        frame_colors = reference_scene.colors[frame_index][frame_mask]
        _, _, frame_coverage, frame_positions, _ = render_points(
            frame_points, frame_colors, query_intrinsic, query_extrinsic, image_shape, **per_frame_kwargs
        )
        agrees = (
            frame_coverage
            & coverage_t0
            & (np.linalg.norm(frame_positions - render_t0_positions, axis=-1) <= corroboration_radius)
        )
        corroboration_count += agrees.astype(np.uint8)

    clean_render, _, coverage_clean, clean_render_positions, _ = render_points(cleaned_points, cleaned_colors, query_intrinsic, query_extrinsic, image_shape, **render_kwargs)
    image_t1 = colors[query_index]

    # This process runs two VGGT-Omega models sequentially (reference_scene's
    # own build, then this call) and stays alive afterward while refine.py /
    # detect.py run as subprocesses in other conda envs on the same GPU.
    # run_vggt_omega's own torch.cuda.empty_cache() call runs while its model
    # is still in scope, so it cannot release that model's memory -- do it
    # here instead, now that both calls' models are actually out of scope.
    import torch

    torch.cuda.empty_cache()

    return ReconstructionResult(
        render_t0=render_t0,
        clean_render=clean_render,
        image_t1=image_t1,
        input_points=input_points,
        input_colors=input_colors,
        target_points=target_points,
        target_colors=target_colors,
        cleaned_points=cleaned_points,
        cleaned_colors=cleaned_colors,
        render_t0_coverage_fraction=float(coverage_t0.mean()),
        clean_render_coverage_fraction=float(coverage_clean.mean()),
        alignment_residual=alignment_residual,
        render_t0_positions=render_t0_positions,
        clean_render_positions=clean_render_positions,
        image_t1_positions=world_points[query_index],
        scene_scale=scene_scale_value,
        render_t0_coverage=coverage_t0,
        render_t0_confidence=render_t0_confidence,
        render_t0_corroboration=corroboration_count,
    )
