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

    def aggregate_uncleaned(frame_slice: slice) -> tuple[np.ndarray, np.ndarray]:
        points, cols, conf = world_points[frame_slice], colors[frame_slice], depth_conf[frame_slice]
        mask = np.isfinite(points).all(axis=-1) & (conf >= conf_threshold)
        return points[mask], cols[mask]

    input_points, input_colors = aggregate_uncleaned(slice(0, num_t0))
    target_points, target_colors = aggregate_uncleaned(slice(num_t0, len(world_points)))

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
    render_t0, _, coverage_t0 = render_points(input_points, input_colors, t1_ref_intrinsic, t1_ref_extrinsic, image_shape, **render_kwargs)
    clean_render, _, coverage_clean = render_points(cleaned_points, cleaned_colors, t1_ref_intrinsic, t1_ref_extrinsic, image_shape, **render_kwargs)
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
    )
