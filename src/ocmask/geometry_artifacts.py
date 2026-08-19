from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import canonical_cloud, colorize_depth, render_points, write_ply
from .io import save_image
from .types import Reconstruction


def save_extended_geometry_artifacts(
    artifact_dir: str | Path,
    reconstruction: Reconstruction,
    keep0: np.ndarray,
    keep1: np.ndarray,
    geometry_config: dict,
    extra_render_points: np.ndarray | None = None,
    extra_render_colors: np.ndarray | None = None,
) -> None:
    """Save every pointmap/cloud/render named in the paper's pairwise pipeline.

    ``extra_render_points``/``extra_render_colors`` (both optional, and only
    meaningful together) are appended *only* to the canonical/star cloud
    used for ``render_clean_to_0/1.png`` -- the actual clean-plate renders
    this repo feeds to downstream Difix3D novel-view synthesis. They are
    deliberately not added to ``clean0``/``clean1`` (the per-view
    ``keep0``/``keep1`` debug dumps), which stay a faithful, unfilled record
    of what the detection-side ``keep0``/``keep1`` arrays actually kept --
    see ``ground_contact``'s module docstring for why the detection and
    rendering paths must not be merged.
    """
    artifact_dir = Path(artifact_dir)
    points0, points1 = reconstruction.points
    image0, image1 = reconstruction.images
    height0, width0 = points0.shape[:2]
    height1, width1 = points1.shape[:2]
    if image0.shape[:2] != (height0, width0) or image1.shape[:2] != (height1, width1):
        raise ValueError("Reconstruction colors and pointmaps must share image dimensions")

    clean0_points, clean0_colors = points0[keep0], image0[keep0]
    clean1_points, clean1_colors = points1[keep1], image1[keep1]
    star_points, star_colors = canonical_cloud(
        points0, image0, keep0, points1, image1, keep1
    )
    if extra_render_points is not None and len(extra_render_points):
        star_points = np.concatenate([star_points, np.asarray(extra_render_points)])
        star_colors = np.concatenate([star_colors, np.asarray(extra_render_colors)])

    # PLY files retain actual XYZ coordinates; PNG previews expose depth and
    # retained appearance directly inside the static HTML report.
    write_ply(str(artifact_dir / "pointmap_0.ply"), points0.reshape(-1, 3), image0.reshape(-1, 3))
    write_ply(str(artifact_dir / "pointmap_1.ply"), points1.reshape(-1, 3), image1.reshape(-1, 3))
    write_ply(str(artifact_dir / "clean_pointcloud_0.ply"), clean0_points, clean0_colors)
    write_ply(str(artifact_dir / "clean_pointcloud_1.ply"), clean1_points, clean1_colors)
    write_ply(str(artifact_dir / "canonical.ply"), star_points, star_colors)
    save_image(artifact_dir / "pointmap_0.png", colorize_depth(reconstruction.depths[0]))
    save_image(artifact_dir / "pointmap_1.png", colorize_depth(reconstruction.depths[1]))
    save_image(artifact_dir / "clean_pointcloud_0.png", image0 * keep0[..., None])
    save_image(artifact_dir / "clean_pointcloud_1.png", image1 * keep1[..., None])

    render_args = {"z_epsilon": geometry_config["z_buffer_epsilon"]}
    cross_view_args = {
        **render_args,
        "splat_radius": geometry_config.get("splat_radius", 0),
        "fill_holes": geometry_config.get("hole_fill_enabled", False),
        "hole_fill_min_neighbors": geometry_config.get(
            "hole_fill_min_neighbors", 5
        ),
        "hole_fill_max_relative_depth": geometry_config.get(
            "hole_fill_max_relative_depth", 0.02
        ),
    }
    renders = {
        "render_0_to_1.png": render_points(
            points0,
            image0,
            reconstruction.intrinsics[1],
            reconstruction.world_to_camera[1],
            (height1, width1),
            **cross_view_args,
        )[0],
        "render_1_to_0.png": render_points(
            points1,
            image1,
            reconstruction.intrinsics[0],
            reconstruction.world_to_camera[0],
            (height0, width0),
            **cross_view_args,
        )[0],
        "render_clean_to_1.png": render_points(
            star_points,
            star_colors,
            reconstruction.intrinsics[1],
            reconstruction.world_to_camera[1],
            (height1, width1),
            **render_args,
        )[0],
        "render_clean_to_0.png": render_points(
            star_points,
            star_colors,
            reconstruction.intrinsics[0],
            reconstruction.world_to_camera[0],
            (height0, width0),
            **render_args,
        )[0],
    }
    for filename, image in renders.items():
        save_image(artifact_dir / filename, image)
